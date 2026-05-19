"""Production FastAPI app with a lightweight Agno-style multi-agent router.

This module loads the TensorFlow model and a pre-fitted MinMaxScaler on startup
and exposes two endpoints:
 - POST /api/v1/predict  -> accepts exactly 24 hourly readings and returns a prediction
 - POST /api/v1/chat     -> accepts a string and routes it to a specialist agent team

Notes:
 - Uses Gemini(id="gemini-2.5-flash") as the preferred AI engine when the `agno` package
   is available. If `agno` is not installed, the module falls back to deterministic
   template-based agent handlers so the service remains functional.
 - To maximize runtime speed and RAM efficiency, this file does NOT load any CSVs or
   Pandas data on startup. The model and scaler are loaded globally.
"""

from typing import List
import logging
import pickle
import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

try:
    # Prefer the real Agno / Gemini integration if available in production
    from agno import Agno, Agent, Gemini  # type: ignore
    AGNO_AVAILABLE = True
except Exception:
    AGNO_AVAILABLE = False

from tensorflow.keras.models import load_model

app = FastAPI(title='Lay Htu AI - Production API')
logger = logging.getLogger('layhtu')
logger.setLevel(logging.INFO)

# Globals populated at startup
MODEL = None
SCALER = None
AGENTS = {}


class PredictPayload(BaseModel):
    readings: conlist(float, min_items=24, max_items=24)


class ChatPayload(BaseModel):
    message: str


class AgentBase:
    """A small local fallback agent abstraction used when `agno` isn't available.

    Each agent exposes an async `respond(message: str) -> str` method.
    """

    def __init__(self, name: str):
        self.name = name

    async def respond(self, message: str) -> str:
        # override in subclasses
        return f"{self.name} received: {message}"


class EcoAgentLocal(AgentBase):
    async def respond(self, message: str) -> str:
        # Professional, scientific tone
        return (
            "**EcoAgent (scientific):**\n\n"
            "Myanmar's PM2.5 is influenced by combustion sources, biomass burning, and weather patterns. "
            "This system provides hourly forecasting based on the last 24-hour sequence."
        )


class TechAgentLocal(AgentBase):
    async def respond(self, message: str) -> str:
        # Explain architecture to software engineers
        return (
            "**TechAgent (engineering):**\n\n"
            "The forecasting core is a custom TensorFlow LSTM model saved as 'lay_htu_model.h5'. "
            "At inference time we read a look-back sequence of the last 24 hours, reshape it to "
            "(1, 24, 1), normalize it using a pre-saved MinMaxScaler matrix loaded from 'scaler.pkl', "
            "and call the model's `predict()` API. This service is designed to integrate with a Java Spring Boot "
            "orchestrator and MongoDB storage layer via a simple JSON REST contract; the model and scaler are loaded "
            "globally on startup to minimize per-request overhead."
        )


class DoctorAgentLocal(AgentBase):
    async def respond(self, message: str) -> str:
        # Always format with the exact two headers and a disclaimer
        what_to_do = (
            "## 💡 WHAT TO DO\n"
            "- Monitor air quality alerts and reduce outdoor activity when PM2.5 is elevated.\n"
            "- Use HEPA-capable air purifiers indoors and wear well-fitting N95/FFP2 masks when exposure is unavoidable.\n"
            "- Ensure good ventilation when burning or cooking and avoid heavy exertion outdoors during peaks.\n"
        )
        what_to_avoid = (
            "## ❌ WHAT TO AVOID\n"
            "- Avoid prolonged outdoor exercise during high PM2.5 periods.\n"
            "- Avoid indoor smoking or other combustion that increases particulate levels.\n"
        )
        disclaimer = (
            "\n---\n"
            "*Medical disclaimer: This information is general and not a substitute for professional medical advice. "
            "Consult a licensed healthcare professional for personal guidance.*"
        )
        return f"{what_to_do}\n{what_to_avoid}{disclaimer}"


class SystemRouter:
    """Simple intent router that dispatches messages to the specialist agents.

    If a production `agno` integration is present the router will delegate to those agents; otherwise
    it uses the local fallbacks above.
    """

    def __init__(self, agents: dict):
        self.agents = agents

    async def route(self, message: str) -> str:
        text = message.lower()
        # Simple keyword-based intent detection
        medical_keywords = ['health', 'doctor', 'hospital', 'symptom', 'breath', 'asthma', 'pulmon']
        tech_keywords = ['api', 'spring', 'java', 'model', 'tensor', 'lstm', 'scaler', 'mongo', 'architecture']
        env_keywords = ['pm2.5', 'air', 'pollution', 'pollut', 'smoke', 'environment']

        if any(k in text for k in medical_keywords):
            agent = self.agents.get('doctor')
        elif any(k in text for k in tech_keywords):
            agent = self.agents.get('tech')
        elif any(k in text for k in env_keywords):
            agent = self.agents.get('eco')
        else:
            # Default to EcoAgent for environmental queries
            agent = self.agents.get('eco')

        if agent is None:
            raise HTTPException(status_code=500, detail='No agent available')

        # Agents may be async; await their response
        return await agent.respond(message)


@app.on_event('startup')
def startup_event():
    global MODEL, SCALER, AGENTS
    logger.info('Starting up: loading model and scaler into memory')

    try:
        MODEL = load_model('lay_htu_model.h5')
    except Exception as e:
        logger.exception('Failed to load model: %s', e)
        raise

    try:
        with open('scaler.pkl', 'rb') as f:
            SCALER = pickle.load(f)
    except Exception as e:
        logger.exception('Failed to load scaler: %s', e)
        raise

    # Initialize agents; prefer Agno/Gemini integration if available
    if AGNO_AVAILABLE:
        try:
            gemini = Gemini(id='gemini-2.5-flash')
            # Create Agno agent instances configured with role instructions
            eco = Agent(name='EcoAgent', model=gemini, system_prompt='Focus on environmental guidance for Myanmar. Use professional tone.')
            tech = Agent(name='TechAgent', model=gemini, system_prompt='Explain system architecture to software engineering peers.')
            doctor = Agent(name='DoctorAgent', model=gemini, system_prompt='Provide medical preventative guidance. Always format with headers and add disclaimer.')
            AGENTS = {'eco': eco, 'tech': tech, 'doctor': doctor}
            logger.info('Agno/Gemini agents initialized')
        except Exception:
            logger.exception('Failed to initialize Agno agents; falling back to local agents')
            AGENTS = {
                'eco': EcoAgentLocal('EcoAgent'),
                'tech': TechAgentLocal('TechAgent'),
                'doctor': DoctorAgentLocal('DoctorAgent'),
            }
    else:
        AGENTS = {
            'eco': EcoAgentLocal('EcoAgent'),
            'tech': TechAgentLocal('TechAgent'),
            'doctor': DoctorAgentLocal('DoctorAgent'),
        }

    logger.info('Startup complete')


@app.post('/api/v1/predict')
async def predict(payload: PredictPayload):
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail='Model or scaler not loaded')

    readings = np.array(payload.readings, dtype=float).reshape(-1, 1)

    if readings.shape[0] != 24:
        raise HTTPException(status_code=400, detail='Exactly 24 hourly readings are required')

    try:
        scaled = SCALER.transform(readings)
    except Exception as e:
        logger.exception('Scaler transform failed: %s', e)
        raise HTTPException(status_code=500, detail='Scaling error')

    model_input = scaled.reshape(1, 24, 1).astype(np.float32)

    try:
        pred = MODEL.predict(model_input)
    except Exception as e:
        logger.exception('Model prediction failed: %s', e)
        raise HTTPException(status_code=500, detail='Prediction error')

    # Inverse transform back to original units
    pred_value = SCALER.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]

    return {'prediction': float(pred_value)}


@app.post('/api/v1/chat')
async def chat(payload: ChatPayload):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail='Message must be non-empty')

    router = SystemRouter(AGENTS)
    response = await router.route(payload.message)

    # Return the raw markdown / text response so clients can render it
    return {'response': response}


if __name__ == '__main__':
    # Minimal developer run helper
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=8000, log_level='info')
