import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Load the model you just trained
model = load_model('lay_htu_model.h5')

# 2. Load the data to get the right scaling
df = pd.read_csv('myanmar_air_data.csv')
data = df['pm2_5'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)

# 3. Get the most recent 24 hours to predict the 25th
last_24_hours = data[-24:]
scaled_input = scaler.transform(last_24_hours).reshape(1, 24, 1)

# 4. Predict!
prediction_scaled = model.predict(scaled_input)
prediction_real = scaler.inverse_transform(prediction_scaled)

print("-" * 30)
print(f"🌍 Current PM2.5: {data[-1][0]}")
print(f"🔮 AI Prediction for next hour: {prediction_real[0][0]:.2f}")
print("-" * 30)