import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# 1. Load data, model, and the pre-fitted scaler
df = pd.read_csv('myanmar_air_data.csv')
model = load_model('lay_htu_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Prepare the data (use the pre-fitted scaler; DO NOT re-fit)
data = df['pm2_5'].values.reshape(-1, 1).astype(float)
scaled_data = scaler.transform(data)

# 3. Create a test set consisting of the last 100 hours
test_inputs = []
start = max(0, len(scaled_data) - 24 - 100)
end = len(scaled_data) - 24
for i in range(start, end):
    test_inputs.append(scaled_data[i:i + 24, 0])

test_inputs = np.array(test_inputs)
test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], test_inputs.shape[1], 1))

# 4. Predict and inverse transform
predictions = model.predict(test_inputs)
predictions = scaler.inverse_transform(predictions)
actual_values = data[-predictions.shape[0]:]

# 5. Plot with enhanced styles
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(12, 6))
plt.plot(actual_values, color='blue', linestyle='-', label='Real Myanmar Air Quality (PM2.5)')
plt.plot(predictions, color='red', linestyle='--', label='Lay Htu AI Prediction')
plt.title('Lay Htu AI: Prediction vs Reality')
plt.xlabel('Time (Hours)')
plt.ylabel('PM2.5 Level (µg/m³)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: save a rendered asset for dashboards
# plt.savefig('layhtu_prediction.png', dpi=200)
