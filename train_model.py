import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the data you just collected
df = pd.read_csv('myanmar_air_data.csv')
# We will focus on predicting PM2.5 levels
data = df['pm2_5'].values.reshape(-1, 1)

# 2. Scale the data (AI works best with numbers between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create 'Sequences' (Look back at the last 24 hours to predict the next 1)
def create_sequences(data, window=24):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(x), np.array(y)

X, y = create_sequences(scaled_data)

# 4. Build the LSTM Architecture
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Start Training!
print("🚀 Training the Lay Htu brain... this might take a minute.")
model.fit(X, y, epochs=20, batch_size=32)

# 6. Save the model for your IntelliJ Backend
model.save('lay_htu_model.h5')
print("✅ Done! 'lay_htu_model.h5' is ready to predict Myanmar's air.")