import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_sequences(data, window=24):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i + window, 0])
        y.append(data[i + window, 0])
    x = np.array(x)
    y = np.array(y)
    # reshape to [samples, timesteps, features]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def build_model(input_shape):
    # Keep the exact LSTM architecture requested
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main(csv_path='myanmar_air_data.csv', model_path='lay_htu_model.h5', scaler_path='scaler.pkl'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV data not found: {csv_path}")

    print('🔁 Loading data...')
    df = pd.read_csv(csv_path)
    if 'pm2_5' not in df.columns:
        raise KeyError("Expected column 'pm2_5' in CSV")

    data = df['pm2_5'].values.reshape(-1, 1).astype(float)

    # Fit a MinMaxScaler and IMPORTANTLY export it for production use
    print('⚖️  Fitting MinMaxScaler (will be exported to scaler.pkl)')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences: last 24 hours -> next hour
    X, y = create_sequences(scaled_data, window=24)

    print('🏗️  Building model...')
    model = build_model((X.shape[1], 1))

    print('🚀 Training the model. This may take a while depending on your hardware...')
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)

    # Save model and scaler for production use
    print(f'💾 Saving model to {model_path}')
    model.save(model_path)

    print(f'💾 Saving fitted scaler to {scaler_path}')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print('✅ Training complete. Saved model and scaler.')


if __name__ == '__main__':
    main()

