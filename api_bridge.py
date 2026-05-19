import sys
from tensorflow.keras.models import load_model
import numpy as np

# Load model once (expensive operation)
model = load_model('lay_htu_model.h5')

def predict(data_string):
    # Convert comma-separated string from Java to numpy array
    input_data = np.array([float(x) for x in data_string.split(',')]).reshape(1, 24, 1)
    prediction = model.predict(input_data, verbose=0)
    print(prediction[0][0]) # Output to be read by Java

if __name__ == "__main__":
    # Java will pass data as a command line argument
    predict(sys.argv[1])