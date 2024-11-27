import json
import pickle
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) to allow requests from Flutter
CORS(app)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Boston House Price Prediction API"

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Parse the incoming JSON data from the request
        data = request.json['data']
        print("Received data:", data)

        # Convert the data to the appropriate shape for prediction
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))

        # Use the trained model to predict
        output = regmodel.predict(new_data)[0]

        # Return the prediction in JSON format
        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
