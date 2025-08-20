# app.py

# 1. Import necessary libraries
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# 2. Create a Flask application instance
app = Flask(__name__)

# 3. Load the pre-trained machine learning model and the scaler
# 'rb' means we are reading in binary mode
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 4. Define the routes (URL endpoints) for our application

# Route for the home page, which will render our HTML form
@app.route('/')
def home():
    """Renders the main page with the prediction form."""
    return render_template('index.html')

# Route for handling the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data from the form, scales it, makes a prediction,
    and returns the result.
    """
    try:
        # Get the feature values from the form's JSON payload
        data = request.json['features']

        # Convert the list of features into a NumPy array and reshape it
        features = np.array(data).reshape(1, -1)

        # IMPORTANT: Scale the features using the loaded scaler
        # The model was trained on scaled data, so new data must also be scaled.
        scaled_features = scaler.transform(features)

        # Use the loaded model to make a prediction on the scaled data
        prediction = model.predict(scaled_features)
        predicted_species = prediction[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_species})

    except Exception as e:
        # If an error occurs, return an error message
        return jsonify({'error': str(e)})

# 5. Run the Flask application
# This block ensures the server runs only when the script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
