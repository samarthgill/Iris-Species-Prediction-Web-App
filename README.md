
Iris Species Prediction Web App
This project is a full-stack application that integrates a machine learning model into an interactive web app. It serves as a practical example of deploying a data science model, allowing users to input an Iris flower's measurements (sepal length/width, petal length/width) and receive an instant species prediction: Iris-setosa, Iris-versicolor, or Iris-virginica.

The entire application is built using a modern Python-centric stack. The backend is powered by Flask, a lightweight and flexible web framework that serves the machine learning model through a simple REST API. This API acts as the bridge between the user interface and the predictive logic. The machine learning model itself is a Logistic Regression classifier, a robust and interpretable algorithm chosen for this classification task. It was trained on the well-known Iris dataset from the UCI Machine Learning Repository. To ensure reliable predictions, the training data was preprocessed using scikit-learn's StandardScaler to normalize the feature values, a critical step that helps the model converge and perform accurately.

The frontend is a clean, single-page application crafted with standard HTML and styled with Tailwind CSS for a responsive and modern design. User interactions are handled by vanilla JavaScript, which captures the form data, sends an asynchronous request to the Flask API, and dynamically displays the model's prediction on the page without requiring a reload. This creates a seamless and responsive user experience, effectively showcasing how a trained data model can be operationalized and made accessible to end-users through a simple and intuitive web interface.

Project Structure
iris_predictor/
│
├── model.py            # Script to train and save the ML model and scaler
├── app.py              # The Flask web server that serves the model as an API
├── model.pkl           # The saved, pre-trained machine learning model
├── scaler.pkl          # The saved scaler for preprocessing data
├── templates/
│   └── index.html      # The HTML frontend for the user interface
│
└── README.md           # This file

How to Run This Project
Prerequisites
Python 3.x

pip (Python package installer)

1. Clone the Repository
git clone <your-repository-url>
cd iris_predictor

2. Install Dependencies
Install the required Python libraries using pip:

pip install pandas scikit-learn flask numpy

3. Train the Model
Run the model.py script once to generate the model.pkl and scaler.pkl files.

python model.py

4. Start the Web Server
Run the app.py script to start the Flask development server.

python app.py

5. Use the Application
Open your web browser and navigate to http://127.0.0.1:5000. You will see the Iris Predictor form. Enter the measurements and click "Predict Species" to see the result.
