# model.py

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Import the scaler
import pickle

# A function to train and save the model
def train_and_save_model():
    """
    This function trains a logistic regression model on the Iris dataset,
    scales the data, and saves both the trained model and the scaler to files.
    """
    print("Training the model...")

    # 2. Load the Iris dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_df = pd.read_csv(url, header=None, names=column_names)

    # 3. Clean the data to prevent ValueErrors
    # This section ensures that feature columns are numeric and removes problematic rows.
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_columns:
        # 'coerce' will turn any non-numeric values into NaN (Not a Number)
        iris_df[col] = pd.to_numeric(iris_df[col], errors='coerce')
    
    # Drop any rows that now have NaN values
    iris_df.dropna(inplace=True)

    # 4. Prepare the data
    # We separate the features (X) from the target variable (y)
    X = iris_df.iloc[:, :-1]
    y = iris_df.iloc[:, -1]

    # Split the dataset into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 5. Scale the features
    # Scaling helps the model converge and often improves performance.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train the Logistic Regression model on the scaled data
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate the model on the scaled test data
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with an accuracy of: {accuracy:.2f}")

    # 8. Save the trained model to a file
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved successfully as model.pkl")

    # 9. Save the scaler to a file
    # This is crucial for preprocessing new data in your web app.
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Scaler saved successfully as scaler.pkl")


# This block ensures that the training function runs only when the script is executed directly
if __name__ == '__main__':
    train_and_save_model()
