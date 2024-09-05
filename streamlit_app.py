# import streamlit as st
# from sklearn.model_selection import train_test_split
# import pandas as pd
# # import pickle
# import joblib

# model = joblib.load('K-Nearest Neighborsmodel.pkl')

# with open('accuracy.txt', 'r') as file:
#   accuracy = file.read()

# st.title(f"Model Selection and Real-Time Prediction")
# st.write(f"Model {accuracy}")

# st.header("Real_Time Prediction")

# test_data = pd.read_csv('mobile_price_range_data.csv')

# x_test = test_data.iloc[:, :-1]
# y_test = test_data.iloc[:, -1]

# input_data = []
# for col in x_test.columns:
#   input_value = st.number_input(f"Input from {col}", value=0.0)
#   input_data.append(input_value)

# input_df = pd.DataFrame([input_data], columns = x_test.columns)

# if st.button("Predict"):
#   prediction = model.predict(input_df)

# st.header("Accuracy Plot")
# st.bar_chart([float(accuracy.split(': ')[1])])
import streamlit as st
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# Load model and accuracy from files
model = joblib.load('K-Nearest Neighborsmodel.pkl')

with open('accuracy.txt', 'r') as file:
    accuracy = file.read().strip()

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Update this if you're using a remote server
mlflow.set_experiment("KNN_Model_Experiment")  # Create or set an experiment name

# Start a new MLflow run
with mlflow.start_run():
    # Log the accuracy as a metric
    accuracy_value = float(accuracy.split(': ')[1])
    mlflow.log_metric("accuracy", accuracy_value)

st.title("Model Selection and Real-Time Prediction")
st.write(f"Model Accuracy: {accuracy}")

st.header("Real-Time Prediction")

# Load test data
test_data = pd.read_csv('mobile_price_range_data.csv')

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Collect user input
input_data = []
for col in x_test.columns:
    input_value = st.number_input(f"Input for {col}", value=0.0)
    input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns=x_test.columns)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")

st.header("Accuracy Plot")
st.bar_chart([accuracy_value])
