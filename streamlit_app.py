import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
# import pickle
import joblib

model = joblib.load('K-Nearest Neighboursmodel.pkl')

with open('accuracy.txt', 'r') as file:
  accuracy = file.read()

st.title(f"Model Selection and Real-Time Prediction")
st.write(f"Model {accuracy}")

st.header("Real_Time Prediction")

test_data = pd.read_csv('mobile_price_range_data.csv')

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[;, -1]

input_data = []
for col in x_test.columns:
  input_value = st.number_input(f"Input from {col}", value=0.0)
  input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns = x_test.columns)

if st.button("Predict"):
  prediction = model.predict(input_df)

st.header("Accuracy Plot")
st.bar_chart([float(accuracy.split(': ')[1])])
