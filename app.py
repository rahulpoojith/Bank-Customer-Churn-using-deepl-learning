import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('one_hot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Streamlit app configuration
st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")
st.title("Bank Customer Churn Prediction")
st.write("This application predicts whether a bank customer will churn based on their profile data.")

# User input form
st.sidebar.header("Customer Information")
geography = st.selectbox("Geography", encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.slider('Credit Score', 300, 900)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

# One-hot encode the 'Geography' column
geo_encoded = encoder.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder.get_feature_names_out(['Geography']))
input_data = input_data.drop('Geography', axis=1)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input features
input_scaled = scaler.transform(input_data)

# Make prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_scaled)
    st.write(f"Churn Probability: {prediction[0][0]:.2f}")
    if prediction[0][0] > 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")