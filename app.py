# app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title and description
st.title("🚗 Car Purchase Amount Prediction")
st.markdown("Enter your financial details to estimate how much you might spend on a car.")

# User Inputs with placeholders
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
salary = st.number_input("Annual Salary ($)", value=50000.0, step=1000.0, format="%.2f")
credit_card_debt = st.number_input("Credit Card Debt ($)", value=5000.0, step=500.0, format="%.2f")
net_worth = st.number_input("Net Worth ($)", value=100000.0, step=5000.0, format="%.2f")

# Predict Button
if st.button("Predict"):
    input_data = np.array([[age, salary, credit_card_debt, net_worth]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    rounded_prediction = round(prediction, 2)

    st.success(f"💰 Estimated Car Purchase Amount: **${rounded_prediction:,.2f}**")
