import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load model and encoder
model = load('xgboost_model.joblib')
le_sex = load('label_encoder_sex.joblib')
le_BP = load('label_encoder_bp.joblib')
le_Cholesterol = load('label_encoder_cholesterol.joblib')
le_Drug = load('label_encoder_drug.joblib')
scaler = load('scaler.joblib')

# Title of the app
st.title('Prediksi Obat Berdasarkan Informasi Pasien')

# User input
age = st.number_input('Umur Pasien', min_value=0, max_value=120, value=30)
sex = st.selectbox('Jenis Kelamin', ['M', 'F'])
bp = st.selectbox('Tekanan Darah (BP)', ['LOW', 'NORMAL', 'HIGH'])
cholesterol = st.selectbox('Kolesterol', ['NORMAL', 'HIGH'])
na_to_k = st.number_input('Rasio Na_to_K', min_value=0.0, max_value=50.0, value=10.0)

# Encode input
sex_encoded = le_sex.transform([sex])[0]
bp_encoded = le_BP.transform([bp])[0]
cholesterol_encoded = le_Cholesterol.transform([cholesterol])[0]

# Prepare input array
input_data = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button('Prediksi Obat'):
    prediction = model.predict(input_data_scaled)
    predicted_drug = le_Drug.inverse_transform(prediction)[0]
    st.success(f'Prediksi: {predicted_drug}')
