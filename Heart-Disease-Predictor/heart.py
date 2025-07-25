import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

# Load trained model
model = joblib.load("Heart_disease_model.pkl")

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# App Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the following medical parameters to predict the presence of heart disease.")

# Sidebar Info
with st.sidebar:
    st.header("🩺 About")
    st.write("""
        This app uses a machine learning model to predict the **likelihood of heart disease** 
        based on patient medical attributes.
    """)
    st.write("Developed by **Ashutosh**")

# Input fields
def user_input():
    age = st.slider("Age", 20, 90, 50)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral in mg/dl (chol)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2])  # or adjust if encoded differently

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([data])

# Predict
input_df = user_input()

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({round(pred_proba * 100, 2)}% confidence)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({round(pred_proba * 100, 2)}% confidence)")

    st.markdown("----")
    st.subheader("🧾 Input Summary")
    st.dataframe(input_df)

