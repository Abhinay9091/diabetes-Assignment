
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient details")

# Correct file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved models and preprocessing objects
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
zero_cols = joblib.load(os.path.join(BASE_DIR, "zero_cols.pkl"))

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=30.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, value=0.47
)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, diabetes_pedigree, age]],
        columns=feature_names
    )

    # Handle zero values
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)

    # Apply the same imputation
    input_df[zero_cols] = imputer.transform(input_df[zero_cols])

    # Apply the same scaling used during training
    scaled_input = scaler.transform(input_df)

    # Prediction
    pred = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0, 1]

    if pred == 1:
        st.error(f"High risk: Diabetic ({proba:.1%} probability)")
    else:
        st.success(
            f"Lower risk: Non-Diabetic ({proba:.1%} probability of diabetes)"
        )

    st.info("This is an educational prediction, not a medical diagnosis.")
