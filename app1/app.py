import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title("🩺 Diabetes Risk Predictor")

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
feature_names = joblib.load("feature_names.pkl")
zero_cols = joblib.load("zero_cols.pkl")

st.title("Diabetes Prediction")

st.write("Enter patient details")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=30.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.47
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)

if st.button("Predict"):
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                             columns=feature_names)
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    input_df[zero_cols] = imputer.transform(input_df[zero_cols])

    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error(f"High risk: Diabetic ({proba:.1%} probability)")
    else:
        st.success(f"Lower risk: Non-Diabetic ({proba:.1%} probability of diabetes)")