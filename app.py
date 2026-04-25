# Credit Risk Default Predictor Streamlit App
# Imports
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt

# Load the preprocessor and model
preprocessor = joblib.load("models/preprocessor_fit.joblib")
model = joblib.load("models/xgb_calibrated.joblib")

# Load Defaults Values for Input Fields
sample_row = pd.read_csv("reports/app_median_row.csv")

def risk_buckets(prob):
    if prob < 0.05: return "Low", "Green", "Approve"
    if prob < 0.16: return "Medium", "Blue", "Approve with Conditions"
    if prob < 0.45: return "High", "Orange", "Manual Review"
    return "Very High", "Red", "Reject"

st.title("Credit Risk Default Predictor")
st.markdown("Enter Applicant's details to assess default probability")

# Input Form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Applicant's Age (years)", 18, 70, 35)
    employed_years = st.number_input("Years Employed", 0, 40, 5)
    amt_income_total = st.number_input("Annual Income", 0, 10000000, 500000)
    ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.51)
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.57)
    amt_credit = st.number_input("Credit Amount", 0, 5000000, 500000)

with col2:
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.54)
    gender = st.selectbox("Gender", ["M", "F"])
    education = st.selectbox("Education",  ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    contract_type = st.selectbox("Loan Type", ["Cash loans", "Revolving loans"])
    has_children = st.selectbox("Has Children?", ["No", "Yes"])
    has_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    has_realty = st.selectbox("Owns a Property?", ["No", "Yes"])

if st.button("Predict Risk"):
    # Start with a real training row as base
    input_row = sample_row.copy()
    # Override with User Inputs      
    input_row["AGE_YEARS"] = float(age)
    input_row["EMPLOYED_YEARS"] = float(-employed_years)
    input_row["AMT_INCOME_TOTAL"] = float(amt_income_total)
    input_row["EXT_SOURCE_1"] = float(ext_source_1)
    input_row["EXT_SOURCE_2"] = float(ext_source_2)
    input_row["CODE_GENDER"] = gender
    input_row["NAME_EDUCATION_TYPE"] = education
    input_row["HAS_CAR"] = 1 if has_car == "Yes" else 0
    input_row["HAS_REALTY"] = 1 if has_realty == "Yes" else 0
    input_row["HAS_CHILDREN"] = 1 if has_children == "Yes" else 0      

    # Preprocess & Predict
    input_processed = preprocessor.transform(input_row)
    prob = model.predict_proba(input_processed)[0][1]
    bucket = risk_buckets(prob)

    # Buckets
    bucket_name, bucket_color, bucket_decision = risk_buckets(prob)

    # Display Results
    st.markdown("---")
    st.subheader("Risk Assessment Result")
    st.markdown(f"### Predicted Default Probability: {prob:.2%}")
    st.markdown(f"### Risk Level: {bucket_name}")
    st.markdown(f"### Decision: {bucket_decision}")

    # Risk Gauge
    st.progress(float(prob))
