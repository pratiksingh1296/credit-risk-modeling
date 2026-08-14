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

# Define Risk Buckets
def risk_buckets(prob):
    if prob < 0.05: return "Low", "Green", "Approve"
    if prob < 0.16: return "Medium", "Blue", "Approve with Conditions"
    if prob < 0.45: return "High", "Orange", "Manual Review"
    return "Very High", "Red", "Reject"


# Load mappings for income and organization types
income_risk_map = joblib.load("models/income_risk_map.joblib")
org_risk_map = joblib.load("models/org_risk_map.joblib")
income_map = joblib.load("models/income_map.joblib")

# Get sorted lists of income and organization types for dropdowns
income_types = sorted(income_map.keys())
organization_types = sorted(org_risk_map.index.tolist())

# Page Title and Description
st.title("Credit Risk Default Predictor")
st.markdown("Enter Applicant's details to assess default probability")

# Input Form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Applicant's Age (years)", 18, 70, 35)
    amt_income_total = st.number_input("Annual Income", 0, 10000000, 500000)
    income_type = st.selectbox("Income Type", income_types)
    employed_years = st.number_input("Years Employed", 0, 40, 5)
    ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.51)
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.57)
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.54)

with col2:
    amt_credit = st.number_input("Credit Amount", 0, 5000000, 500000)
    education = st.selectbox("Education",  ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    organization_type = st.selectbox("Organization Type",organization_types)
    contract_type = st.selectbox("Loan Type", ["Cash loans", "Revolving loans"])
    marital_status = st.selectbox("Marital Status",["Married", "Single / not married", "Civil marriage", "Widow", "Separated"])
    has_children = st.selectbox("Has Children?", ["No", "Yes"])
    has_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    has_realty = st.selectbox("Owns a Property?", ["No", "Yes"])

button_col1, button_col2, button_col3 = st.columns([2, 1, 2])

with button_col2:
    predict = st.button("Predict Risk", use_container_width=True)

if predict:
    # Start with a real training row as base
    input_row = sample_row.copy()
    # Override with User Inputs      
    input_row["AGE_YEARS"] = float(age)
    input_row["EMPLOYED_YEARS"] = float(employed_years)

    input_row["AMT_INCOME_TOTAL"] = float(amt_income_total)
    input_row["AMT_CREDIT"] = float(amt_credit)

    input_row["EXT_SOURCE_1"] = float(ext_source_1)
    input_row["EXT_SOURCE_2"] = float(ext_source_2)
    input_row["EXT_SOURCE_3"] = float(ext_source_3)

    input_row["NAME_CONTRACT_TYPE"] = contract_type
    input_row["CODE_GENDER"] = gender

    input_row["NAME_EDUCATION_TYPE"] = education
    input_row["NAME_FAMILY_STATUS"] = marital_status

    input_row["HAS_CAR"] = 1 if has_car == "Yes" else 0
    input_row["HAS_REALTY"] = 1 if has_realty == "Yes" else 0
    input_row["HAS_CHILDREN"] = 1 if has_children == "Yes" else 0
    
    income_group = income_map.get(income_type, "Other")
    input_row['INCOME_RISK'] = income_risk_map.get(income_group, income_risk_map.mean())
    input_row["ORG_RISK"] = org_risk_map.get(organization_type, org_risk_map.mean())

    # Preprocess & Predict
    input_processed = preprocessor.transform(input_row)
    prob = model.predict_proba(input_processed)[0][1]

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

    st.markdown("### Why this prediction?")
    explainer = shap.TreeExplainer(joblib.load("models/xgb_model.joblib"))

    # get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split('__')[1] for name in feature_names]

    input_df_shap = pd.DataFrame(input_processed, columns=clean_names)
    shap_values = explainer.shap_values(input_df_shap)
    
    st.caption("The waterfall chart shows which factors most influenced this applicant's default probability. Red bars increase risk, blue bars decrease it.")

    fig , ax = plt.subplots(figsize=(10,5))
    shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                feature_names=clean_names
            ),
        show=False
        )
    st.pyplot(fig)
    plt.close()
