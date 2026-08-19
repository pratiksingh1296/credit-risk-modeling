# Credit Risk Default Predictor Streamlit App
# Imports
import os

import streamlit as st
import requests
import matplotlib.pyplot as plt

from utils.shap_plot import plot_shap_waterfall


# API URL
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

# Get Matadata
try:
    metadata_response = requests.get(
        f"{API_URL}/metadata",
        timeout=10
    )
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
except requests.RequestException:
    st.error("Unable to connect to the Credit Risk API.")
    st.stop()

# Get info from metadata
income_types = metadata["income_types"]
organization_types = metadata["organization_types"]
education_types = metadata["education_types"]
loan_types = metadata["loan_types"]
marital_statuses = metadata["marital_statuses"]

# Page Title and Description
st.title("Credit Risk Assessment")
st.markdown("Assess the probability of loan default using applicant, financial, credit, and employment information.")
st.caption("Enter the applicant's information below and select Predict Risk to generate an assessment.")

# Application Form
st.subheader("Applicant Information")

# Family Section
with st.expander("Personal & Family", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Applicant's Age (years)", 18, 70, 35)
        marital_status = st.selectbox("Marital Status", marital_statuses)

    with col2:
        children_count = st.number_input("Number of Children", 0, 20, 0)
        family_member_count = st.number_input("Family Members", 1, 30, 1)

# Financial Section
with st.expander("Financial Information", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        amt_income_total = st.number_input("Annual Income", 0, 10000000, 500000)
        amt_credit = st.number_input("Credit Amount", 0, 5000000, 500000)

    with col2:
        annuity = st.number_input("Annuity", 0, 5000000, 25000)
        goods_price = st.number_input("Amount to Be Financed", 0, 5000000, 500000)

# Credit Profile
with st.expander("Credit Profile", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.51)

    with col2:
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.57)

    with col3:
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.54)

# Employment & Loan Details
with st.expander("Employment & Loan", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        income_type = st.selectbox("Income Type", income_types)
        employed_years = st.number_input("Years Employed", 0, 40, 5)
        organization_type = st.selectbox("Organization Type", organization_types)

    with col2:
        education = st.selectbox("Education", education_types)
        contract_type = st.selectbox("Loan Type", loan_types)

# Assets Details
with st.expander("Assets", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        has_car = st.selectbox("Owns a Car?", ["No", "Yes"])

    with col2:
        has_realty = st.selectbox("Owns a Property?", ["No", "Yes"])

# Prediction Button
button_col1, button_col2, button_col3 = st.columns([2, 1, 2])

with button_col2:
    predict = st.button("Predict Risk", use_container_width=True)

if predict:

    api_request = {
        "age": age,
        "gender": gender,
        "education": education,
        "marital_status": marital_status,
        "income_type": income_type,
        "employed_years": employed_years,
        "annual_income": amt_income_total,
        "ext_credit_score_1": ext_source_1,
        "ext_credit_score_2": ext_source_2,
        "ext_credit_score_3": ext_source_3,
        "credit_amount": amt_credit,
        "annuity": annuity,
        "goods_price": goods_price,
        "has_car": has_car == "Yes",
        "has_property": has_realty == "Yes",
        "children_count": children_count,
        "family_member_count": family_member_count,
        "loan_type": contract_type,
        "organization_type": organization_type
    }

    # Predict
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=api_request,
            timeout=10
        )
        response.raise_for_status()
        api_result = response.json()

    except requests.RequestException:
        st.error("Unable to connect to the Credit Risk API.")
        st.stop()

    # Buckets, Probabilities, and Decisions
    prob = api_result["default_probability"]
    bucket_name = api_result["risk_level"]
    bucket_decision = api_result["decision"]

    # Explainability 
    try: 
        explain_response = requests.post(
            f"{API_URL}/explain",
            json=api_request,
            timeout=10
        )
        explain_response.raise_for_status()
        explain_result = explain_response.json()

    except requests.RequestException:
        st.error("Unable to retrieve the model explanation.")
        st.stop()


    # Display Results
    st.markdown("---")
    st.subheader("Risk Assessment")

    # Risk configuration
    risk_config = {
        "Low": {
            "color": "#16a34a",
            "range": "< 5%",
        },
        "Medium": {
            "color": "#f59e0b",
            "range": "5–16%",
        },
        "High": {
            "color": "#f97316",
            "range": "16–45%",
        },
        "Very High": {
            "color": "#dc2626",
            "range": "≥ 45%",
        }
    }

    config = risk_config[bucket_name]
    color = config["color"]

    col1, col2, col3 = st.columns([1.1, 0.9, 1.6])

    with col1:
        st.metric("Default Probability", f"{prob:.2%}")

    with col2:
        st.markdown(
            f"""
            <div style="font-size: 14px; color: #9ca3af;">
                Risk Level
            </div>
            <div style="
                display: inline-block;
                margin-top: 6px;
                padding: 4px 12px;
                border-radius: 999px;
                background-color: {color}22;
                color: {color};
                font-size: 24px;
                font-weight: 600;
            ">
                {bucket_name}
            </div>
            """,
            unsafe_allow_html=True
    )

    with col3:
        st.markdown(
            f"""
            <div style="font-size: 14px; color: #9ca3af;">
                Decision
            </div>
            <div style="
                margin-top: 6px;
                font-size: 24px;
                font-weight: 500;
            ">
                {bucket_decision}
            </div>
            """,
            unsafe_allow_html=True
    )

    # Risk Gauge
    st.caption(f"Default probability · {config['range']} = {bucket_name}")
    st.markdown(
        f"""
        <div style="
            position: relative;
            height: 12px;
            border-radius: 6px;
            background: linear-gradient(
                to right,
                #16a34a 0%,
                #16a34a 5%,
                #f59e0b 5%,
                #f59e0b 16%,
                #f97316 16%,
                #f97316 45%,
                #dc2626 45%,
                #dc2626 100%
            );
        ">
            <div style="
                position: absolute;
                left: {prob * 100}%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 18px;
                height: 18px;
                background: white;
                border: 3px solid {color};
                border-radius: 50%;
                box-shadow: 0 1px 4px rgba(0,0,0,0.4);
            "></div>
        </div>

        <div style="
            display: flex;
            justify-content: space-between;
            margin-top: 6px;
            color: #9ca3af;
            font-size: 12px;
        ">
            <span>0%</span>
            <span>5%</span>
            <span>16%</span>
            <span>45%</span>
            <span>100%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # SHAP
    st.markdown("---")
    st.subheader("Model Explanation")
    st.markdown(
        "The chart below shows the features that had the greatest influence "
        "on the model's prediction. Positive SHAP values push the model "
        "toward higher risk, while negative values push it toward lower risk."
    )
    st.caption(
        "SHAP values represent each feature's contribution to the model score; "
        "they are not percentage-point changes in default probability."
    )

    feature_names = explain_result["feature_names"]
    shap_values = explain_result["shap_values"]
    base_value = explain_result["base_value"]

    fig = plot_shap_waterfall(
        feature_names,
        shap_values,
        base_value
        )

    st.pyplot(fig)
    plt.close(fig)