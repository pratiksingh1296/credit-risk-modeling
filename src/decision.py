import pandas as pd
import joblib

def apply_business_policy(df, pd_column):
    # Risk Bucketing
    df['RiskBucket'] = pd.cut(
        df[pd_column],
        bins=[0, 0.05, 0.16, 0.45, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
        right=False
    )

    # Decision Mapping
    decision_map = {
        "Low": "Approve",
        "Medium": "Approve (Review Terms)",
        "High": "Manual Review",
        "Very High": "Reject"
    }
    df["Decision"] = df["RiskBucket"].map(decision_map)
    return df

def generate_policy_report(df, output_path):
    # Generates a crosstab of Risk vs Actual Defaults for stakeholders
    cross_tab = pd.crosstab(df['RiskBucket'], df['TARGET'], normalize="index")
    cross_tab.rename(columns={0: "Non-Default", 1: "Default"}, inplace=True)
    cross_tab.to_csv(output_path)
    return cross_tab

if __name__ == "__main__":
    # Load calibrated model and test data
    model = joblib.load("../models/logreg_platt.joblib")
    X_test, y_test = joblib.load('../models/test_data.joblib')

    # Run inference
    df_results = X_test.copy()
    df_results['TARGET'] = y_test
    df_results['PD'] = model.predict_proba(X_test)[:, 1]

    # Apply Policy
    df_results = apply_business_policy(df_results, 'PD')

    # Save Decisioned Data
    df_results.to_csv("../data/processed/final_decisions.csv", index=False)
    
    # Save Policy Report
    generate_policy_report(df_results, "../reports/summary_tables/policy_crosstab.csv")