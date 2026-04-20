import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay

def calculate_ece(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 0, 1 - 1e-8)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for m in range(n_bins):
        bin_mask = (bin_ids == m)
        if np.any(bin_mask):
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_prob[bin_mask].mean()
            ece += np.sum(bin_mask) / len(y_true) * np.abs(bin_acc - bin_conf)
    return ece

def train_calibrator(model, X_train, y_train, method='sigmoid'):
    # Method = sigmoid is Platt & Method = isotonic = Isotonic Regression
    calibrator = CalibratedClassifierCV(model, method=method, cv=5)
    calibrator.fit(X_train, y_train)
    return calibrator

def generate_risk_bucket(y_proba, y_true, bins=[0, 0.10, 0.30, 1.0], labels=['Low','Medium','High']):
    risk_df = pd.DataFrame({
        'PD': y_proba,
        'TARGET': y_true
    })
    risk_df['RiskBucket'] = pd.cut(risk_df['PD'], bins=bins, labels=labels, right=False)

    summary = risk_df.groupby("RiskBucket", observed=False).agg(
        count=('TARGET','size'),
        Avg_PD=('PD','mean'),
        default_rate=('TARGET','mean')
    ).sort_values('Avg_PD')

    summary["population_share"] = summary["count"] / summary["count"].sum()
    return summary

def plot_calibration_comparison(y_true, prob_list, label_list, save_path):
    """Plots multiple reliability curves on one graph for comparison."""
    plt.figure(figsize=(8, 6))
    for prob, label in zip(prob_list, label_list):
        prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=label)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Comparison")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # 1. Load Data & Baseline Model
    model = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/logres_baseline.joblib')
    X_test, y_test = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib')
    X_train, y_train = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/train_data.joblib')

    # Baseline Predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. Train Calibrators
    platt_model = train_calibrator(model, X_train, y_train, method='sigmoid')
    joblib.dump(platt_model, "../models/logreg_platt.joblib")
    y_prob_platt = platt_model.predict_proba(X_test)[:, 1]

    iso_model = train_calibrator(model, X_train, y_train, method='isotonic')
    joblib.dump(iso_model, "../models/logreg_iso.joblib")
    y_prob_iso = iso_model.predict_proba(X_test)[:, 1]

    print("Both calibration models trained successfully using 5-fold CV.")

    # 3. Calculate Metrics
    # ECE
    ece_platt = calculate_ece(y_test, y_prob_platt)
    ece_iso = calculate_ece(y_test, y_prob_iso)
    ece_base = calculate_ece(y_test, y_proba)
    # Brier
    brier_platt = brier_score_loss(y_test, y_prob_platt)
    brier_iso = brier_score_loss(y_test, y_prob_iso)
    # ROC
    roc_auc_platt = roc_auc_score(y_test, y_prob_platt)
    roc_auc_iso = roc_auc_score(y_test, y_prob_iso)

    # 4. Display or Save Comparison
    comparison = pd.DataFrame({
        'Model': ['Baseline', 'Platt', 'Isotonic'],
        'ECE': [ece_base, ece_platt, ece_iso],
        'Brier': [brier_score_loss(y_test, y_proba), brier_platt, brier_iso],
        'AUC': [roc_auc_score(y_test, y_proba), roc_auc_platt, roc_auc_iso]
    })
    print(comparison)

    # Generate Comparison Plot
    plot_calibration_comparison(
        y_test, 
        [y_proba, y_prob_platt, y_prob_iso], 
        ['Baseline', 'Platt (Sigmoid)', 'Isotonic'],
        save_path="C:/Users/Pratik/DS/credit-risk-ml/reports/figures/calibration_comparison.png"
    )
    # Using Platt Since it was better
    bucket_summary = generate_risk_bucket(
        y_prob_platt, 
        y_test, 
        bins=[0, 0.05, 0.16, 0.45, 1.0], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # Save the summary table
    bucket_summary.to_csv("C:/Users/Pratik/DS/credit-risk-ml/reports/summary_tables/risk_bucket_validation.csv")
    
    print("\nRisk Bucket Validation:")
    print(bucket_summary)

    print("\nUncertainty analysis complete. Plots and tables saved.")






