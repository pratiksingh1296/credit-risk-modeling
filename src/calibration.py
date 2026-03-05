import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay

model = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/logres_baseline.joblib')
X_test, y_test = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib')

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calibration Curve Plot
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
plt.plot(prob_pred, prob_true, marker = 'o', label='Model')
plt.plot([0,1],[0,1],linestyle='--',label='Perfect Calibration')
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed Frequency")
plt.legend()
plt.title("Reliability/Calibration Curve")
plt.show()

# Calculating ECE - Expected Calibration Error

def expected_calibration_error(y_true, y_prob, n_bins=10):
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

ece = expected_calibration_error(y_test, y_proba)
print(f"Expected Calibration Error (ECE): {ece}")

# Brier Score 
brier = brier_score_loss(y_test, y_proba)
print(f"Brier score: {brier}")

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:\n", roc_auc)

# Platt Calibration
platt = CalibratedClassifierCV(
    model, method='sigmoid', cv='prefit'
)
platt.fit(X_test,y_test)
joblib.dump(platt, "C:/Users/Pratik/DS/credit-risk-ml/models/logreg_platt.joblib")
y_prob_platt = platt.predict_proba(X_test)[:, 1]

iso = CalibratedClassifierCV(
    model, method="isotonic", cv="prefit"
)
iso.fit(X_test, y_test)
y_prob_iso = iso.predict_proba(X_test)[:, 1]

ece_platt = expected_calibration_error(y_test, y_prob_platt)
brier_platt = brier_score_loss(y_test, y_prob_platt)
roc_auc_platt = roc_auc_score(y_test, y_prob_platt)

ece_iso = expected_calibration_error(y_test, y_prob_iso)
brier_iso = brier_score_loss(y_test, y_prob_iso)
roc_auc_iso = roc_auc_score(y_test, y_prob_iso)

roc_base_disp = RocCurveDisplay.from_predictions(y_test, y_proba)
roc_platt_disp = RocCurveDisplay.from_predictions(y_test, y_prob_platt)
roc_iso_disp = RocCurveDisplay.from_predictions(y_test, y_prob_iso)

comparison = pd.DataFrame({
    'Model':[
        'Baseline',
        'Platt-Calibrated',
        'Iso-calibrated'
    ],
    'ROC-AUC':[
        roc_auc,
        roc_auc_platt,
        roc_auc_iso
    ],
    'ECE':[
        ece,
        ece_platt,
        ece_iso
    ],
    'Brier':[
        brier,
        brier_platt,
        brier_iso
    ]
})
comparison

pd_series = pd.Series(y_prob_platt, name='PD')
risk_bucket = pd.cut(
    pd_series,
    bins = [0, 0.10, 0.30, 1.0],
    labels = ['Low','Medium','High'],
    right=False
)
risk_bucket

risk_df = pd.DataFrame({
    'PD':y_prob_platt,
    'TARGET': y_test.values,
    'RiskBucket': risk_bucket
})
risk_df

bucket_summary = (
    risk_df.groupby("RiskBucket", observed=False).agg(
        count=('TARGET','size'),
        Avg_PD=('PD','mean'),
        default_rate=('TARGET','mean')
    )
    .sort_values('Avg_PD')
)
bucket_summary

bucket_summary["default_rate"].plot(
    kind="bar",
    title="Observed Default Rate by Risk Bucket",
    ylabel="Default Rate",
    xlabel="Risk Bucket",
    rot=0
)
plt.show()

bucket_summary["population_share"] = (bucket_summary["count"] / bucket_summary["count"].sum())
bucket_summary.to_csv("C:/Users/Pratik/DS/credit-risk-ml/data/processed/risk_bucket_summary.csv",index=True)