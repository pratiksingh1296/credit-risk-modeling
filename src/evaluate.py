import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, log_loss, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.calibration import calibration_curve
import seaborn as sns

# Load Models
model = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/logres_baseline.joblib')
X_test, y_test = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib')

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -- Calibrations --
# ROC_AUC | Log loss | Classification Report

print("ROC-AUC:\n", roc_auc_score(y_test, y_proba))
print('Log-Loss:',log_loss(y_test, y_proba))
print(classification_report(y_test, y_pred))

# Roc Display Graph
roc_disp = RocCurveDisplay.from_predictions(y_test, y_proba)
plt.savefig('C:/Users/Pratik/DS/credit-risk-ml/reports/figures/roc_disp.png',dpi=300,bbox_inches='tight')
plt.show()

# PrecisionRecallDisplay
prd = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.savefig('C:/Users/Pratik/DS/credit-risk-ml/reports/figures/precision_recall_disp.png',dpi=300,bbox_inches='tight')
plt.show()

# Classification Performnace - Threshold Based
threshold = 0.3
y_pred_label = (y_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_label)
ConfusionMatrixDisplay(cm).plot()

# Finding Optimal Threshold
thresholds = np.linspace(0.05, 0.95, 50)

results = []
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    results.append({
        "threshold": t,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# Plot for Threshold vs Classification Metrics
plt.figure(figsize=(8,5))
plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
plt.plot(results_df["threshold"], results_df["recall"], label="Recall")
plt.plot(results_df["threshold"], results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Classification Metrics")
plt.legend()
plt.grid(True)
plt.show()

# Use optimal threshold
optimal_threshold = results_df.loc[results_df["f1"].idxmax(), "threshold"]
optimal_threshold

# Confusion Matrix with Optimal threshold
y_pred_opt = (y_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)

ConfusionMatrixDisplay(cm).plot()
plt.title(f"Confusion Matrix (threshold = {optimal_threshold:.2f})")
plt.savefig('C:/Users/Pratik/DS/credit-risk-ml/reports/figures/optimal_threshold_CM.png',dpi=300,bbox_inches='tight')
plt.show()

# Feature Names Driving Model Predictions
feature_names = (
    model.named_steps['preprocess']
    .get_feature_names_out()
)

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.named_steps['clf'].coef_[0]
}).sort_values(by='coefficient', ascending=False)

coef_df.head(15)

# HIstogram
sns.histplot(y_proba, bins=50)
plt.title("Predicted Default Probability Distribution")

# Calibration Curve plot
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Predicted probability")
plt.ylabel("Observed default rate")
plt.title("Calibration Curve (Baseline)")

# Showing Example Predictions
example_df = X_test.copy()
example_df['actual'] = y_test
example_df['pred_proba'] = y_proba
example_df.sort_values('pred_proba', ascending=False).head(10)