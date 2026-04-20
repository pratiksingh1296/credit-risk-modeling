# Credit Risk Modeling with Probability Calibration, Risk Bucketing, and SHAP Explainability

## Project Overview
This project builds an end-to-end **credit risk modeling pipeline** focused on:
- Predicting probability of default (PD)
- Calibrating model probabilities
- Translating predictions into business decisions
- Explaining decisions using SHAP

Rather than optimizing for accuracy alone, the project emphasizes **risk ranking, uncertainty awareness, and explainability**, 
closely mirroring how credit risk models are used in real financial institutions.

---

## Key Results
### ROC Curve
![ROC Curve](reports/figures/roc_disp.png)

### Global SHAP Feature Importance
![SHAP](reports/figures/summary_global_imp_bar.png)

---

## Dataset
- **Source:** [Home Credit Default Risk Dataset – Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- Raw dataset is not included in this repository due to size constraints.
- **Target:** `TARGET` (1 = default, 0 = non-default)
- **Data type:** Structured tabular data (numerical + categorical)
- **Challenges:**
  - Class imbalance
  - Missing values
  - High-cardinality categorical features
  - Regulatory need for explainability

---

## Problem Statement
Given applicant-level financial and demographic data, predict the probability that a loan applicant will default and translate that risk into actionable credit decisions.

Key objectives:
- Produce **well-calibrated probabilities**, not just class labels
- Rank applicants by risk
- Create interpretable risk buckets
- Justify decisions using model explainability

---

## Project Structure
credit-risk-ml/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling_baseline.ipynb
│ ├── 04_uncertainty_calibration.ipynb
│ ├── 05_business_decisions.ipynb
│ └── 06_explainability_shap.ipynb
│
├── src/
│ ├── data_prep.py
│ ├── features.py
│ ├── train.py
│ ├── evaluate.py
│ ├── uncertainty.py
│ └── explainability.py
│
├── models/
│ ├── logreg_baseline.joblib
│ ├── logreg_platt.joblib
│ └──  preprocessor_fit.joblib
│
├── reports/
│ └── figures/
| └── summary_tables
│
└── README.md
└── requirements.txt


---

## Modeling Approach

### Baseline Model
- Logistic Regression
- Class-weighted to handle imbalance
- Pipeline with preprocessing

### Evaluation Metrics
- ROC-AUC (risk ranking)
- Precision / Recall
- Confusion Matrix
- Probability distributions

---

## Model Performance Summary

| Metric | Baseline Logistic | Calibrated (Platt) |
|--------|-------------------|--------------------|
| ROC-AUC | 0.74 | 0.74 |
| Brier Score | 0.198 | 0.182 |
| ECE | 0.041 | 0.004 |

The baseline logistic regression achieved an ROC-AUC of 0.74, indicating moderate
risk-ranking ability.
### ROC Curve (Baseline)
![ROC](reports/figures/roc_disp.png)
### Precision-Recall Curve (Baseline)
![PR](reports/figures/precision_recall_disp.png)

Calibration reduced Brier Score and ECE while ROC-AUC remained unchanged.

---

## Probability Calibration
To ensure predicted probabilities reflect real-world default rates:
- **Platt Scaling**
- **Isotonic Regression**

Platt Scaling was selected after comparing calibration methods using:
- Reliability curves
- Expected Calibration Error (ECE)
- Brier Score

Platt Scaling provided near-zero ECE and improved probability alignment without degrading AUC.
Calibration improved probability reliability without degrading discrimination (AUC remained stable).

---

## Threshold Analysis

Before defining risk buckets, the model's performance across probability thresholds was analyzed to understand trade-offs between approval rate,
false positives, and missed defaults.

![Threshold](reports/figures/optimal_threshold_CM.png)

This analysis illustrates how decision thresholds influence portfolio risk
and approval rates. However, rather than relying on a single cutoff, the
final decision framework uses multiple risk buckets aligned with lending policy.

---

## Risk Bucketing & Decisions
Predicted PDs are converted into risk buckets aligned with business policy:

| Risk Bucket | PD Range | Decision |
|------------|---------|----------|
| Low        | < 5%     | Auto-approve |
| Medium     | 5–16%    | Approve with conditions |
| High       | 16–45%   | Manual review |
| Very High  | > 45%    | Reject |

This allows graded decision-making rather than binary classification.

---

## Business Decisions
The project illustrates how model outputs directly influence lending decisions and portfolio performance. It demonstrates:
- How approval thresholds determine portfolio risk exposure
- The trade-off between growth (approval rate) and credit losses (default rate)
- Why applicants with seemingly reasonable profiles may still fall below risk tolerance
- How explainability increases stakeholder trust in automated underwriting
- How model-driven decisions align with institutional risk appetite

---

## Explainability with SHAP

### Global Feature Importance
![SHAP](reports/figures/summary_global_imp_bar.png)

SHAP (SHapley Additive exPlanations) is used to:
- Identify global drivers of default risk
- Explain individual applicant decisions
- Support transparency and regulatory compliance

Key outputs:
- Global feature importance (mean absolute SHAP)
- Individual applicant explanations
- Quantification of each feature’s contribution to individual PD
- Consistency between SHAP risk drivers and domain expectations
- Auditable explanation layer for regulatory transparency

---

### Example: High-Risk Applicant Explanation

Predicted Probability of Default (PD): 0.72  
Risk Bucket: Very High → Reject

Top risk-increasing drivers:
- Low income
- Short employment history
- High external risk score
- Previous late payments

Top risk-reducing drivers:
- Age
- Stable organization type

This demonstrates how SHAP provides transparent, case-level explanations that align with lending policy decisions.

---

## Key Takeaways
- Calibration is critical when probabilities drive decisions
- Risk ranking matters more than raw accuracy
- Explainability is essential for regulated domains
- Business logic must be explicitly defined, not implied

---

## Future Improvements
- Tree-based models (GBM / XGBoost) with monotonic constraints
- Cost-sensitive optimization
- Reject inference
- Temporal validation
- Policy stress testing

---

## Author
Built as a portfolio project to demonstrate **end-to-end applied data science**, bridging modeling, uncertainty, and business decision-making.

---

## What This Project Demonstrates

- Risk modeling beyond accuracy (ranking + calibration)
- Business-aligned threshold optimization
- Probability calibration for financial reliability
- Model transparency using SHAP for interpretability
- Translation of predictions into actionable credit policy

---

## Setup
```bash
pip install -r requirements.txt
```