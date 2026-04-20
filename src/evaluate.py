import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay, log_loss
import seaborn as sns

def calculate_ece(y_true, y_prob, n_bins=10):
    """Adds the ECE calculation to your evaluation script."""
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

def get_performance_metrics(y_true, y_proba):
    # Calculate Statistical Metrics
    return {
        "auc-roc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba)                        
    }

def plot_ranking_curves(y_true, y_proba, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Roc Curve
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax1)
    ax1.set_title("ROC Curve (Ranking Power)")

    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax2)
    ax2.set_title("PR Curve (Imbalance Handling)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_reliability_curve(y_true, y_proba, label, save_path):
    # Plot Calibration to see if PD matches reality 
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=label)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel("Predicted Probability of Default")
    plt.ylabel("Actual Default Rate")
    plt.title(f"Reliability Curve: {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_score_distribution(y_proba, save_path):
    # Plots the distribution of raw probabilities.
    plt.figure(figsize=(8, 5))
    sns.histplot(y_proba, bins=50, kde=True, color='royalblue')
    plt.title("Distribution of Model Predicted Probabilities")
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()

def plot_all_diagnostics(y_true,y_proba, model_name, output_dir):
    plot_ranking_curves(y_true, y_proba, f"{output_dir}/{model_name}_roc.png")
    plot_reliability_curve(y_true, y_proba, model_name, f"{output_dir}/{model_name}_calibration.png")
    plot_score_distribution(y_proba, f"{output_dir}/{model_name}_distribution.png")

def create_comparison_table(model_results_list):
    df = pd.DataFrame(model_results_list)
    df = df.round(4)
    return df

if __name__ == "__main__":
    # Paths
    MODEL_DIR = "C:/Users/Pratik/DS/credit-risk-ml/models/"
    FIG_DIR = "C:/Users/Pratik/DS/credit-risk-ml/reports/figures"
    TAB_DIR = "C:/Users/Pratik/DS/credit-risk-ml/reports/summary_tables"

    # Load Data
    X_test, y_test = joblib.load('C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib')

    # Models we want to compare
    model_files = {
        "LogReg_Baseline": "logreg_baseline.joblib",
        "LogReg_Calibrated": "logreg_calibrated.joblib",
        "XGB_Baseline": "xgb_baseline.joblib",
        "XGB_Calibrated": "xgb_calibrated.joblib"
    }
    
    comparison_list = []

    for label, filename in model_files.items():
        try:
            # Load and Predict
            model = joblib.load(f"{MODEL_DIR}{filename}")
            y_proba = model.predict_proba(X_test)[:, 1]

            print(f"Evaluating {label}...")

            # Generate Individual Diagnostics (ROC, Calibration, Dist)
            plot_all_diagnostics(y_test, y_proba, label, FIG_DIR)

            # Collect Metrics for the Comparison Table
            metrics = get_performance_metrics(y_test, y_proba)
            metrics["model"] = label
            metrics["ece"] = calculate_ece(y_test, y_proba) # Add the ECE you liked
            comparison_list.append(metrics)
            
        except FileNotFoundError:
            print(f"Skipping {label}: File not found.")

    MODEL_LABEL = "logreg_baseline"
    OUTPUT_DIR = "C:/Users/Pratik/DS/credit-risk-ml/reports/figures/"

    metrics = get_performance_metrics(y_test, y_proba)
    print(f"Metrics for {MODEL_LABEL}: {metrics}")

    plot_all_diagnostics(y_test, y_proba, MODEL_LABEL, OUTPUT_DIR)

    # Save and Show the Comparison Table
    comparison_df = pd.DataFrame(comparison_list)
    # Reorder columns to make 'model' first
    cols = ['model'] + [c for c in comparison_df.columns if c != 'model']
    comparison_df = comparison_df[cols].round(4)
    
    print("\nModel Comparison Table:")
    print(comparison_df)
    
    comparison_df.to_csv(f"{TAB_DIR}/model_comparison_results.csv", index=False)