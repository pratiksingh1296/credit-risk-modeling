import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def get_shap_explainer(model, data):
    if "XGBClassifier" in str(type(model)):
        return shap.TreeExplainer(model)
    else:
        return shap.LinearExplainer(model, data)


def get_shap_components(model_path, X_sample):
    
    full_pipeline = joblib.load(model_path)
    
    # Extract components from the pipeline
    preprocessor = full_pipeline.named_steps["preprocess"]
    clf = full_pipeline.named_steps["clf"]
    
    # Transform data and get clean feature names
    X_transformed = preprocessor.transform(X_sample)
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [name.split('__')[1] for name in raw_names]
    
    X_df = pd.DataFrame(X_transformed, columns=clean_names)
    
    # Initialize Explainer
    explainer = shap.LinearExplainer(clf, X_df, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_df)
    
    return explainer, shap_values, X_df

def save_global_importance(shap_values, X_df, output_dir):
    # Table
    importance_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(f"{output_dir}/top_risk_drivers.csv", index=False)
    
    # Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.savefig(f"{output_dir}/shap_bee_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

def save_local_explanations(explainer, shap_values, X_df, indices, output_dir):
    # Create a summary table for specific rows
    example_shap = pd.DataFrame(
        shap_values[indices],
        columns=X_df.columns,
        index=indices
    )
    example_shap.to_csv(f"{output_dir}/individual_explanations.csv")
    
    # Waterfall plot as an image:
    for idx in indices:
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[idx], 
            X_df.iloc[idx],
            show=False
        )
        plt.savefig(f"{output_dir}/explanation_applicant_{idx}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Paths
    MODEL_PATH = "C:/Users/Pratik/DS/credit-risk-ml/models/logreg_baseline.joblib"
    TEST_DATA_PATH = "C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib"
    OUT_DIR_FIGS = "C:/Users/Pratik/DS/credit-risk-ml/reports/figures"
    OUT_DIR_TABS = "C:/Users/Pratik/DS/credit-risk-ml/reports/summary_tables"

    # Load data
    _, X_test = joblib.load(TEST_DATA_PATH) # Assuming it's (X, y)
    
    # Get SHAP values
    explainer, shap_vals, X_test_df = get_shap_components(MODEL_PATH, X_test)
    
    # Global Importance
    save_global_importance(shap_vals, X_test_df, OUT_DIR_FIGS)
    
    # ocal (Individual) Explanations
    # Let's pick 5 diverse examples (e.g., index 0, 5, 10, 25, 50)
    save_local_explanations(explainer, shap_vals, X_test_df, [0, 5, 10, 25, 50], OUT_DIR_FIGS)
    
    print("SHAP analysis complete. Plots saved to reports/figures/")