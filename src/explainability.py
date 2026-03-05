import joblib
import shap
import numpy as np
import pandas as pd

def compute_global_shap(model_path, preprocessor_path, X_sample):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_transformed = preprocessor.transform(X_sample)

    explainer = shap.LinearExplainer(
        model.base_estimator_,
        X_transformed,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_transformed)

    return shap_values


def global_shap_table(shap_values, feature_names):
    return (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        })
        .sort_values("mean_abs_shap", ascending=False)
    )