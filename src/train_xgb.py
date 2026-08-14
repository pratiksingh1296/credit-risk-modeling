from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def build_xgb_pipeline(preprocessor):

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=9, # Excellent for imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1 # Uses all CPU cores
    )
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('clf', xgb_model)
    ])
    
    return pipeline

'''
def train_and_calibrate_xgb(pipeline, X_train, y_train):
    calibrated_xgb = CalibratedClassifierCV(
        estimator=pipeline,
        method='sigmoid',
        cv=5
    )
    
    print("Training and Calibrating XGBoost (this may take a while)...")
    calibrated_xgb.fit(X_train, y_train)
    return calibrated_xgb
'''

if __name__ == "__main__":

    # Load preprocessor 
    preprocessor = joblib.load( MODEL_DIR / "preprocessor_fit.joblib")

    # Load fit data
    X_fit, y_fit = joblib.load( MODEL_DIR / "fit_data.joblib")
    
    # Build Model
    xgb_pipe = build_xgb_pipeline(preprocessor)

    # Train
    print("Training XGBoost...")
    xgb_pipe.fit(X_fit, y_fit)

    # final_xgb = train_and_calibrate_xgb(xgb_pipe, X_train, y_train)
    
    # Save 
    joblib.dump(xgb_pipe, MODEL_DIR / "xgb_model.joblib")

    print("XGBoost Model Saved!")