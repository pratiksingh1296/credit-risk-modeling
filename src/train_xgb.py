from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

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

def train_and_calibrate_xgb(pipeline, X_train, y_train):
    calibrated_xgb = CalibratedClassifierCV(
        estimator=pipeline,
        method='sigmoid',
        cv=5
    )
    
    print("Training and Calibrating XGBoost (this may take a while)...")
    calibrated_xgb.fit(X_train, y_train)
    return calibrated_xgb

if __name__ == "__main__":
    
    logreg_pipe = joblib.load("../models/logres_baseline.joblib")
    preprocessor = logreg_pipe.named_steps['preprocess']
    
    X_train, y_train = joblib.load('../models/train_data.joblib')
    
    
    xgb_pipe = build_xgb_pipeline(preprocessor)
    final_xgb = train_and_calibrate_xgb(xgb_pipe, X_train, y_train)
    
    # Save
    joblib.dump(final_xgb, "../models/xgb_calibrated.joblib")
    print("XGBoost Model Saved!")