import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def build_pipeline(preprocessor, model_type='logreg'):
    if model_type == 'logreg':
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    elif model_type == 'xgb':
        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=9, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='auc'
        )
    
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('clf', clf)
    ])


def train_and_save_baseline(X_train, y_train, X_test, y_test, preprocessor, model_type, output_path):

    # Validate Columns
    expected_cols = []
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            expected_cols.extend(cols)

    missing = set(expected_cols) - set(X_train.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    # Align Columns
    X_train = X_train[expected_cols]
    X_test = X_test[expected_cols]


    pipeline = build_pipeline(preprocessor, model_type)
    pipeline.fit(X_train, y_train)

    # Quick Validation
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"[{model_type.upper()}] Trained. Test AUC: {auc:.4f}")

    # Save
    joblib.dump(pipeline, output_path)
    return pipeline

if __name__ == "__main__":
    # Paths
    DATA_PATH = "C:/Users/Pratik/DS/credit-risk-ml/data/processed/credit_features_v1.csv"
    PREPROCESSOR_PATH = "C:/Users/Pratik/DS/credit-risk-ml/models/preprocessor_fit.joblib"
    MODEL_DIR = "C:/Users/Pratik/DS/credit-risk-ml/models/"

    # Load Data & Preprocessor
    df = pd.read_csv(DATA_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Split and Save Data (Saving splits ensures all other scripts use the SAME data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    joblib.dump((X_train, y_train), f"{MODEL_DIR}train_data.joblib")
    joblib.dump((X_test, y_test), f"{MODEL_DIR}test_data.joblib")

    # Train Both Models
    # Train Logistic Regression
    train_and_save_baseline(
        X_train, y_train, X_test, y_test, preprocessor, 
        model_type='logreg', 
        output_path=f"{MODEL_DIR}logreg_baseline.joblib"
    )

    # Train XGBoost
    train_and_save_baseline(
        X_train, y_train, X_test, y_test, preprocessor, 
        model_type='xgb', 
        output_path=f"{MODEL_DIR}xgb_baseline.joblib"
    )