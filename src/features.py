import numpy as np
import pandas as pd
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utility Functions 

# Checking if columns exist
def check_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# Safe Division (0 Divide Error)
def safe_divide(a, b):
    return a / b.replace(0, np.nan)

# Feature Engineering Functions

# Function to create Numerical Features
def create_numerical_features(df):
    df = df.copy()

    # Check if columns are there
    required_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT',
        'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]

    check_columns(df, required_cols)

    # Log Transforms
    log_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for col in log_cols:
        df[f'{col}_LOG'] = np.log1p(df[col])
    
    # Age 
    df['AGE_YEARS'] = - df['DAYS_BIRTH'] / 365

    # Employement Cleaning
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED'].notna(),np.nan) / 365 

    # Drop Raw columns 
    df.drop(columns=['DAYS_EMPLOYED','DAYS_BIRTH'],inplace=True)

    return df

# Function for creating ratio features
def create_ratios(df):
    df = df.copy()

    required_cols = [
        'AMT_CREDIT', 'AMT_INCOME_TOTAL',
        'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS'
    ]
    check_columns(df, required_cols)

    # Creating Ratio Features
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    return df

# Function to create binary features
def create_binary_features(df):
    df = df.copy()

    required_cols = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'EMPLOYED_YEARS'
    ]
    check_columns(df, required_cols)

    # Creating Flag Features
    df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype('Int64')
    df['IS_SINGLE'] = (df['CNT_FAM_MEMBERS'] == 1).astype('Int64')
    df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype('Int64')
    df['HAS_REALTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype('Int64')

    # Conditional logic for employement
    df['LONG_EMPLOYED'] = np.where(df['EMPLOYED_YEARS'].isna(),np.nan,(df['EMPLOYED_YEARS'] > 5).astype(int))

    df.drop(columns=['FLAG_OWN_CAR', 'FLAG_OWN_REALTY'], inplace=True)

    return df

# Function to Handle Categorical Features
def encode_categorical(df):
    df = df.copy()

    # Encoding Education level 
    education_map = {
    'Lower secondary': 0, 'Secondary / secondary special': 1,
    'Incomplete higher': 2, 'Higher education': 3,
    'Academic degree': 4
    }
    if 'NAME_EDUCATION_TYPE' in df.columns:
        df['NAME_EDUCATION_ENC'] = df['NAME_EDUCATION_TYPE'].map(education_map).fillna(-1)

    # Mapping Income
    income_map = {
    "Working": "Employed", "Commercial associate": "Employed", "State servant": "Employed",
    "Pensioner": "Pension", "Student": "Non-working", "Unemployed": "Non-working",
    "Maternity leave": "Non-working", "Businessman": "Self-employed", "Other": "Other"
    }
    if "NAME_INCOME_TYPE" in df.columns:
        df["INCOME_GROUP"] = df["NAME_INCOME_TYPE"].map(income_map).fillna("Unknown")

    return df


# Pipeline

def run_feature_engineering(input_path, output_path):
    logging.info("Loading data...")
    df = pd.read_csv(input_path)

    logging.info("Running feature engineering pipeline...")
    df = (df.pipe(create_numerical_features)
            .pipe(create_ratios)
            .pipe(create_binary_features)
            .pipe(encode_categorical))
    
    logging.info("Saving output...")
    df.to_csv(output_path, index=False)
    print(f"Features engineered and saved to {output_path}")


if __name__ == "__main__":
    IN_PATH = "C:/Users/Pratik/DS/credit-risk-ml/data/processed/clean_data.csv"
    OUT_PATH = "C:/Users/Pratik/DS/credit-risk-ml/data/processed/credit_features_v1.csv"
    run_feature_engineering(IN_PATH, OUT_PATH)
