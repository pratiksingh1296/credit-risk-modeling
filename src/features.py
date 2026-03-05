import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/Pratik/DS/credit-risk-ml/data/processed/clean_data.csv')

# Numeric Feature Transformations
df['AMT_INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
df['AMT_CREDIT_LOG'] = np.log1p(df['AMT_CREDIT'])
df['AMT_ANNUITY_LOG'] = np.log1p(df['AMT_ANNUITY'])
df['AMT_GOODS_LOG'] = np.log1p(df['AMT_GOODS_PRICE'])

df['AGE_YEARS'] = - df['DAYS_BIRTH'] / 365
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED'].notna(),np.nan) / 365 
df.drop(columns=['DAYS_EMPLOYED','DAYS_BIRTH'],inplace=True)

# Creating Ratio Features
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

# Binary Features
df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype('Int64')
df['IS_SINGLE'] = (df['CNT_FAM_MEMBERS'] == 1).astype('Int64')
df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype('Int64')
df['HAS_REALTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype('Int64')
df['LONG_EMPLOYED'] = np.where(
    df['EMPLOYED_YEARS'].isna(),
    np.nan,
    (df['EMPLOYED_YEARS'] > 5).astype(int)
)
df.drop(columns=['FLAG_OWN_CAR', 'FLAG_OWN_REALTY'], inplace=True)

# Ordinal Category Encoding
# Encoding Education level 
education_map = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
}

df['NAME_EDUCATION_ENC'] = df['NAME_EDUCATION_TYPE'].map(education_map)

# Grouping Income Types
income_map = {
    "Working": "Employed",
    "Commercial associate": "Employed",
    "State servant": "Employed",

    "Pensioner": "Pension",
    "Student": "Non-working",

    "Unemployed": "Non-working",
    "Maternity leave": "Non-working",

    "Businessman": "Self-employed",

    "Other": "Other"
}

df["INCOME_GROUP"] = df["NAME_INCOME_TYPE"].map(income_map)

pd.get_dummies(df["INCOME_GROUP"], prefix="INC", drop_first=True)

df.to_csv("C:/Users/Pratik/DS/credit-risk-ml/data/processed/credit_features_v1.csv", index=False)