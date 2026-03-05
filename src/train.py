import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

df = pd.read_csv("C:/Users/Pratik/DS/credit-risk-ml/data/processed/credit_features_v1.csv")

# Drop Columns for V1
df.drop(columns=['NAME_INCOME_TYPE','OCCUPATION_TYPE','NAME_TYPE_SUITE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','FONDKAPREMONT_MODE'],inplace=True)

# Split 
X = df.drop(columns=['TARGET'])
y = df['TARGET']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Risk Features 
income_risk_map = (X_train.assign(TARGET=y_train).groupby('INCOME_GROUP')['TARGET'].mean())
org_risk_map = (X_train.assign(TARGET=y_train).groupby('ORGANIZATION_TYPE')['TARGET'].mean())

X_train['INCOME_RISK'] = X_train['INCOME_GROUP'].map(income_risk_map)
X_test['INCOME_RISK'] = X_test['INCOME_GROUP'].map(income_risk_map)
X_train['ORG_RISK'] = X_train['ORGANIZATION_TYPE'].map(org_risk_map)
X_test['ORG_RISK'] = X_test['ORGANIZATION_TYPE'].map(org_risk_map)

# Drop Columns No Longer Needed
X_train = X_train.drop(columns=['INCOME_GROUP','ORGANIZATION_TYPE'])
X_test  = X_test.drop(columns=['INCOME_GROUP','ORGANIZATION_TYPE'])

# Saving Test Data
dump((X_test, y_test),"C:/Users/Pratik/DS/credit-risk-ml/models/test_data.joblib")

# Logistic Regression 
num_features = [
    'AMT_INCOME_TOTAL',
    'AGE_YEARS',
    'EMPLOYED_YEARS',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'INCOME_RISK',
    'ORG_RISK'
]

bin_features = [
    'HAS_CHILDREN',
    'IS_SINGLE',
    'HAS_CAR',
    'HAS_REALTY',
    'LONG_EMPLOYED'
]

cat_features = [
    'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'CODE_GENDER',
    'WEEKDAY_APPR_PROCESS_START'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear',
    ))
])
model.fit(X_train, y_train)

dump(model, "C:/Users/Pratik/DS/credit-risk-ml/models/logres_baseline.joblib")