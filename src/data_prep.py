import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/Pratik/DS/credit-risk-ml/data/raw/application_train.csv')

# Handling Missing Values in Numerical Columns
num_cols = df.select_dtypes(include='number')
df[num_cols.columns] = num_cols.fillna(num_cols.median())

# Handling Missing Values in Category Columns
cat_col = df.select_dtypes(include='object')
df[cat_col.columns] = df[cat_col.columns].fillna('Unknown')

df.to_csv('C:/Users/Pratik/DS/credit-risk-ml/data/processed/clean_data.csv')