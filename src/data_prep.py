import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.copy()
    
    # Numerical Data
    num_cols = df.select_dtypes(include='number')
    df[num_cols.columns] = num_cols.fillna(num_cols.median())

    # Categorical
    cat_col = df.select_dtypes(include='object')
    df[cat_col.columns] = df[cat_col.columns].fillna('Unknown')

    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"File saved successfully to {output_path}")

if __name__ == "__main__":
    RAW_PATH = 'C:/Users/Pratik/DS/credit-risk-ml/data/raw/application_train.csv'
    PROCESSED_PATH = 'C:/Users/Pratik/DS/credit-risk-ml/data/processed/clean_data.csv'
    data = load_data(RAW_PATH)
    clean_df = clean_data(data)
    save_data(clean_df, PROCESSED_PATH)