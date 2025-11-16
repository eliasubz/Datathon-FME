# %%
import pandas as pd
import numpy as np
import pathlib 

path = pathlib.Path(__file__).resolve().parents[1] / "clean_data" / "train.parquet"

# %% 
def load_data():
    """Load cleaned data from parquet file."""
    df = pd.read_parquet(path)
    # Drop all target columns from X
    X = df.drop(columns=[col for col in ["y", "y_per_store", "y_sales"] if col in df.columns])
    y = df["y"]
    # y_per_store and y_sales are available for future use
    return X, y
    
def load_data_multihead():
    """Load cleaned data and return X and multi-output y vector (y, y_per_store, y_sales)."""
    df = pd.read_parquet(path)
    X = df.drop(columns=[col for col in ["y", "y_per_store", "y_sales"] if col in df.columns])
    y_multi = df[[col for col in ["y", "y_per_store", "y_sales"] if col in df.columns]].to_numpy(dtype=float)
    return X, y_multi

def split_data(X, y): 
    # use the last 10% of the data as test set
    n_samples = X.shape[0]
    split_index = int(n_samples * 0.9)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_data()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

# %% MAIN SCRIPT
X, y = load_data()
X.head()