# %%
import pandas as pd
import numpy as np
import pathlib 

path = pathlib.Path(__file__).resolve().parents[1] / "clean_data" / "train.parquet"

# %% 
def load_data():
    """Load cleaned data from parquet file."""
    df = pd.read_parquet(path)
    X = df.drop(columns=["weekly_demand", "Production"])

    y = df["weekly_demand"]
    return X, y
    
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