# %%
import pandas as pd
import numpy as np
import pathlib 

path = pathlib.Path(__file__).resolve().parents[1] / "clean_data" / "train.parquet"

# %% 
def load_data():
    """Load cleaned data from parquet file."""
    df = pd.read_parquet(path)
    X = df.drop(columns=["weekly_demand"])
    y = df["weekly_demand"]
    return X, y
    