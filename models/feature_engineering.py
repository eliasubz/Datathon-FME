import pandas as pd
import numpy as np

def create_ts_features(df, date_col='date_week', target_col='weekly_sales',
                       lags=[1,2,3,4], rolling_windows=[2,4], expanding=True, external=None):
    """
    Create time series features for ML models.

    Parameters:
    - df: DataFrame with datetime and target columns
    - date_col: datetime column name
    - target_col: target variable
    - lags: list of lag periods (weeks) to create
    - rolling_windows: list of window sizes for rolling stats
    - expanding: whether to include expanding mean
    - external: list of external regressor columns to include
    
    Returns:
    - df_features: DataFrame with features
    """
    
    df = df.copy()
    df = df.sort_values(date_col)
    
    # Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling stats (mean, std, min, max)
    for window in rolling_windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        df[f'{target_col}_roll_min_{window}'] = df[target_col].shift(1).rolling(window).min()
        df[f'{target_col}_roll_max_{window}'] = df[target_col].shift(1).rolling(window).max()
    
    # Expanding stats
    if expanding:
        df[f'{target_col}_expanding_mean'] = df[target_col].shift(1).expanding().mean()
        df[f'{target_col}_expanding_std'] = df[target_col].shift(1).expanding().std()
    
    # Date/time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    
    # Cyclical encoding for month and dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # External regressors
    if external:
        for col in external:
            df[col] = df[col]
    
    # Drop rows with NaNs caused by lags/rolling
    df_features = df.dropna().reset_index(drop=True)
    
    return df_features

# Example usage
df_features = create_ts_features(train, date_col='date_week', target_col='weekly_sales',
                                 lags=[1,2,3,4], rolling_windows=[2,4], external=['promotion'])
print(df_features.head())

