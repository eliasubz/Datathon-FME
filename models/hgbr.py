# %%

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import load_data


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit  # Assuming this is from sklearn

# 1. Define constants
date_cols = ["phase_in", "phase_out"]
drop_cols = [
    "phase_in",
    "knit_structure",
    "weekly_sales",
    "weekly_demand",
    "Production",
    "color_rgb",
    "image_embedding",
]
categorical_cols = [
    "aggregated_family",
    "family",
    "category",
    "fabric",
    "color_name",
    "length_type",
    "sleeve_length_type",
    "heel_shape_type",
    "toecap_type",
    "woven_structure",
    "waist_type",
    "phase_out",
    "print_type",
    "archetype",
    "moment",
    "silhouette_type",
    "neck_lapel_type",
]


# 2. Function to load and clean data
def load_and_clean_data(file_path, is_train=True, agg_weeks=False):
    """Loads, parses dates, and cleans the DataFrame."""
    df = pd.read_csv(
        file_path,
        delimiter=";",
        parse_dates=date_cols,
        infer_datetime_format=True,
    )

    # Columns that cannot be used in X when aggregating, because they vary weekly
    weekly_variable_cols = ["num_week_iso", "weekly_sales", "weekly_demand"]

    if agg_weeks:

        # Sum weekly sales per ID → this becomes the target
        agg_df = df.groupby("ID", as_index=False)["weekly_sales"].mean()

        # y = aggregated target
        y = agg_df["weekly_sales"]

        # Merge attributes (stable columns only)
        # Take the FIRST occurrence of each ID for stable attributes
        attr_cols = df.drop(
            columns=weekly_variable_cols + ["weekly_sales"], errors="ignore"
        )
        attr_df = attr_cols.groupby("ID", as_index=False).first()

        # X = stable attributes
        X = attr_df

        # Drop unwanted global drop_cols
        cols_to_drop = [c for c in drop_cols if c in X.columns]
        X = X.drop(columns=cols_to_drop, errors="ignore")

        return X, y

    # if f_x_pred is True:
    # For prediction, we don't have weekly sales to aggregate
    # Just drop the weekly variable columns
    # df = df.drop(columns=weekly_variable_cols, errors="ignore")

    # Drop columns. Need to handle 'weekly_sales' if it's the target in train.
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    X_clean = df.drop(columns=cols_to_drop)

    y_target = None
    if is_train and "weekly_sales" in df.columns:
        y_target = df["weekly_sales"]

    return X_clean, y_target


# 3. Load and clean train and test data
train, y_train_full = load_and_clean_data(
    "../data/train.csv", is_train=True, agg_weeks=True
)
r_test, _ = load_and_clean_data("../data/test.csv", is_train=False)


r_test["year"] = 2025
r_test["num_week_iso"] = 25


X = train.copy()
X_test_full = r_test.copy()

print(y_train_full.describe())

# 4. Encode Categorical Columns (Fit on Train, Transform on Train & Test)
X_enc = X.copy()
X_test_enc = X_test_full.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit the encoder only on the training data (X_enc)
    le.fit(pd.concat([X_enc[col], X_test_enc[col]]).astype(str).unique())

    # Transform both the training data and the full test data
    X_enc[col] = le.transform(X_enc[col].astype(str))
    X_test_enc[col] = le.transform(X_test_enc[col].astype(str))
    label_encoders[col] = le  # save for later if needed


# 5. Time Series Split on the encoded training data
ts_cv = TimeSeriesSplit(
    n_splits=3,
    gap=20,
    test_size=2000,
)
all_splits = list(ts_cv.split(X_enc, y_train_full))

train_idx, test_idx = all_splits[0]

X_train, X_cv = X_enc.iloc[train_idx, :], X_enc.iloc[test_idx, :]
y_train, y_cv = y_train_full.iloc[train_idx], y_train_full.iloc[test_idx]

# Fit HGBR with Poisson loss
max_iter = 1000
gbrt_mean_poisson = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=max_iter,
    categorical_features=[X_enc.columns.get_loc(c) for c in categorical_cols],
)

gbrt_mean_poisson.fit(X_train, y_train)

# Predict
mean_predictions = gbrt_mean_poisson.predict(X_test_enc)
# mean_predictions = gbrt_mean_poisson.predict(X_cv)

# Soft improve predictions
mean_predictions[mean_predictions < 0] = np.percentile(mean_predictions, 20)

# Add buffer
mean_predictions *= 1.125


# Saving submission file for CV setx
df_submission_cv = r_test[["ID"]].copy()
df_submission_cv["Production"] = mean_predictions * r_test["life_cycle_length"]
df_summed_production = df_submission_cv.groupby("ID")["Production"].sum().reset_index()
output_csv_filename = "../output/cv_predictions_output.csv"
df_summed_production.to_csv(output_csv_filename, index=False)
print(df_summed_production.describe())


print(f"\n✅ Data saved to DataFrame: df_submission_cv")


print(mean_predictions[:10])


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


print(smape(y_cv, mean_predictions))

# Prediction works on trying to predict the weekly sales.
