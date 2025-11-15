# %%

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Load data
date_cols = ["phase_in", "phase_out"]

train = pd.read_csv(
    "../data/train.csv",
    delimiter=";",
    parse_dates=date_cols,
    infer_datetime_format=True,
)
# test = pd.read_csv("../data/test.csv", delimiter=';', parse_dates=date_cols,
#     infer_datetime_format=True,)


agg = train.groupby("ID")["weekly_sales"].sum()

X = train.drop(
    columns=[
        "phase_in",
        "knit_structure",
        "weekly_sales",
        "weekly_demand",
        "Production",
        "color_rgb",
        "image_embedding",
    ]
)
y = train["weekly_sales"]
print(y.describe())

# Encode
# X has some categorical columns (strings)
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
]  # string columns


# Label Encode categorical columns
X_enc = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col])
    label_encoders[col] = le  # save for later if needed


ts_cv = TimeSeriesSplit(
    n_splits=3,  # to keep the notebook fast enough on common laptops
    gap=20,  # 2 days data gap between train and test
    max_train_size=10000,  # keep train sets of comparable sizes
    test_size=2000,  # for 2 or 3 digits of precision in scores
)
all_splits = list(ts_cv.split(X, y))

train_idx, test_idx = all_splits[0]

X_train, X_test = X_enc.iloc[train_idx, :], X_enc.iloc[test_idx, :]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# Fit HGBR with Poisson loss
max_iter = 1000
gbrt_mean_poisson = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=max_iter,
    categorical_features=[X_enc.columns.get_loc(c) for c in categorical_cols],
)

gbrt_mean_poisson.fit(X_train, y_train)

# Predict
mean_predictions = gbrt_mean_poisson.predict(X_test)

print(mean_predictions[:10])


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


print(smape(y_test, mean_predictions))
# %%
gbrt_median = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, max_iter=max_iter
)
gbrt_median.fit(X_train, y_train)
median_predictions = gbrt_median.predict(X_test)

gbrt_percentile_5 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.05, max_iter=max_iter
)
gbrt_percentile_5.fit(X_train, y_train)
percentile_5_predictions = gbrt_percentile_5.predict(X_test)

gbrt_percentile_95 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.95, max_iter=max_iter
)
gbrt_percentile_95.fit(X_train, y_train)
percentile_95_predictions = gbrt_percentile_95.predict(X_test)
