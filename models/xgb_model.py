"""Train an XGBoost model on the cleaned data.

This follows the simple script style used in preprocess.py.
"""

# %% IMPORTS
import xgboost as xgb

import load_data
import metrics


# %% LOAD DATA
X, y = load_data.load_data()
X_train, X_test, y_train, y_test = load_data.split_data(X, y)


# %% TRAIN MODEL
model = xgb.XGBRegressor(
	n_estimators=100,
	learning_rate=0.05,
	max_depth=8,
	subsample=0.8,
	colsample_bytree=0.8,
	objective="reg:squarederror",
	tree_method="hist",
	random_state=42,
	n_jobs=-1,
)

model.fit(X_train, y_train)


# %% EVALUATE
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
metrics.print_regression_report(y_test, y_pred, y_train, y_train_pred)