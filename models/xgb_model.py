"""Train an XGBoost model on the cleaned data.

This follows the simple script style used in preprocess.py.
"""

# %% IMPORTS
import pathlib
import pickle

import xgboost as xgb

import load_data
import metrics


# %% LOAD DATA
X, y = load_data.load_data()
X_train, X_test, y_train, y_test = load_data.split_data(X, y)


# %% TRAIN MODEL
model = xgb.XGBRegressor(
	n_estimators=1500,
	learning_rate=0.02,
	min_child_weight=5,
	max_depth=6,
	subsample=0.8,
	colsample_bytree=0.6,
	objective="reg:squarederror",
	tree_method="hist",
	random_state=42,
	n_jobs=-1,
	gamma=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
	early_stopping_rounds=50,
)

model.fit(
	# X, y,
	X_train, y_train,
	eval_set=[(X_test, y_test)],
	verbose=100
)


# %% EVALUATE
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

feature_importances = model.feature_importances_
feature_names = X.columns
importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importances)}
additional_properties = {"feature_importance": importance_dict}

metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="xgb_model", additional_properties=additional_properties)


# %% SAVE TRAINED MODEL
TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = TRAINED_DIR / "xgb_model.pkl"

with MODEL_PATH.open("wb") as f:
	pickle.dump(model, f)

print(f"Saved trained XGBoost model to {MODEL_PATH}")