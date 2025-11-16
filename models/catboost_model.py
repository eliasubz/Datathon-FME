"""Train a CatBoost model on the cleaned data.

This mirrors the interface and report style of `xgb_model.py`.
"""

# %% IMPORTS
import pathlib
import pickle

from catboost import CatBoostRegressor

import load_data
import metrics


# %% LOAD DATA
X, y = load_data.load_data()
X_train, X_test, y_train, y_test = load_data.split_data(X, y)


# %% TRAIN MODEL
model = CatBoostRegressor(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=3.0,
    iterations=1500,
    random_seed=42,
    subsample=0.8,
    rsm=0.6,
    od_type="Iter",
    od_wait=50,
    verbose=100,
)

print("Training CatBoost model...")
model.fit(
    # X, y,
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
    verbose=100,
)


# %% EVALUATE
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

feature_importances = model.get_feature_importance()
feature_names = X.columns
importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importances)}
additional_properties = {"feature_importance": importance_dict}

metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="catboost_model", additional_properties=additional_properties)


# %% SAVE TRAINED MODEL
TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = TRAINED_DIR / "catboost_model.pkl"

with MODEL_PATH.open("wb") as f:
    pickle.dump(model, f)

print(f"Saved trained CatBoost model to {MODEL_PATH}")
