"""Train a Ridge regression model on the cleaned data.

This mirrors the interface and report style of `xgb_model.py`.
"""

# %% IMPORTS
print("imports...")
import pathlib
import pickle

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import load_data
import metrics


# %% LOAD DATA
X, y = load_data.load_data()
X_train, X_test, y_train, y_test = load_data.split_data(X, y)


# %% TRAIN MODEL
# Use a pipeline with standardization + Ridge
model = make_pipeline(
    StandardScaler(with_mean=False),  # with_mean=False for sparse / many one-hot cols
    Ridge(
        alpha=1.0,
        random_state=42,
    ),
)

print("Starting Ridge model training...")
model.fit(X_train, y_train)
print("Finished Ridge model training.")


# %% EVALUATE
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("Evaluating model...")
metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="ridge_model")
print("Evaluation complete.")


# %% SAVE TRAINED MODEL
TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = TRAINED_DIR / "ridge_model.pkl"

with MODEL_PATH.open("wb") as f:
    pickle.dump(model, f)
print(f"Saved trained Ridge model to {MODEL_PATH}")
