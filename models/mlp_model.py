"""Train an MLPRegressor model on the cleaned data.

This mirrors the interface and report style of the other models.
"""

# %% IMPORTS
import pathlib
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import load_data
import metrics

# %% LOAD DATA
print("[INFO] Loading data...")
X, y = load_data.load_data()
X_train, X_test, y_train, y_test = load_data.split_data(X, y)

# %% TRAIN MODEL
print("[INFO] Training MLPRegressor model...")
model = make_pipeline(
    StandardScaler(with_mean=False),
    MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        verbose=100,
    ),
)
model.fit(X_train, y_train)
print("[INFO] Finished training.")

# %% EVALUATE
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="mlp_model")
print("[INFO] Evaluation complete.")

# %% SAVE TRAINED MODEL
print("[INFO] Saving trained model...")
TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = TRAINED_DIR / "mlp_model.pkl"
with MODEL_PATH.open("wb") as f:
    pickle.dump(model, f)
print(f"[INFO] Saved trained MLPRegressor model to {MODEL_PATH}")
