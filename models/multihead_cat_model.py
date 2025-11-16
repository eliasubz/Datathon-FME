"""Train a multi-output CatBoost model on the cleaned data using all target heads."""

# %% IMPORTS
import pathlib
import pickle
from catboost import CatBoostRegressor
import load_data
import metrics

# %% LOAD DATA
X, y_multi = load_data.load_data_multihead()
X_train, X_test = X.iloc[:int(0.9*len(X))], X.iloc[int(0.9*len(X)):]
y_train, y_test = y_multi[:int(0.9*len(X))], y_multi[int(0.9*len(X)):]  # shape: (n_samples, 3)

# %% TRAIN MODEL
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=3.0,
    iterations=1500,
    random_seed=42,
    rsm=0.6,
    od_type="Iter",
    od_wait=50,
    verbose=100,
)

print("Training multi-head CatBoost model...")
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
    verbose=100,
)

# %% EVALUATE
# Predict all heads
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Feature importances
feature_importances = model.get_feature_importance()
feature_names = list(X_train.columns)
importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importances)}
additional_properties = {"feature_importance": importance_dict}

# Save metrics for each head
for i, head_name in enumerate(["y", "y_per_store", "y_sales"]):
    metrics.regression_report(
        y_test[:, i], y_pred[:, i], y_train[:, i], y_train_pred[:, i],
        model_name=f"multihead_cat_{head_name}",
        additional_properties=additional_properties
    )

# %% SAVE TRAINED MODEL
TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = TRAINED_DIR / "multihead_cat_model.pkl"

with MODEL_PATH.open("wb") as f:
    pickle.dump(model, f)

print(f"Saved trained multi-head CatBoost model to {MODEL_PATH}")
