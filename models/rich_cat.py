"""Hybrid: Ridge regression followed by CatBoost using Ridge predictions as a feature."""

# %% IMPORTS
import pathlib
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
import load_data
import metrics


class HybridModel:
    def __init__(self, ridge_model, cat_model):
        self.ridge_model = ridge_model
        self.cat_model = cat_model

    def predict(self, X):
        X_aug = X.copy()
        X_aug["ridge_pred"] = self.ridge_model.predict(X)
        return self.cat_model.predict(X_aug)

if __name__ == "__main__":
    # %% LOAD DATA
    print("[INFO] Loading data...")
    X, y = load_data.load_data()
    X_train, X_test, y_train, y_test = load_data.split_data(X, y)

    # %% TRAIN RIDGE
    print("[INFO] Training Ridge regression...")
    ridge_model = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=1.0, random_state=42),
    )
    ridge_model.fit(X_train, y_train)
    print("[INFO] Ridge training complete.")

    # %% ADD RIDGE PREDICTIONS AS FEATURE
    print("[INFO] Adding Ridge predictions as feature...")
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug["ridge_pred"] = ridge_model.predict(X_train)
    X_test_aug["ridge_pred"] = ridge_model.predict(X_test)

    # %% TRAIN CATBOOST
    print("[INFO] Training CatBoost regressor...")
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        iterations=3000,
        random_seed=42,
        subsample=0.8,
        rsm=0.6,
        od_type="Iter",
        od_wait=100,
        verbose=100,
    )
    cat_model.fit(
        X_train_aug,
        y_train,
        eval_set=(X_test_aug, y_test),
        use_best_model=True,
    )
    print("[INFO] CatBoost training complete.")


    # %% EVALUATE
    print("[INFO] Evaluating hybrid model...")
    hybrid_model = HybridModel(ridge_model, cat_model)
    y_pred = hybrid_model.predict(X_test)
    y_train_pred = hybrid_model.predict(X_train)
    metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="rich_cat")
    print("[INFO] Evaluation complete.")

    # %% SAVE MODELS
    print("[INFO] Saving trained models...")
    TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = TRAINED_DIR / "rich_cat_model.pkl"
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(hybrid_model, f)
    print(f"[INFO] Saved trained hybrid model to {OUTPUT_PATH}")
