"""Hybrid: Ridge regression followed by CatBoost using Ridge predictions as a feature."""

# %% IMPORTS
import pathlib
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from catboost import CatBoostRegressor
import load_data
import metrics


class HybridModel3:
    def __init__(self, ridge_model, xgb_model):
        self.ridge_model = ridge_model
        self.xgb_model = xgb_model

    def predict(self, X):
        X_aug = X.copy()
        X_aug = X_aug.fillna(0)
        X_aug["ridge_pred"] = self.ridge_model.predict(X_aug)
        return self.xgb_model.predict(X_aug)

if __name__ == "__main__":
    # %% LOAD DATA
    print("[INFO] Loading data...")
    X, y = load_data.load_data()
    X_train, X_test, y_train, y_test = load_data.split_data(X, y)
    #replace nanas with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

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

    # %% TRAIN XGBOOST
    print("[INFO] Training XGBoost regressor...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train_aug,
        y_train,
        eval_set=[(X_test_aug, y_test)],
        verbose=100,
    )
    print("[INFO] XGBoost training complete.")


    # %% EVALUATE
    print("[INFO] Evaluating hybrid model...")
    hybrid_model = HybridModel3(ridge_model, xgb_model)
    y_pred = hybrid_model.predict(X_test)
    y_train_pred = hybrid_model.predict(X_train)
    
    feature_importances = xgb_model.feature_importances_
    feature_names = X_train_aug.columns
    importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importances)}
    additional_properties = {"feature_importance": importance_dict}

    metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="rich_xgb", additional_properties=additional_properties)
    print("[INFO] Evaluation complete.")

    # %% SAVE MODELS
    print("[INFO] Saving trained models...")
    TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = TRAINED_DIR / "rich_xgb_model.pkl"
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(hybrid_model, f)
    print(f"[INFO] Saved trained hybrid model to {OUTPUT_PATH}")
