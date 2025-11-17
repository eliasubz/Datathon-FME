"""Hybrid model that uses Ridge and a multi-head CatBoost model predictions as features for a final CatBoost model."""

# %% IMPORTS
import pathlib
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
import load_data
import metrics


class HybridModel2:
    def __init__(self, ridge_model, multihead_cat_model, final_cat_model):
        self.ridge_model = ridge_model
        self.multihead_cat_model = multihead_cat_model
        self.final_cat_model = final_cat_model

    def predict(self, X):
        X_aug = X.copy()
        X_aug = X_aug.fillna(0)

        # Get Ridge predictions
        X_aug["ridge_pred"] = self.ridge_model.predict(X_aug)

        # Get multi-head CatBoost predictions
        multihead_preds = self.multihead_cat_model.predict(X_aug)
        X_aug["multihead_pred_y"] = multihead_preds[:, 0]
        X_aug["multihead_pred_y_per_store"] = multihead_preds[:, 1]
        X_aug["multihead_pred_y_sales"] = multihead_preds[:, 2]

        return self.final_cat_model.predict(X_aug)

if __name__ == "__main__":
    # %% LOAD DATA
    print("[INFO] Loading data...")
    X, y = load_data.load_data()
    X_train, X_test, y_train, y_test = load_data.split_data(X, y)
    
    # Replace NaNs with 0
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

    # %% LOAD MULTI-HEAD CATBOOST MODEL
    print("[INFO] Loading multi-head CatBoost model...")
    TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
    MULTIHEAD_MODEL_PATH = TRAINED_DIR / "multihead_cat_model.pkl"
    with MULTIHEAD_MODEL_PATH.open("rb") as f:
        multihead_cat_model = pickle.load(f)
    print("[INFO] Multi-head CatBoost model loaded.")

    # %% AUGMENT FEATURES
    print("[INFO] Augmenting features with model predictions...")
    
    def augment_features(X, ridge_model, multihead_model):
        X_aug = X.copy()
        
        # Add Ridge predictions
        X_aug["ridge_pred"] = ridge_model.predict(X)
        
        # Add multi-head CatBoost predictions
        multihead_preds = multihead_model.predict(X)
        X_aug["multihead_pred_y"] = multihead_preds[:, 0]
        X_aug["multihead_pred_y_per_store"] = multihead_preds[:, 1]
        X_aug["multihead_pred_y_sales"] = multihead_preds[:, 2]
        
        return X_aug

    X_train_aug = augment_features(X_train, ridge_model, multihead_cat_model)
    X_test_aug = augment_features(X_test, ridge_model, multihead_cat_model)

    # %% TRAIN FINAL CATBOOST
    print("[INFO] Training final CatBoost regressor...")
    final_cat_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.01,
        l2_leaf_reg=3.0,
        iterations=3000,
        random_seed=42,
        subsample=0.8,
        rsm=0.6,
        od_type="Iter",
        od_wait=100,
        verbose=100,
    )
    final_cat_model.fit(
        # X, y
        X_train_aug,
        y_train,
        eval_set=(X_test_aug, y_test),
        use_best_model=True,
    )
    print("[INFO] Final CatBoost training complete.")

    # %% EVALUATE
    print("[INFO] Evaluating hybrid model...")
    hybrid_model = HybridModel2(ridge_model, multihead_cat_model, final_cat_model)
    y_pred = hybrid_model.predict(X_test)
    y_train_pred = hybrid_model.predict(X_train)
    
    feature_importances = final_cat_model.get_feature_importance()
    feature_names = X_train_aug.columns
    importance_dict = {name: float(score) for name, score in zip(feature_names, feature_importances)}
    additional_properties = {"feature_importance": importance_dict}

    metrics.regression_report(y_test, y_pred, y_train, y_train_pred, model_name="hybrid_model", additional_properties=additional_properties)
    print("[INFO] Evaluation complete.")

    # %% SAVE MODELS
    print("[INFO] Saving trained models...")
    TRAINED_DIR = pathlib.Path(__file__).resolve().parent / "trained"
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = TRAINED_DIR / "hybrid_model.pkl"
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(hybrid_model, f)
    print(f"[INFO] Saved trained hybrid model to {OUTPUT_PATH}")
