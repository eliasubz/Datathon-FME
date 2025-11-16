import pathlib
import pickle
from typing import List

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
CLEAN_DATA_DIR = ROOT / "clean_data"
TRAINED_MODELS_DIR = pathlib.Path(__file__).resolve().parent / "trained"
OUTPUT_DIR = ROOT / "output"

TEST_FEATURES_PATH = CLEAN_DATA_DIR / "test_extended.parquet"
ID_COLUMN = "ID"


def load_test_features() -> pd.DataFrame:
    """Load test features, ensuring the ID column is present and first."""

    df = pd.read_parquet(TEST_FEATURES_PATH)

    if ID_COLUMN not in df.columns:
        raise KeyError(f"Expected ID column '{ID_COLUMN}' in {TEST_FEATURES_PATH}")

    # Ensure ID is the first column for clarity (not strictly required)
    cols = [ID_COLUMN] + [c for c in df.columns if c != ID_COLUMN]
    df = df[cols]
    return df


def find_trained_models() -> List[pathlib.Path]:
    """Return a list of all model files in the trained models directory."""

    if not TRAINED_MODELS_DIR.exists():
        raise FileNotFoundError(f"Trained models directory not found: {TRAINED_MODELS_DIR}")

    return sorted([p for p in TRAINED_MODELS_DIR.iterdir() if p.is_file()])


def load_model(model_path: pathlib.Path):
    with model_path.open("rb") as f:
        return pickle.load(f)


def make_predictions_for_model(model_path: pathlib.Path, features: pd.DataFrame) -> pd.DataFrame:
    """Run a single model on the test features and return aggregated predictions.

    The model is assumed to expose a ``predict`` method.
    Predictions are summed per ID and written to an output CSV file whose
    name is derived from the model filename.
    """

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    X_test = features
    print(f"Running predictions for model {model_path.name} on shape {X_test.shape}")
    preds = model.predict(X_test)

    # Attach ID and aggregate by ID (sum of predictions per ID)
    pred_df = pd.DataFrame({ID_COLUMN: features[ID_COLUMN].values, "prediction": np.asarray(preds, dtype="float64")})
    agg_df = pred_df.groupby(ID_COLUMN, as_index=False)["prediction"].sum()

    # Save to output file with the same basename as the model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{model_path.name}.csv"

    # Required submission format: ID;weekly_demand
    out = agg_df.rename(columns={"prediction": "weekly_demand"}) # type: ignore
    #convert both cols to int
    out[ID_COLUMN] = out[ID_COLUMN].astype(int)
    out["weekly_demand"] = out["weekly_demand"].astype(int)
    out.to_csv(output_path, sep=";", index=False)


    print(f"Saved predictions for {model_path.name} to {output_path}")
    return out


def main() -> None:
    features = load_test_features()
    model_paths = find_trained_models()

    if not model_paths:
        raise RuntimeError(f"No trained model files found in {TRAINED_MODELS_DIR}")

    print(f"Found {len(model_paths)} trained model(s): {[p.name for p in model_paths]}")

    for model_path in model_paths:
        make_predictions_for_model(model_path, features)


if __name__ == "__main__":
    main()