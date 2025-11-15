import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# These constants are duplicated from preprocess.py to avoid import-time
# dependency on encoders.json (which this script is responsible for creating).
INPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
INPUT_PATH = INPUT_DIR / "train.csv"

DATA_TYPES = {
    "ID": "int64",
    "id_season": "int64",
    "aggregated_family": "string",
    "family": "string",
    "category": "string",
    "fabric": "string",
    "color_name": "string",
    "color_rgb": "string",
    "image_embedding": "string",
    "length_type": "string",
    "silhouette_type": "string",
    "waist_type": "string",
    "neck_lapel_type": "string",
    "sleeve_length_type": "string",
    "heel_shape_type": "string",
    "toecap_type": "string",
    "woven_structure": "string",
    "knit_structure": "string",
    "print_type": "string",
    "archetype": "string",
    "moment": "string",
    "phase_in": "string",
    "phase_out": "string",
    "life_cycle_length": "float64",
    "num_stores": "int64",
    "num_sizes": "int64",
    "has_plus_sizes": "int64",
    "price": "float64",
    "year": "int64",
    "num_week_iso": "int64",
    "weekly_sales": "float64",
    "weekly_demand": "float64",
    "Production": "float64",
}

EMBEDDING_DIM = 128
NUM_COLORS = 20


# Single JSON file that will hold all encoders/parameters
ENCODER_PATH = pathlib.Path(__file__).resolve().parent / "encoders.json"


def fit_categorical_encoders(df: pd.DataFrame, categorical_columns: list[str]) -> dict:
    mappings: dict[str, list[str]] = {}
    for col in categorical_columns:
        values = df[col].astype("string")
        _, uniques = pd.factorize(values, sort=True)
        mappings[col] = [str(u) for u in uniques.tolist()]
    return mappings


def fit_image_pca(df: pd.DataFrame, column: str) -> dict:
    parsed = [np.array(s.split(","), dtype="float32") for s in df[column]]
    emb_matrix = np.vstack(parsed)

    pca = PCA(n_components=EMBEDDING_DIM)
    pca.fit(emb_matrix)

    return {
        "mean": pca.mean_.astype("float32").tolist(),
        "components": pca.components_.astype("float32").tolist(),
    }


def fit_color_kmeans(df: pd.DataFrame, column: str) -> dict:
    def parse_rgb(s: str) -> np.ndarray:
        r, g, b = [float(x) for x in str(s).split(",")]
        return np.array([r, g, b], dtype="float32")

    rgb_matrix = np.vstack([parse_rgb(v) for v in df[column]])

    kmeans = KMeans(n_clusters=NUM_COLORS, n_init=10, random_state=42)
    kmeans.fit(rgb_matrix)

    return {
        "cluster_centers": kmeans.cluster_centers_.astype("float32").tolist(),
    }


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype=DATA_TYPES, delimiter=";") # type: ignore

    categorical_columns = [
        "aggregated_family",
        "family",
        "category",
        "fabric",
        "length_type",
        "silhouette_type",
        "waist_type",
        "neck_lapel_type",
        "sleeve_length_type",
        "heel_shape_type",
        "toecap_type",
        "woven_structure",
        "knit_structure",
        "print_type",
        "archetype",
        "moment",
    ]

    cat_mappings = fit_categorical_encoders(df, categorical_columns)
    emb_pca_params = fit_image_pca(df, "image_embedding")
    color_kmeans_params = fit_color_kmeans(df, "color_rgb")

    encoders = {
        "categorical_mappings": cat_mappings,
        "image_embedding_pca": emb_pca_params,
        "color_kmeans": color_kmeans_params,
    }

    with ENCODER_PATH.open("w", encoding="utf-8") as f:
        json.dump(encoders, f)

    print("Saved encoders to", ENCODER_PATH)
