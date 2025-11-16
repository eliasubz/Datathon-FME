# %% IMPORTS 
import json
import pandas as pd
import numpy as np
import pathlib

# %% CONSTANTS AND PATHS

INPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
TRAIN_INPUT_PATH = INPUT_DIR / "train.csv"
TEST_INPUT_PATH = INPUT_DIR / "test_extended.csv"

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "clean_data"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.parquet"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test_extended.parquet"

EMBEDDING_DIM = 128  # target dimension after PCA

DATA_TYPES = {
    "ID": "int64",
    "id_season": "int64",
    "aggregated_family": "string",
    "family": "string",
    "category": "string",
    "fabric": "string",
    "color_name": "string",
    "color_rgb": "string",  # keep as string; could be parsed later
    "image_embedding": "string",  # text/embedding representation
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
    "has_plus_sizes": "int64",  # 0/1 flag
    "price": "float64",
    "year": "int64",
    "num_week_iso": "int64",
    "weekly_sales": "float64",
    "weekly_demand": "float64",
    "Production": "float64",
}

"""Main preprocessing script.

This script assumes that categorical encoders, PCA parameters for
image embeddings, and color KMeans cluster centers have already been
fitted and saved by `fit_encoders.py` as JSON files.
"""

# parse options

ENCODER_PATH = pathlib.Path(__file__).resolve().parent / "encoders.json"


# %% PARSING HELPERS
def parse_date_column(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    # Convert to nanoseconds since epoch as int64 without using deprecated .view
    int_vals = dt.astype("int64")
    return int_vals.to_numpy().reshape(-1, 1)

def parse_phase_in_iso_week(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    iso_cal = dt.dt.isocalendar()
    iso_week = iso_cal.week.astype("int32").to_numpy()
    return iso_week.reshape(-1, 1)

def load_encoders() -> dict:
    with ENCODER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_categorical_column(series: pd.Series, mappings: dict, column_name: str) -> np.ndarray:
    values = series.astype("string").fillna("")
    vocab = mappings.get(column_name, [])

    # Map each value to its index in the stored vocabulary
    index_map = {v: i for i, v in enumerate(vocab)}
    codes = np.array([index_map.get(str(v), -1) for v in values], dtype="int32")

    num_rows = len(values)
    num_classes = len(vocab)
    one_hot = np.zeros((num_rows, num_classes), dtype="float32")
    valid_mask = codes >= 0
    if num_classes > 0:
        one_hot[np.where(valid_mask)[0], codes[valid_mask]] = 1.0
    return one_hot

def parse_embedding_column(series: pd.Series, pca_params: dict) -> np.ndarray:
    parsed = [np.array(s.split(","), dtype="float32") for s in series]
    emb_matrix = np.vstack(parsed)

    mean = np.array(pca_params["mean"], dtype="float32")
    components = np.array(pca_params["components"], dtype="float32")[:EMBEDDING_DIM, :]


    centered = emb_matrix - mean
    reduced = centered @ components.T
    return reduced.astype("float32")

def load_color_centers_from_encoders(encoders: dict) -> np.ndarray:
    params = encoders["color_kmeans"]
    return np.array(params["cluster_centers"], dtype="float32")


def parse_color_column(series: pd.Series, centers: np.ndarray) -> np.ndarray:
    # Parse "R,G,B" strings, assign to nearest stored center, and one-hot encode
    def parse_rgb(s: str) -> np.ndarray:
        r, g, b = [float(x) for x in str(s).split(",")]
        return np.array([r, g, b], dtype="float32")

    rgb_matrix = np.vstack([parse_rgb(v) for v in series])

    # Compute distances to cluster centers
    # rgb_matrix: (n_samples, 3), centers: (NUM_COLORS, 3)
    diff = rgb_matrix[:, None, :] - centers[None, :, :]
    dists = np.sum(diff**2, axis=2)
    labels = np.argmin(dists, axis=1)

    num_rows = rgb_matrix.shape[0]
    num_colors = centers.shape[0]
    one_hot = np.zeros((num_rows, num_colors), dtype="float32")
    one_hot[np.arange(num_rows), labels] = 1.0
    return one_hot

def parse_numeric_column(series: pd.Series) -> np.ndarray:
    num_series = pd.to_numeric(series, errors="coerce")
    arr = num_series.to_numpy(dtype="float32")
    return arr.reshape(-1, 1)

def parse_year_column(series: pd.Series) -> np.ndarray:
    num_series = pd.to_numeric(series, errors="coerce")
    arr = num_series.to_numpy(dtype="int32")
    arr -= 2022
    return arr.reshape(-1, 1)


def parse_week_cosine_basis(series: pd.Series) -> np.ndarray:
    """Encode ISO week as a 2D sinusoidal basis (cos, sin).

    Weeks are treated as positions on a yearly circle with period 52.
    """

    weeks = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float32")
    # Map week index to angle on circle; use 52 as approximate period
    angles = 2.0 * np.pi * (weeks / 52.0)
    cos_comp = np.cos(angles).astype("float32")
    sin_comp = np.sin(angles).astype("float32")
    return np.stack([cos_comp, sin_comp], axis=1)

def preprocess_dataframe(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Apply full feature pipeline to a dataframe.

    For test data, the target columns are ignored by design.
    """

    df = df.copy()

    # EXTRA TIME FEATURES (only if year and num_week_iso are available)
    phase_in_dt = pd.to_datetime(df["phase_in"], format="%d/%m/%Y", errors="coerce")
    iso_cal = phase_in_dt.dt.isocalendar()
    intro_iso_year = iso_cal.year
    intro_iso_week = iso_cal.week

    # %% ADD AGGREGATED FEATURES
    # weeks_on_market
    intro_week_index = intro_iso_year * 52 + intro_iso_week
    current_week_index = df["year"] * 52 + df["num_week_iso"]
    df["weeks_on_market"] = (current_week_index - intro_week_index).astype("float32")
    df["weeks_on_market"] = np.log1p(df["weeks_on_market"].clip(lower=0)).astype("float32")
    df["squared_weeks_on_market"] = (df["weeks_on_market"] ** 2).astype("float32")
    # average price per category
    category_avg_price = df.groupby("category")["price"].transform("mean")
    df["category_avg_price"] = category_avg_price.astype("float32")
    df["price_over_category_avg"] = (df["price"] / category_avg_price).astype("float32")

    # sort chronologically: first by year, then by iso week
    df = df.sort_values(["year", "num_week_iso"]).reset_index(drop=True)
    features = []

    for parse_fn, columns in what_to_parse.items():
        for col in columns:
            # Skip columns that are not present (e.g. in test vs train)
            if col not in df.columns:
                print(f"WARNING: Column '{col}' not found in dataframe; skipping.")
                continue
            print(f"Parsing column: {col} using {parse_fn.__name__}")

            if parse_fn is parse_categorical_column:
                block = parse_fn(df[col], CAT_MAPPINGS, col)
            elif parse_fn is parse_embedding_column:
                block = parse_fn(df[col], PCA_PARAMS)
            elif parse_fn is parse_color_column:
                block = parse_fn(df[col], COLOR_CENTERS)
            else:
                block = parse_fn(df[col])

            if block.shape[1] == 1:
                features.append(pd.Series(block[:, 0], name=col))
                continue
            for j in range(block.shape[1]):
                col_name = f"{col}_{j}"
                features.append(pd.Series(block[:, j], name=col_name))

    features_df = pd.concat(features, axis=1)
    print(f"Features dataframe shape: {features_df.shape}")
    return features_df

ENCODERS = load_encoders()
CAT_MAPPINGS = ENCODERS["categorical_mappings"]
PCA_PARAMS = ENCODERS["image_embedding_pca"]
COLOR_CENTERS = load_color_centers_from_encoders(ENCODERS)


what_to_parse = {
    # ensure year and week are first
    parse_year_column: ["year"],
    parse_numeric_column: ["num_week_iso"],
    parse_week_cosine_basis: ["num_week_iso"],
    # parse_date_column: ["phase_in", "phase_out"],
    parse_categorical_column: [
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
    ],
    parse_embedding_column: ["image_embedding"],
    # parse_embedding_cluster_column: ["image_embedding"],
    parse_numeric_column: [
        "price",        
        # "weekly_sales",
        "weekly_demand",
        # "Production",
        "num_stores",
        "num_sizes",
        "has_plus_sizes",
        "weeks_on_market",
        "ID",
    ],
    parse_color_column: ["color_rgb"],
    parse_phase_in_iso_week: ["phase_in"],
}

# %% ENTRY POINT: PREPROCESS TRAIN AND TEST
# Train
train_df = pd.read_csv(TRAIN_INPUT_PATH, dtype=DATA_TYPES, delimiter=";")  # type: ignore
train_features = preprocess_dataframe(train_df, is_test=False)

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_features.to_parquet(TRAIN_OUTPUT_PATH, index=False)
print("Train total size:", train_features.to_numpy().shape)

# Test (no output/target column present or needed)
test_df = pd.read_csv(TEST_INPUT_PATH, dtype=DATA_TYPES, delimiter=";")  # type: ignore
test_features = preprocess_dataframe(test_df, is_test=True)
test_features.to_parquet(TEST_OUTPUT_PATH, index=False)
print("Test total size:", test_features.to_numpy().shape)
