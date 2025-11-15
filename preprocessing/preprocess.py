# %% IMPORTS 
import pandas as pd
import numpy as np
import pathlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %% CONSTANTS AND PATHS

INPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
INPUT_PATH = INPUT_DIR / "train.csv"

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "clean_data"
OUTPUT_PATH = OUTPUT_DIR / "train.parquet"

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

# parse options 
EMBEDDING_DIM = 128  # target dimension after PCA
NUM_COLORS = 20

# %% READ CSV
df = pd.read_csv(INPUT_PATH, dtype=DATA_TYPES, delimiter=";")

# %% EXTRA TIME FEATURES
phase_in_dt = pd.to_datetime(df["phase_in"], format="%d/%m/%Y", errors="coerce")
iso_cal = phase_in_dt.dt.isocalendar()
intro_iso_year = iso_cal.year
intro_iso_week = iso_cal.week

intro_week_index = intro_iso_year * 52 + intro_iso_week
current_week_index = df["year"] * 52 + df["num_week_iso"]

df["weeks_on_market"] = (current_week_index - intro_week_index).astype("float32")

# sort chronologically: first by year, then by iso week
df = df.sort_values(["year", "num_week_iso"]).reset_index(drop=True)

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

def parse_categorical_column(series: pd.Series) -> np.ndarray:
    # Convert to string, factorize, then one‑hot encode
    values = series.astype("string")
    codes, uniques = pd.factorize(values, sort=True)
    codes = np.where(codes < 0, -1, codes)

    num_rows = len(values)
    num_classes = len(uniques)
    one_hot = np.zeros((num_rows, num_classes), dtype="float32")
    valid_mask = codes >= 0
    if num_classes > 0:
        one_hot[np.where(valid_mask)[0], codes[valid_mask]] = 1.0
    return one_hot

def parse_embedding_column(series: pd.Series) -> np.ndarray:
    # Simplified: assume parsing always works and original dim > EMBEDDING_DIM
    # 1) Parse each embedding string into a float32 vector
    parsed = [np.array(s.split(","), dtype="float32") for s in series]

    emb_matrix = np.vstack(parsed)
    pca = PCA(n_components=EMBEDDING_DIM)
    reduced = pca.fit_transform(emb_matrix).astype("float32")

    return reduced


def parse_color_column(series: pd.Series) -> np.ndarray:
    # Parse "R,G,B" strings, cluster them, and one-hot encode cluster labels
    def parse_rgb(s: str) -> np.ndarray:
        r, g, b = [float(x) for x in str(s).split(",")]
        return np.array([r, g, b], dtype="float32")

    rgb_matrix = np.vstack([parse_rgb(v) for v in series])

    kmeans = KMeans(n_clusters=NUM_COLORS, n_init=10, random_state=42)
    labels = kmeans.fit_predict(rgb_matrix)

    num_rows = rgb_matrix.shape[0]
    one_hot = np.zeros((num_rows, NUM_COLORS), dtype="float32")
    one_hot[np.arange(num_rows), labels] = 1.0
    return one_hot

def parse_numeric_column(series: pd.Series) -> np.ndarray:
    num_series = pd.to_numeric(series, errors="coerce")
    arr = num_series.to_numpy(dtype="float32")
    return arr.reshape(-1, 1)

def parse_year_column(series: pd.Series) -> np.ndarray:
    num_series = pd.to_numeric(series, errors="coerce")
    arr = num_series.to_numpy(dtype="int32")
    arr -= 2023
    return arr.reshape(-1, 1)

what_to_parse = {
    # ensure year and week are first
    parse_year_column: ["year"],
    parse_numeric_column: ["num_week_iso"],
    # parse_date_column: ["phase_in", "phase_out"],
    parse_categorical_column: [
        "aggregated_family",
        "family",
        "category",
        "fabric",
        # "color_name",
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
    parse_numeric_column: [
        "price",        
        # "weekly_sales",
        "weekly_demand",
        "Production",
        "num_stores",
        "num_sizes",
        "has_plus_sizes",
        "weeks_on_market",
    ],
    parse_color_column: ["color_rgb"],
    parse_phase_in_iso_week: ["phase_in"],
}

# %% BUILD FEATURE DATAFRAME
features_df = pd.DataFrame(index=df.index)

for parse_fn, columns in what_to_parse.items():
    for col in columns:
        print(f"Parsing column: {col} using {parse_fn.__name__}")
        block = parse_fn(df[col])

        if block.shape[1] == 1:
            features_df[col] = block
            continue
        for j in range(block.shape[1]):
            col_name = f"{col}_{j}"
            features_df[col_name] = block[:, j]


print(f"Features dataframe shape: {features_df.shape}")
final_df = features_df.copy()

# %% SAVE OUTPUTS
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_df.to_parquet(OUTPUT_PATH, index=False)
print("total size:", final_df.to_numpy().shape)
final_df.head()