import pathlib

import pandas as pd

from preprocess import (
    DATA_TYPES,
    parse_date_column,
    parse_phase_in_iso_week,
    parse_categorical_column,
    parse_embedding_column,
    parse_color_column,
    parse_numeric_column,
    parse_year_column,
    parse_week_cosine_basis,
    CAT_MAPPINGS,
    PCA_PARAMS,
    COLOR_CENTERS,
)


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "test.csv"
OUTPUT_PATH = BASE_DIR / "clean_data" / "test.parquet"


df = pd.read_csv(INPUT_PATH, dtype=DATA_TYPES, delimiter=";")


phase_in_dt = pd.to_datetime(df["phase_in"], format="%d/%m/%Y", errors="coerce")
iso_cal = phase_in_dt.dt.isocalendar()
intro_iso_year = iso_cal.year
intro_iso_week = iso_cal.week

intro_week_index = intro_iso_year * 52 + intro_iso_week
current_week_index = df["year"] * 52 + df["num_week_iso"]


df["weeks_on_market"] = (current_week_index - intro_week_index).astype("float32")

df = df.sort_values(["year", "num_week_iso"]).reset_index(drop=True)


what_to_parse = {
    parse_year_column: ["year"],
    parse_numeric_column: ["num_week_iso"],
    parse_week_cosine_basis: ["num_week_iso"],
    parse_date_column: ["phase_in", "phase_out"],
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
    parse_numeric_column: [
        "price",
        "Production",
        "num_stores",
        "num_sizes",
        "has_plus_sizes",
        "weeks_on_market",
    ],
    parse_color_column: ["color_rgb"],
    parse_phase_in_iso_week: ["phase_in"],
}


features = []

for parse_fn, columns in what_to_parse.items():
    for col in columns:
        print(f"Parsing column (test): {col} using {parse_fn.__name__}")

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
print(f"Test features dataframe shape: {features_df.shape}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features_df.to_parquet(OUTPUT_PATH, index=False)
print("Saved test features to", OUTPUT_PATH)
