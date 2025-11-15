import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder


# Load your datasets
train = pd.read_csv("./data/train.csv", sep=';')
test = pd.read_csv("./data/test.csv", sep=';')

# Extract embeddings FIRST
print("Extracting embeddings...")
embeddings_train = np.array([list(map(float, s.split(','))) for s in train['image_embedding']])
embeddings_test = np.array([list(map(float, s.split(','))) for s in test['image_embedding']])
print(f"Original embeddings shape: {embeddings_train.shape}")

# Apply PCA to embeddings
print("\nApplying PCA to embeddings...")
n_components = 128

pca = PCA(n_components=n_components, random_state=42)
embeddings_train_pca = pca.fit_transform(embeddings_train)
embeddings_test_pca = pca.transform(embeddings_test)

print(f"PCA embeddings shape: {embeddings_train_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Drop image_embedding column
train = train.drop(columns=['image_embedding'])
test = test.drop(columns=['image_embedding'])

# Process color_rgb for train
print("\nProcessing color_rgb...")
rgb_split_train = train['color_rgb'].str.split(',', expand=True).astype(float)
rgb_split_train.columns = ['R', 'G', 'B']
train = train.drop(columns=['color_rgb'])
train[['R','G','B']] = rgb_split_train

# Process color_rgb for test
rgb_split_test = test['color_rgb'].str.split(',', expand=True).astype(float)
rgb_split_test.columns = ['R', 'G', 'B']
test = test.drop(columns=['color_rgb'])
test[['R','G','B']] = rgb_split_test

# SEPARATE OUT columns that don't exist in test (but keep them for later)
columns_not_in_test = ['weekly_demand', 'Production', 'year', 'num_week_iso']
extra_train_features = train[[c for c in columns_not_in_test if c in train.columns]].copy()
print(f"\nSaved extra training features: {extra_train_features.columns.tolist()}")
print(f"Extra features shape: {extra_train_features.shape}")


def build_full_preprocessing_pipeline(df):
    one_hot_columns = [
        "toecap_type",
        "heel_shape_type",
        "has_plus_sizes",
        "id_season",
        "knit_structure",
        "woven_structure",
        "waist_type",
        "category",
        "moment",
        "fabric",
        "sleeve_length_type",
        "archetype",
        "aggregated_family",
        "length_type",
        "print_type",
        "life_cycle_length",
        "family",
        "neck_lapel_type",
        "silhouette_type"
    ]

    target_encode_columns = [
        "color_name",
    ]

    # Filter only columns that exist in the DataFrame
    one_hot_columns = [c for c in one_hot_columns if c in df.columns]
    target_encode_columns = [c for c in target_encode_columns if c in df.columns]

    print(f"One-hot encoding {len(one_hot_columns)} columns")
    print(f"Target encoding {len(target_encode_columns)} columns")

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_columns),
            ("target", TargetEncoder(cols=target_encode_columns), target_encode_columns),
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    return pipeline


# Prepare data - DROP columns not in test (temporarily)
print("\nPreparing data for pipeline...")
X_train_for_pipeline = train.drop(columns=['weekly_sales'] + [c for c in columns_not_in_test if c in train.columns])
y_train = train['weekly_sales']

print(f"X_train for pipeline shape: {X_train_for_pipeline.shape}")
print(f"X_test shape: {test.shape}")

# Build pipeline
print("\nBuilding pipeline...")
pipeline = build_full_preprocessing_pipeline(X_train_for_pipeline)

# Fit and transform training data
print("Fitting pipeline on training data...")
pipeline.fit(X_train_for_pipeline, y_train)

print("Transforming training data...")
X_train_processed = pipeline.transform(X_train_for_pipeline)

# Transform test data
print("Transforming test data...")
X_test_processed = pipeline.transform(test)

# Combine: processed features + PCA embeddings + extra train features
print("\nCombining all features...")
X_train_final = np.hstack([
    X_train_processed, 
    embeddings_train_pca,
    extra_train_features.values  # ← ADD BACK the extra features
])

X_test_final = np.hstack([
    X_test_processed, 
    embeddings_test_pca
])

print("\n" + "="*70)
print("FINAL RESULTS - TRAINING DATA:")
print("="*70)
print(f"  Pipeline processed features:  {X_train_processed.shape}")
print(f"  PCA embeddings:               {embeddings_train_pca.shape}")
print(f"  Extra features (added back):  {extra_train_features.shape}")
print(f"  ---")
print(f"  FINAL COMBINED SHAPE:         {X_train_final.shape}")
print(f"  Total features: {X_train_processed.shape[1]} + {embeddings_train_pca.shape[1]} + {extra_train_features.shape[1]} = {X_train_final.shape[1]}")

print("\n" + "="*70)
print("FINAL RESULTS - TEST DATA:")
print("="*70)
print(f"  Pipeline processed features:  {X_test_processed.shape}")
print(f"  PCA embeddings:               {embeddings_test_pca.shape}")
print(f"  ---")
print(f"  FINAL COMBINED SHAPE:         {X_test_final.shape}")
print(f"  Total features: {X_test_processed.shape[1]} + {embeddings_test_pca.shape[1]} = {X_test_final.shape[1]}")
print("="*70)

print(f"\n Extra training features included: {extra_train_features.columns.tolist()}")
print(f" Training has {X_train_final.shape[1] - X_test_final.shape[1]} more features than test")
