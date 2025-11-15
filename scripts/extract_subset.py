import pandas as pd
import numpy as np

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load your datasets
train = pd.read_csv("data/train.csv", sep=';')
test = pd.read_csv("data/test.csv", sep=';')


print("Datasets head:")
print(train.head())
print(test.head())

# Choose how many rows you want to sample
subset_size = 2000   # <-- change this as you like

# Create subsets (random but reproducible)
train_subset = train.sample(n=subset_size, random_state=SEED)
test_subset = test.sample(n=subset_size, random_state=SEED)

# Save them
train_subset.to_csv("data/train_subset.csv", index=False, sep=';')
test_subset.to_csv("data/test_subset.csv", index=False, sep=';')

print("Subset files saved: train_subset.csv, test_subset.csv")


print(train.nunique().sort_values())