import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load preprocessed data
X_train_final = np.load('X_train_final.npy')
X_test_final = np.load('X_test_final.npy')
y_train = np.load('y_train.npy')

print(f"X_train shape: {X_train_final.shape}")
print(f"X_test shape: {X_test_final.shape}")

# Split data
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42
)

# Build model
model = keras.Sequential([
    layers.Input(shape=(X_train_final.shape[1],)),
    
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Train
print("Training model...")
history = model.fit(
    X_train_nn, y_train_nn,
    validation_data=(X_val_nn, y_val_nn),
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
y_val_pred = model.predict(X_val_nn).flatten()

print(f"\nValidation MSE: {mean_squared_error(y_val_nn, y_val_pred):.4f}")
print(f"Validation MAE: {mean_absolute_error(y_val_nn, y_val_pred):.4f}")
print(f"Validation R²: {r2_score(y_val_nn, y_val_pred):.4f}")

# Predict on test
y_test_pred = model.predict(X_test_final).flatten()

# Save predictions
import pandas as pd
pd.DataFrame({'predicted_weekly_demand': y_test_pred}).to_csv('test_predictions.csv', index=False)
model.save('demand_model.keras')

print("Done!")