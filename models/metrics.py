"""Utility functions to evaluate regression models.

Provides a simple report function that prints several metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error


def print_regression_report(y_true, y_pred, y_train_true, y_train_pred) -> None:
    """Compute and print common regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    """

    # Ensure NumPy arrays
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_train_true_arr = np.asarray(y_train_true, dtype=float)
    y_train_pred_arr = np.asarray(y_train_pred, dtype=float)

    mean_err = np.mean(np.abs(y_true_arr - y_pred_arr))
    mean_train_err = np.mean(np.abs(y_train_true_arr - y_train_pred_arr))
    y_mean = np.mean(y_true_arr)
    y_train_mean = np.mean(y_train_true_arr)
    
    print(f"REGRESSION REPORT")
    print(f"-----------------")
    print(f"mean absolute error / mean_y: {mean_err / y_mean:.6f}")
    print(f"train mean absolute error / mean_y_train: {mean_train_err / y_train_mean:.6f}")
    
