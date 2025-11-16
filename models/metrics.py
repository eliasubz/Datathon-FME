"""Utility functions to evaluate regression models.

Provides a simple report function that prints several metrics.
"""

import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error


def regression_report(y_true, y_pred, y_train_true, y_train_pred, model_name) -> None:
    """Compute, print, and save common regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    y_train_true : array-like
        True train target values.
    y_train_pred : array-like
        Predicted train target values.
    model_name : str, optional
        Name of the model (used for saving report).
    """

    import pathlib
    # Ensure NumPy arrays
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_train_true_arr = np.asarray(y_train_true, dtype=float)
    y_train_pred_arr = np.asarray(y_train_pred, dtype=float)

    mean_err = np.mean(np.abs(y_true_arr - y_pred_arr))
    mean_train_err = np.mean(np.abs(y_train_true_arr - y_train_pred_arr))
    y_mean = np.mean(y_true_arr)
    y_train_mean = np.mean(y_train_true_arr)

    # Prepare report as text
    report = []
    report.append("REGRESSION REPORT")
    report.append("-----------------")
    report.append(f"mean absolute error / mean_y: {mean_err / y_mean:.6f}")
    report.append(f"train mean absolute error / mean_y_train: {mean_train_err / y_train_mean:.6f}")

    # Prepare report as dict for YAML
    report_dict = {
        "mean_absolute_error_over_mean_y": float(mean_err / y_mean),
        "train_mean_absolute_error_over_mean_y_train": float(mean_train_err / y_train_mean),
    }

    # Print to console
    for line in report:
        print(line)

    # Save to file if model_name is provided
    if model_name:
        reports_dir = pathlib.Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save as YAML
        yaml_path = reports_dir / f"{model_name}_report.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(report_dict, f)
        print(f"Saved regression report as YAML to {yaml_path}")

