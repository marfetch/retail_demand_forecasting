from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numpy(values) -> np.ndarray:
    """Convert pandas/numpy-like values to a 1D float array."""
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _clean_aligned_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    return y_true_arr[valid_mask], y_pred_arr[valid_mask]


def mae(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _clean_aligned_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _clean_aligned_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _clean_aligned_arrays(y_true, y_pred)
    non_zero_mask = y_true_arr != 0

    if not np.any(non_zero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask])
        / y_true_arr[non_zero_mask]
    )
    return float(np.mean(percentage_errors) * 100)


def wape(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _clean_aligned_arrays(y_true, y_pred)
    denominator = np.sum(np.abs(y_true_arr))

    if denominator == 0:
        return float("nan")

    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / denominator * 100)


def evaluate_forecast(y_true, y_pred, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [model_name],
            "MAE": [mae(y_true, y_pred)],
            "RMSE": [rmse(y_true, y_pred)],
            "MAPE": [mape(y_true, y_pred)],
            "WAPE": [wape(y_true, y_pred)],
        }
    )
