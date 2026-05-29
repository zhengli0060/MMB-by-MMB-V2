import numpy as np
import pandas as pd
from typing import Union


def sanitize_data(data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy matrix
        if np.isnan(data.values).any() or np.isinf(data.values).any():
            raise ValueError("Input DataFrame contains NaN values. Please handle missing data before using this function.")
        assert len(set(data.columns.to_list())) == len(data.columns), "DataFrame columns must be unique."
        return data
    if isinstance(data, np.ndarray):
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Input numpy array contains NaN or infinite values. Please handle missing data before using this function.")
        # Check if the numpy array is 2-dimensional (matrix)
        if data.ndim == 2:
            return pd.DataFrame(data, columns=[i for i in range(data.shape[1])])
        raise ValueError("Input numpy array is not a matrix (2-dimensional).")
    raise TypeError("Input must be a pandas DataFrame or a numpy array.")


def local_mark_evaluation(pred_pag: pd.DataFrame, true_pag: pd.DataFrame, target: str) -> dict:
    """
    Evaluate target row and target column for DataFrame matrices.
    """

    assert isinstance(pred_pag, pd.DataFrame)
    assert isinstance(true_pag, pd.DataFrame)
    assert pred_pag.shape == true_pag.shape
    assert list(pred_pag.index) == list(true_pag.index)
    assert list(pred_pag.columns) == list(true_pag.columns)

    # Target row excluding [target, target]
    row_true = true_pag.loc[target, true_pag.columns != target].to_numpy()
    row_pred = pred_pag.loc[target, pred_pag.columns != target].to_numpy()

    # Target column excluding [target, target]
    col_true = true_pag.loc[true_pag.index != target, target].to_numpy()
    col_pred = pred_pag.loc[pred_pag.index != target, target].to_numpy()

    y_true = np.concatenate([row_true, col_true])
    y_pred = np.concatenate([row_pred, col_pred])

    local_shd = np.sum(y_true != y_pred)


    tp = np.sum((y_pred != 0) & (y_pred == y_true))
    mark_precision = tp / np.sum(y_pred != 0) if np.sum(y_pred != 0) > 0 else 0
    mark_recall = tp / np.sum(y_true != 0) if np.sum(y_true != 0) > 0 else 0
    mark_f1 = 2 * (mark_precision * mark_recall) / (mark_precision + mark_recall) if (mark_precision + mark_recall) > 0 else 0

    results = {
        "Local-SHD": int(local_shd),
        "Mark-Precision": mark_precision,
        "Mark-Recall": mark_recall,
        "Mark-F1": mark_f1
    }

    return results

