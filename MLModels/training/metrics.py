from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def validate_classification_score_values(
    y_score: Any,
    *,
    context: str,
) -> np.ndarray:
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if not np.isfinite(y_score_arr).all():
        raise ValueError(f"{context}: non-finite values in classification scores")
    return y_score_arr


def validate_regression_metric_inputs(
    y_true: Any,
    y_pred: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"{context}: shape mismatch y_true={y_true_arr.shape} y_pred={y_pred_arr.shape}"
        )
    if not np.isfinite(y_true_arr).all():
        raise ValueError(f"{context}: non-finite values in regression targets")
    if not np.isfinite(y_pred_arr).all():
        raise ValueError(f"{context}: non-finite values in regression predictions")
    return y_true_arr, y_pred_arr


def validate_classification_metric_inputs(
    y_true: Any,
    y_score: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = validate_classification_score_values(y_score, context=context)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError(
            f"{context}: shape mismatch y_true={y_true_arr.shape} y_score={y_score_arr.shape}"
        )
    if not np.isfinite(y_true_arr.astype(float)).all():
        raise ValueError(f"{context}: non-finite values in classification targets")
    return y_true_arr.astype(int), y_score_arr


def safe_auc(y_true: Any, y_score: Any) -> float | None:
    try:
        y_true_arr, y_score_arr = validate_classification_metric_inputs(
            y_true,
            y_score,
            context="classification metric",
        )
    except ValueError:
        return None
    if len(np.unique(y_true_arr)) < 2:
        logging.warning("AUROC undefined for single-class test set.")
        return None
    return float(roc_auc_score(y_true_arr, y_score_arr))


def safe_auprc(y_true: Any, y_score: Any) -> float | None:
    try:
        y_true_arr, y_score_arr = validate_classification_metric_inputs(
            y_true,
            y_score,
            context="classification metric",
        )
    except ValueError:
        return None
    if len(np.unique(y_true_arr)) < 2:
        logging.warning("AUPRC undefined for single-class test set.")
        return None
    return float(average_precision_score(y_true_arr, y_score_arr))


def classification_metrics_from_outputs(
    y_true: Any,
    y_score: Any,
    y_pred: Any,
    *,
    context: str,
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]:
    y_true_arr, y_score_arr = validate_classification_metric_inputs(
        y_true,
        y_score,
        context=context,
    )
    y_pred_arr = ensure_binary_labels(pd.Series(np.asarray(y_pred).reshape(-1))).to_numpy(dtype=int)
    if y_pred_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            f"{context}: shape mismatch y_true={y_true_arr.shape} y_pred={y_pred_arr.shape}"
        )
    metrics = {
        "auc": safe_auc(y_true_arr, y_score_arr),
        "auprc": safe_auprc(y_true_arr, y_score_arr),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr)),
    }
    return y_true_arr, y_score_arr, y_pred_arr, metrics


def safe_r2(y_true: Any, y_pred: Any) -> float | None:
    try:
        y_true_arr, y_pred_arr = validate_regression_metric_inputs(
            y_true,
            y_pred,
            context="regression metric",
        )
        return float(r2_score(y_true_arr, y_pred_arr))
    except ValueError:
        return None


def safe_mae(y_true: Any, y_pred: Any) -> float | None:
    try:
        y_true_arr, y_pred_arr = validate_regression_metric_inputs(
            y_true,
            y_pred,
            context="regression metric",
        )
        return float(mean_absolute_error(y_true_arr, y_pred_arr))
    except ValueError:
        return None
