from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

_XGBOOST_INVALID_FEATURE_CHARS = re.compile(r"[\[\]<>]")


def sanitize_xgboost_feature_columns(columns: pd.Index) -> tuple[list[str], list[dict[str, Any]], int]:
    sanitized: list[str] = []
    records: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    changed_count = 0

    for idx, col in enumerate(columns):
        original = str(col)
        base = _XGBOOST_INVALID_FEATURE_CHARS.sub("_", original) or "feature"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used[candidate] = 1
        if candidate != original:
            changed_count += 1
        sanitized.append(candidate)
        records.append(
            {
                "index": idx,
                "original": original,
                "sanitized": candidate,
                "changed": candidate != original,
            }
        )
    return sanitized, records, changed_count


def assign_feature_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame | None:
    if df is None:
        return None
    if df.shape[1] != len(columns):
        raise ValueError(
            "Feature column mismatch while applying sanitized feature names: "
            f"expected {len(columns)} columns, got {df.shape[1]}."
        )
    out = df.copy()
    out.columns = columns
    return out


def sanitize_xgboost_feature_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    sanitized_columns, records, changed_count = sanitize_xgboost_feature_columns(X_train.columns)
    payload = {
        "sanitized_columns": sanitized_columns,
        "columns": records,
        "changed_count": changed_count,
        "invalid_char_pattern": r"[\[\]<>]",
    }
    return (
        assign_feature_columns(X_train, sanitized_columns),
        assign_feature_columns(X_test, sanitized_columns),
        assign_feature_columns(X_val, sanitized_columns),
        payload,
    )


def maybe_sanitize_xgboost_feature_frames(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str | None]:
    if model_type not in {"xgboost", "ensemble"}:
        return X_train, X_test, X_val, None

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        return X_train, X_test, X_val, None

    X_train_s, X_test_s, X_val_s, payload = sanitize_xgboost_feature_frames(X_train, X_test, X_val)
    map_path = os.path.join(output_dir, f"{model_type}_feature_name_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    if payload["changed_count"] > 0:
        logging.info(
            "Sanitized %d feature names for %s to satisfy XGBoost constraints. map=%s",
            payload["changed_count"],
            model_type,
            map_path,
        )
    return X_train_s, X_test_s, X_val_s, map_path


def maybe_apply_xgboost_feature_map_for_explain(
    model_type: str,
    output_dir: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_type not in {"xgboost", "ensemble"}:
        return X_train, X_test

    map_path = os.path.join(output_dir, f"{model_type}_feature_name_map.json")
    columns: list[str] | None = None
    if os.path.exists(map_path):
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            mapped = payload.get("sanitized_columns")
            if isinstance(mapped, list) and mapped:
                columns = [str(col) for col in mapped]
        except Exception as exc:
            logging.warning("Failed to load feature-name map at %s: %s", map_path, exc)

    if columns is None:
        columns, _, _ = sanitize_xgboost_feature_columns(X_train.columns)

    X_train_s = assign_feature_columns(X_train, columns)
    X_test_s = assign_feature_columns(X_test, columns)
    assert X_train_s is not None and X_test_s is not None
    return X_train_s, X_test_s


def sigmoid(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return 1.0 / (1.0 + np.exp(-arr))


def ensure_binary_labels(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("Expected a single label column for classification.")
        series = series.iloc[:, 0]
    values = series.dropna().unique()
    if len(values) == 0:
        raise ValueError("Empty label series for classification.")
    if len(values) > 2:
        raise ValueError(f"Expected binary labels; got {len(values)} classes.")

    if set(values).issubset({0, 1}):
        return series.astype(int)

    def _coerce_numeric(val):
        if isinstance(val, (int, float, np.generic)) and val in {0, 1}:
            return int(val)
        if isinstance(val, str):
            token = val.strip().lower()
            if token in {"0", "1"}:
                return int(token)
            try:
                parsed = float(token)
            except ValueError:
                return None
            if parsed in {0.0, 1.0}:
                return int(parsed)
        return None

    def _normalize(v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    normalized = {_normalize(v) for v in values}
    if normalized == {"active", "inactive"}:
        mapping = {"active": 1, "inactive": 0}
    elif normalized == {"inactive", "active"}:
        mapping = {"active": 1, "inactive": 0}
    else:
        coerced = series.map(_coerce_numeric)
        if coerced.notna().all():
            return coerced.astype(int)
        raise ValueError("Classification labels must be 0/1 or active/inactive; add label.normalize.")

    return series.map(lambda v: mapping.get(_normalize(v)))


def predict_classification_outputs(
    estimator: object,
    model_type: str,
    X: pd.DataFrame,
    *,
    predict_dl: Callable[[object, np.ndarray], np.ndarray],
    validate_classification_score_values: Callable[..., np.ndarray],
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
    sigmoid_fn: Callable[[np.ndarray | pd.Series | list[float]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_type.startswith("dl_"):
        logits = validate_classification_score_values(
            predict_dl(estimator, X.values),
            context=f"{model_type} classification raw scores",
        )
        y_proba = sigmoid_fn(logits)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_proba, y_pred, logits

    if hasattr(estimator, "predict_proba"):
        proba_raw = np.asarray(estimator.predict_proba(X))
        if proba_raw.ndim == 2:
            if proba_raw.shape[1] < 2:
                classes = np.asarray(getattr(estimator, "classes_", []))
                if classes.size == 1 and int(classes[0]) == 1:
                    y_proba = np.ones(proba_raw.shape[0], dtype=float)
                else:
                    y_proba = np.zeros(proba_raw.shape[0], dtype=float)
            else:
                y_proba = proba_raw[:, 1].reshape(-1).astype(float)
        else:
            y_proba = proba_raw.reshape(-1).astype(float)
        y_proba = validate_classification_score_values(
            y_proba,
            context=f"{model_type} classification probabilities",
        )
        y_pred = ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
        return y_proba, y_pred, y_proba

    if hasattr(estimator, "decision_function"):
        scores_raw = np.asarray(estimator.decision_function(X))
        if scores_raw.ndim == 2 and scores_raw.shape[1] >= 2:
            scores = scores_raw[:, 1].reshape(-1).astype(float)
        else:
            scores = scores_raw.reshape(-1).astype(float)
        scores = validate_classification_score_values(
            scores,
            context=f"{model_type} classification raw scores",
        )
        y_proba = sigmoid_fn(scores)
        y_pred = ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
        return y_proba, y_pred, scores

    y_pred = ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
    y_proba = y_pred.astype(float)
    return y_proba, y_pred, y_pred.astype(float)


def require_chemprop() -> None:
    try:
        import chemprop  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "chemprop is required for model.type=chemprop. "
            "Install it (and torch/lightning) e.g. `pip install chemprop`."
        ) from exc


def resolve_chemprop_split_positions(
    curated_df: pd.DataFrame,
    split_indices: dict[str, Any],
    row_index_col: str = "__row_index",
    allow_legacy_positions: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    def _to_int_list(split_name: str, values: Any) -> list[int]:
        out: list[int] = []
        for value in values or []:
            try:
                out.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"split.{split_name} contains a non-integer index value: {value!r}."
                ) from exc
        return out

    train_raw = _to_int_list("train", split_indices.get("train", []))
    val_raw = _to_int_list("val", split_indices.get("val", []))
    test_raw = _to_int_list("test", split_indices.get("test", []))
    if not test_raw and val_raw:
        test_raw, val_raw = val_raw, []

    n_rows = int(len(curated_df))
    if row_index_col in curated_df.columns:
        row_id_series = pd.to_numeric(curated_df[row_index_col], errors="coerce")
        if row_id_series.isna().any():
            raise ValueError(
                f"Curated data contains non-numeric {row_index_col!r} values; cannot map split row IDs."
            )
        row_ids = [int(v) for v in row_id_series.tolist()]
    else:
        row_ids = list(range(n_rows))

    if len(set(row_ids)) != len(row_ids):
        raise ValueError(
            f"Curated data contains duplicate {row_index_col!r} values; split row-ID mapping is ambiguous."
        )

    id_to_pos = {rid: pos for pos, rid in enumerate(row_ids)}

    def _map_row_ids(raw_ids: list[int]) -> tuple[list[int], list[int]]:
        mapped: list[int] = []
        missing: list[int] = []
        for rid in raw_ids:
            pos = id_to_pos.get(rid)
            if pos is None:
                missing.append(rid)
            else:
                mapped.append(int(pos))
        return mapped, missing

    train_pos, missing_train = _map_row_ids(train_raw)
    val_pos, missing_val = _map_row_ids(val_raw)
    test_pos, missing_test = _map_row_ids(test_raw)

    missing_total = len(missing_train) + len(missing_val) + len(missing_test)
    if missing_total > 0:
        all_raw = train_raw + val_raw + test_raw
        looks_like_legacy_positions = all(0 <= idx < n_rows for idx in all_raw)
        if looks_like_legacy_positions and allow_legacy_positions:
            logging.warning(
                "Chemprop split indices did not match %s values; treating them as legacy positional indices.",
                row_index_col,
            )
            train_pos, val_pos, test_pos = train_raw, val_raw, test_raw
        else:
            legacy_hint = ""
            if looks_like_legacy_positions:
                legacy_hint = (
                    " Indices look like legacy positional indices; regenerate splits with row IDs "
                    "or set train.model.allow_legacy_split_positions=true to opt in."
                )
            raise ValueError(
                "Chemprop split indices do not match curated row IDs. "
                f"row_index_column={row_index_col!r} "
                f"missing(train/val/test)=({missing_train[:10]}, {missing_val[:10]}, {missing_test[:10]})."
                f"{legacy_hint}"
            )

    return train_pos, val_pos, test_pos, row_ids


def resolve_chemprop_predictor_ctor(nn_module, task_type: str):
    if task_type == "classification":
        predictor_ctor = getattr(nn_module, "BinaryClassificationFFN", None)
        predictor_name = "BinaryClassificationFFN"
    else:
        predictor_ctor = getattr(nn_module, "RegressionFFN", None)
        predictor_name = "RegressionFFN"
    if predictor_ctor is None:
        raise ValueError(
            f"chemprop {task_type} requires nn.{predictor_name}, "
            "but it is unavailable in the installed chemprop version."
        )
    return predictor_ctor
