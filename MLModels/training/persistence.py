from __future__ import annotations

import json
import os
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_params(params: dict[str, Any], params_path: str) -> None:
    joblib.dump(params, params_path)


def save_metrics_series(metrics: dict[str, Any], metrics_path: str) -> None:
    pd.Series(metrics).to_json(metrics_path)


def save_metrics_json(metrics: dict[str, Any], metrics_path: str, *, indent: int = 2) -> None:
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=indent)


def save_model_pickle(estimator: object, model_path: str) -> None:
    joblib.dump(estimator, model_path)


def load_pickle(path: str) -> object:
    return joblib.load(path)


def save_torch_state_dict(estimator: object, model_path: str) -> None:
    import torch

    torch.save(estimator.state_dict(), model_path)
