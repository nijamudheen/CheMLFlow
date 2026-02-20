from __future__ import annotations

from pathlib import Path

import pandas as pd

import main
from MLModels import train_models


def _make_context(tmp_path: Path, *, global_seed: int, train_seed: int | None) -> dict:
    base_dir = tmp_path / "data_seed_test"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run_seed_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = main.build_paths(str(base_dir))

    train_cfg = {
        "model": {"type": "random_forest"},
        "tuning": {},
        "early_stopping": {},
    }
    if train_seed is not None:
        train_cfg["random_state"] = int(train_seed)

    return {
        "pipeline_type": "flash",
        "run_dir": str(run_dir),
        "model_type": "random_forest",
        "target_column": "y",
        "paths": paths,
        "task_type": "regression",
        "train_config": train_cfg,
        "model_config": {"type": "random_forest"},
        "global_random_state": int(global_seed),
        "split_indices": {"train": [0, 1, 2], "val": [3], "test": [4]},
        "split_config": {},
        "categorical_features": [],
        "debug_logging": False,
    }


def _patch_data_loading(monkeypatch):
    X = pd.DataFrame(
        {
            "f0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[0, 1, 2, 3, 4],
    )
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=[0, 1, 2, 3, 4], name="y")

    monkeypatch.setattr(main, "validate_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main.data_preprocessing,
        "load_features_labels",
        lambda *args, **kwargs: (X.copy(), y.copy()),
    )
    monkeypatch.setattr(main.data_preprocessing, "verify_data_quality", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.data_preprocessing, "check_data_leakage", lambda *args, **kwargs: None)


def test_train_random_state_overrides_global(monkeypatch, tmp_path: Path) -> None:
    _patch_data_loading(monkeypatch)
    captured: dict[str, int] = {}

    def _fake_train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        output_dir,
        random_state=42,
        **kwargs,
    ):
        _ = (X_train, y_train, X_test, y_test, model_type, output_dir, kwargs)
        captured["seed"] = int(random_state)
        return object(), train_models.TrainResult("model.pkl", "params.pkl", "metrics.json")

    monkeypatch.setattr(main.train_models, "train_model", _fake_train_model)

    ctx = _make_context(tmp_path, global_seed=999, train_seed=123)
    main.run_node_train(ctx)

    assert captured["seed"] == 123


def test_train_random_state_falls_back_to_global(monkeypatch, tmp_path: Path) -> None:
    _patch_data_loading(monkeypatch)
    captured: dict[str, int] = {}

    def _fake_train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        output_dir,
        random_state=42,
        **kwargs,
    ):
        _ = (X_train, y_train, X_test, y_test, model_type, output_dir, kwargs)
        captured["seed"] = int(random_state)
        return object(), train_models.TrainResult("model.pkl", "params.pkl", "metrics.json")

    monkeypatch.setattr(main.train_models, "train_model", _fake_train_model)

    ctx = _make_context(tmp_path, global_seed=999, train_seed=None)
    main.run_node_train(ctx)

    assert captured["seed"] == 999
