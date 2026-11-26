import sys
import types

import numpy as np
import pandas as pd
import pytest

from MLModels import train_models


def test_train_model_dl_fixed_respects_model_params(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    saved_tensors: dict[str, object] = {}

    class _FakeDlModel:
        def __init__(self, params: dict[str, object]) -> None:
            self.params = dict(params)
            self.loaded_state = None

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state_dict: dict[str, object]) -> None:
            self.loaded_state = dict(state_dict)

        def eval(self):
            return self

    cfg = train_models.DLSearchConfig(
        model_class=_FakeDlModel,
        search_space={},
        default_params={
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "task_type": "regression",
        },
    )

    monkeypatch.setattr(train_models, "_initialize_model", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    fake_torch = types.SimpleNamespace(
        save=lambda obj, path: saved_tensors.__setitem__(str(path), obj),
        load=lambda path, map_location=None, weights_only=True: saved_tensors[str(path)],
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def _fake_train_dl(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs,
        batch_size,
        learning_rate,
        patience,
        random_state,
        task_type,
        device="auto",
    ):
        _ = (X_train, y_train, X_val, y_val, patience, random_state, task_type, device)
        captured["model_params"] = dict(model.params)
        captured["epochs"] = int(epochs)
        captured["batch_size"] = int(batch_size)
        captured["learning_rate"] = float(learning_rate)
        return {"model": model, "best_params": {"epochs": int(epochs)}}

    monkeypatch.setattr(train_models, "_train_dl", _fake_train_dl)
    monkeypatch.setattr(train_models, "_predict_dl", lambda model, X: np.zeros(len(X), dtype=float))

    X = pd.DataFrame(np.ones((16, 4), dtype=float), columns=["f0", "f1", "f2", "f3"])
    y = pd.Series(np.linspace(0.0, 1.0, num=16))
    X_train, X_val, X_test = X.iloc[:8], X.iloc[8:12], X.iloc[12:]
    y_train, y_val, y_test = y.iloc[:8], y.iloc[8:12], y.iloc[12:]

    estimator, train_result = train_models.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="dl_simple",
        output_dir=str(tmp_path),
        task_type="regression",
        model_config={
            "params": {
                "epochs": 3,
                "batch_size": 7,
                "learning_rate": 0.02,
                "hidden_dim": 17,
            }
        },
        X_val=X_val,
        y_val=y_val,
    )

    assert captured["epochs"] == 3
    assert captured["batch_size"] == 7
    assert captured["learning_rate"] == 0.02
    assert captured["model_params"]["hidden_dim"] == 17
    assert estimator.params["hidden_dim"] == 17

    reloaded = train_models.load_model(
        train_result.model_path,
        "dl_simple",
        input_dim=X_train.shape[1],
    )
    assert reloaded.params["hidden_dim"] == 17
    assert reloaded.params["batch_size"] == 7
    assert reloaded.params["learning_rate"] == 0.02
    assert reloaded.params["epochs"] == 3


def test_train_model_dl_fixed_raises_clear_error_for_non_finite_predictions(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeDlModel:
        def __init__(self, params: dict[str, object]) -> None:
            self.params = dict(params)

        def state_dict(self) -> dict[str, object]:
            return {}

    cfg = train_models.DLSearchConfig(
        model_class=_FakeDlModel,
        search_space={},
        default_params={
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "hidden_dim": 16,
        },
    )

    monkeypatch.setattr(train_models, "_initialize_model", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    monkeypatch.setattr(
        train_models,
        "_train_dl",
        lambda *args, **kwargs: {"model": args[0], "best_params": {"epochs": 5}},
    )
    monkeypatch.setattr(
        train_models,
        "_predict_dl",
        lambda model, X: np.array([0.0, np.nan, 1.0, 1.5], dtype=float),
    )

    X = pd.DataFrame(np.ones((16, 4), dtype=float), columns=["f0", "f1", "f2", "f3"])
    y = pd.Series(np.linspace(0.0, 1.0, num=16))
    X_train, X_val, X_test = X.iloc[:8], X.iloc[8:12], X.iloc[12:]
    y_train, y_val, y_test = y.iloc[:8], y.iloc[8:12], y.iloc[12:]

    with pytest.raises(
        ValueError,
        match="dl_simple test scoring: non-finite values in regression predictions",
    ):
        train_models.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type="dl_simple",
            output_dir=str(tmp_path),
            task_type="regression",
            model_config={"params": {"epochs": 5, "batch_size": 8, "learning_rate": 1e-3}},
            X_val=X_val,
            y_val=y_val,
        )
