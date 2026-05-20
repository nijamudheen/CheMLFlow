import sys
import types

import numpy as np
import pandas as pd
import pytest

from MLModels import train_models


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        _ = (high, log)
        value = float(low)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        _ = (high, log)
        value = int(low)
        self.params[name] = value
        return value


class _FakeStudy:
    def __init__(self) -> None:
        self.best_params: dict[str, object] = {}

    def optimize(self, objective, n_trials: int, show_progress_bar: bool = False) -> None:
        _ = show_progress_bar
        best_score = float("-inf")
        best_params: dict[str, object] = {}
        for trial_num in range(int(n_trials)):
            trial = _FakeTrial(trial_num)
            score = float(objective(trial))
            if score > best_score:
                best_score = score
                best_params = dict(trial.params)
        self.best_params = best_params


def test_run_optuna_seeds_before_model_construction(monkeypatch) -> None:
    fake_optuna = types.SimpleNamespace(
        Trial=_FakeTrial,
        samplers=types.SimpleNamespace(TPESampler=lambda seed: ("sampler", seed)),
        exceptions=types.SimpleNamespace(TrialPruned=RuntimeError),
        create_study=lambda direction, sampler: _FakeStudy(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    seed_calls: list[int] = []

    def _fake_seed(seed: int) -> None:
        seed_calls.append(int(seed))
        np.random.seed(int(seed))

    monkeypatch.setattr(train_models, "_seed_dl_runtime", _fake_seed)
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
    ):
        _ = (X_train, y_train, X_val, y_val, learning_rate, patience, random_state, task_type)
        return {
            "model": model,
            "best_params": {"epochs": int(epochs), "batch_size": int(batch_size)},
        }

    monkeypatch.setattr(train_models, "_train_dl", _fake_train_dl)

    y_val = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    monkeypatch.setattr(train_models, "_predict_dl", lambda model, X: y_val.copy())

    model_draws: list[int] = []

    def _model_factory(params: dict[str, object]) -> dict[str, object]:
        draw = int(np.random.randint(0, 1_000_000))
        model_draws.append(draw)
        return {"draw": draw, "params": dict(params)}

    cfg = train_models.DLSearchConfig(
        model_class=_model_factory,
        search_space={
            "hidden_dim": {"type": "categorical", "choices": [64, 128]},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32]},
            "epochs": {"type": "categorical", "choices": [10, 20]},
        },
        default_params={},
    )

    x_train = np.ones((6, 3), dtype=float)
    y_train = np.arange(6, dtype=float)
    x_val = np.ones((4, 3), dtype=float)

    np.random.seed(999)
    best_model_a, best_params_a = train_models._run_optuna(
        cfg,
        x_train,
        y_train,
        x_val,
        y_val,
        max_evals=2,
        random_state=123,
        patience=3,
        task_type="regression",
    )

    np.random.seed(12345)
    best_model_b, best_params_b = train_models._run_optuna(
        cfg,
        x_train,
        y_train,
        x_val,
        y_val,
        max_evals=2,
        random_state=123,
        patience=3,
        task_type="regression",
    )

    assert best_params_a == best_params_b
    assert best_model_a["draw"] == best_model_b["draw"]
    assert seed_calls == [124, 125, 124, 125]
    assert len(model_draws) == 4


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
        load=lambda path, weights_only=True: saved_tensors[str(path)],
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
    ):
        _ = (X_train, y_train, X_val, y_val, patience, random_state, task_type)
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


def test_run_optuna_prunes_non_finite_regression_predictions(monkeypatch) -> None:
    class _FakeTrialPruned(RuntimeError):
        pass

    class _AllPrunedStudy:
        def optimize(self, objective, n_trials: int, show_progress_bar: bool = False) -> None:
            _ = show_progress_bar
            for trial_num in range(int(n_trials)):
                trial = _FakeTrial(trial_num)
                try:
                    objective(trial)
                except _FakeTrialPruned:
                    continue

        @property
        def best_params(self) -> dict[str, object]:
            raise ValueError("No trials are completed yet.")

    fake_optuna = types.SimpleNamespace(
        Trial=_FakeTrial,
        samplers=types.SimpleNamespace(TPESampler=lambda seed: ("sampler", seed)),
        exceptions=types.SimpleNamespace(TrialPruned=_FakeTrialPruned),
        create_study=lambda direction, sampler: _AllPrunedStudy(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    monkeypatch.setattr(
        train_models,
        "_train_dl",
        lambda **kwargs: {"model": kwargs["model"], "best_params": {}},
    )
    monkeypatch.setattr(
        train_models,
        "_predict_dl",
        lambda model, X: np.array([0.1, np.nan, 0.3], dtype=float),
    )

    cfg = train_models.DLSearchConfig(
        model_class=lambda params: {"params": dict(params)},
        search_space={
            "epochs": {"type": "categorical", "choices": [10]},
            "batch_size": {"type": "categorical", "choices": [16]},
            "learning_rate": {"type": "categorical", "choices": [1e-3]},
        },
        default_params={},
    )

    x_train = np.ones((6, 3), dtype=float)
    y_train = np.arange(6, dtype=float)
    x_val = np.ones((3, 3), dtype=float)
    y_val = np.array([0.0, 1.0, 2.0], dtype=float)

    with pytest.raises(
        ValueError,
        match=(
            "DL hyperparameter search completed no valid trials; "
            "last prune reason: optuna validation scoring: non-finite values in regression predictions"
        ),
    ):
        train_models._run_optuna(
            cfg,
            x_train,
            y_train,
            x_val,
            y_val,
            max_evals=1,
            random_state=123,
            patience=3,
            task_type="regression",
        )


def test_run_optuna_prunes_non_finite_classification_predictions(monkeypatch) -> None:
    class _FakeTrialPruned(RuntimeError):
        pass

    class _AllPrunedStudy:
        def optimize(self, objective, n_trials: int, show_progress_bar: bool = False) -> None:
            _ = show_progress_bar
            for trial_num in range(int(n_trials)):
                trial = _FakeTrial(trial_num)
                try:
                    objective(trial)
                except _FakeTrialPruned:
                    continue

        @property
        def best_params(self) -> dict[str, object]:
            raise ValueError("No trials are completed yet.")

    fake_optuna = types.SimpleNamespace(
        Trial=_FakeTrial,
        samplers=types.SimpleNamespace(TPESampler=lambda seed: ("sampler", seed)),
        exceptions=types.SimpleNamespace(TrialPruned=_FakeTrialPruned),
        create_study=lambda direction, sampler: _AllPrunedStudy(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    monkeypatch.setattr(
        train_models,
        "_train_dl",
        lambda **kwargs: {"model": kwargs["model"], "best_params": {}},
    )
    monkeypatch.setattr(
        train_models,
        "_predict_dl",
        lambda model, X: np.array([0.0, np.nan, 1.0], dtype=float),
    )

    cfg = train_models.DLSearchConfig(
        model_class=lambda params: {"params": dict(params)},
        search_space={
            "epochs": {"type": "categorical", "choices": [10]},
            "batch_size": {"type": "categorical", "choices": [16]},
            "learning_rate": {"type": "categorical", "choices": [1e-3]},
        },
        default_params={},
    )

    x_train = np.ones((6, 3), dtype=float)
    y_train = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    x_val = np.ones((3, 3), dtype=float)
    y_val = np.array([0, 1, 0], dtype=int)

    with pytest.raises(
        ValueError,
        match=(
            "DL hyperparameter search completed no valid trials; "
            "last prune reason: optuna validation scoring: non-finite values in classification scores"
        ),
    ):
        train_models._run_optuna(
            cfg,
            x_train,
            y_train,
            x_val,
            y_val,
            max_evals=1,
            random_state=123,
            patience=3,
            task_type="classification",
        )


def test_run_optuna_prunes_infinite_classification_logits(monkeypatch) -> None:
    class _FakeTrialPruned(RuntimeError):
        pass

    class _AllPrunedStudy:
        def optimize(self, objective, n_trials: int, show_progress_bar: bool = False) -> None:
            _ = show_progress_bar
            for trial_num in range(int(n_trials)):
                trial = _FakeTrial(trial_num)
                try:
                    objective(trial)
                except _FakeTrialPruned:
                    continue

        @property
        def best_params(self) -> dict[str, object]:
            raise ValueError("No trials are completed yet.")

    fake_optuna = types.SimpleNamespace(
        Trial=_FakeTrial,
        samplers=types.SimpleNamespace(TPESampler=lambda seed: ("sampler", seed)),
        exceptions=types.SimpleNamespace(TrialPruned=_FakeTrialPruned),
        create_study=lambda direction, sampler: _AllPrunedStudy(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    monkeypatch.setattr(
        train_models,
        "_train_dl",
        lambda **kwargs: {"model": kwargs["model"], "best_params": {}},
    )
    monkeypatch.setattr(
        train_models,
        "_predict_dl",
        lambda model, X: np.array([0.0, np.inf, 1.0], dtype=float),
    )

    cfg = train_models.DLSearchConfig(
        model_class=lambda params: {"params": dict(params)},
        search_space={
            "epochs": {"type": "categorical", "choices": [10]},
            "batch_size": {"type": "categorical", "choices": [16]},
            "learning_rate": {"type": "categorical", "choices": [1e-3]},
        },
        default_params={},
    )

    x_train = np.ones((6, 3), dtype=float)
    y_train = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    x_val = np.ones((3, 3), dtype=float)
    y_val = np.array([0, 1, 0], dtype=int)

    with pytest.raises(
        ValueError,
        match=(
            "DL hyperparameter search completed no valid trials; "
            "last prune reason: optuna validation scoring: non-finite values in classification scores"
        ),
    ):
        train_models._run_optuna(
            cfg,
            x_train,
            y_train,
            x_val,
            y_val,
            max_evals=1,
            random_state=123,
            patience=3,
            task_type="classification",
        )


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
