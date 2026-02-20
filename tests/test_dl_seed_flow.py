import sys
import types

import numpy as np

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
