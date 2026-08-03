from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MLModels import train_models
from MLModels.training import model_loader, sklearn_models


class _FakeModelVersion:
    V2_6 = object()


class _FakeTabPFNClassifier:
    create_calls: list[tuple[object, dict[str, object]]] = []
    load_calls: list[tuple[str, str]] = []

    @classmethod
    def create_default_for_version(cls, version, **overrides):
        cls.create_calls.append((version, dict(overrides)))
        return cls()

    @classmethod
    def load_from_fit_state(cls, path, *, device="cpu"):
        cls.load_calls.append((str(path), str(device)))
        return cls()


@pytest.fixture
def fake_tabpfn(monkeypatch):
    _FakeTabPFNClassifier.create_calls.clear()
    _FakeTabPFNClassifier.load_calls.clear()
    package = types.ModuleType("tabpfn")
    package.TabPFNClassifier = _FakeTabPFNClassifier
    constants = types.ModuleType("tabpfn.constants")
    constants.ModelVersion = _FakeModelVersion
    monkeypatch.setitem(sys.modules, "tabpfn", package)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    return _FakeTabPFNClassifier


def test_build_tabpfn_classifier_pins_version_2_6(fake_tabpfn) -> None:
    estimator = sklearn_models.build_tabular_model(
        model_type="tabpfn",
        random_state=17,
        cv_folds=3,
        search_iters=1,
        n_jobs=4,
        tuning_method="fixed",
        model_params={"n_estimators": 4, "device": "cpu"},
        task_type="classification",
    )

    assert isinstance(estimator, _FakeTabPFNClassifier)
    version, options = fake_tabpfn.create_calls[-1]
    assert version is _FakeModelVersion.V2_6
    assert options == {
        "device": "cpu",
        "ignore_pretraining_limits": True,
        "n_estimators": 4,
        "n_preprocessing_jobs": 4,
        "random_state": 17,
    }


def test_build_tabpfn_rejects_regression_before_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "tabpfn", raising=False)
    monkeypatch.delitem(sys.modules, "tabpfn.constants", raising=False)

    with pytest.raises(ValueError, match="classification"):
        sklearn_models.build_tabular_model(
            model_type="tabpfn",
            random_state=17,
            cv_folds=3,
            search_iters=1,
            n_jobs=1,
            tuning_method="fixed",
            model_params={},
            task_type="regression",
        )


def test_load_model_uses_tabpfn_fit_state(fake_tabpfn, tmp_path: Path) -> None:
    model_path = tmp_path / "tabpfn_best_model.tabpfn_fit"
    model_path.write_bytes(b"test archive placeholder")

    estimator = model_loader.load_model(
        str(model_path),
        "tabpfn",
        is_dl_model=lambda _model_type: False,
        initialize_model=lambda *args, **kwargs: None,
        load_pickle=lambda _path: pytest.fail("TabPFN must not use generic pickle loading"),
    )

    assert isinstance(estimator, _FakeTabPFNClassifier)
    assert fake_tabpfn.load_calls == [(str(model_path), "cpu")]


def test_train_model_saves_tabpfn_fit_state(monkeypatch, tmp_path: Path) -> None:
    class _FakeFittedTabPFN:
        classes_ = np.array([0, 1], dtype=int)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            scores = np.linspace(0.2, 0.8, num=len(X))
            return np.column_stack([1.0 - scores, scores])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        def save_fit_state(self, path):
            Path(path).write_bytes(b"tabpfn fit state")

    monkeypatch.setattr(
        train_models,
        "_initialize_model",
        lambda *args, **kwargs: _FakeFittedTabPFN(),
    )
    X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])

    _, result = train_models.train_model(
        X_train=X.iloc[:4],
        y_train=y.iloc[:4],
        X_test=X.iloc[4:],
        y_test=y.iloc[4:],
        model_type="tabpfn",
        output_dir=str(tmp_path),
        task_type="classification",
        model_config={"params": {"model_version": "2.6"}},
    )

    assert result.model_path.endswith("tabpfn_best_model.tabpfn_fit")
    assert Path(result.model_path).read_bytes() == b"tabpfn fit state"
    assert Path(result.metrics_path).is_file()
