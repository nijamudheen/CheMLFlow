from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from MLModels.train_models import TrainResult


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None


@dataclass(frozen=True)
class TrainSpec:
    model_type: str
    output_dir: str
    task_type: str = "regression"
    random_state: int = 42
    cv_folds: int = 5
    search_iters: int = 100
    use_hpo: bool = False
    hpo_trials: int = 30
    patience: int = 20
    model_config: dict[str, Any] = field(default_factory=dict)


def train(dataset: DatasetSplit, spec: TrainSpec) -> tuple[object, TrainResult]:
    from MLModels import train_models

    return train_models.train_model(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        model_type=spec.model_type,
        output_dir=spec.output_dir,
        random_state=spec.random_state,
        cv_folds=spec.cv_folds,
        search_iters=spec.search_iters,
        use_hpo=spec.use_hpo,
        hpo_trials=spec.hpo_trials,
        patience=spec.patience,
        task_type=spec.task_type,
        model_config=spec.model_config,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
    )


def train_from_frames(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    random_state: int = 42,
    cv_folds: int = 5,
    search_iters: int = 100,
    use_hpo: bool = False,
    hpo_trials: int = 30,
    patience: int = 20,
    task_type: str = "regression",
    model_config: dict[str, Any] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> tuple[object, TrainResult]:
    dataset = DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
    )
    spec = TrainSpec(
        model_type=model_type,
        output_dir=output_dir,
        random_state=random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        use_hpo=use_hpo,
        hpo_trials=hpo_trials,
        patience=patience,
        task_type=task_type,
        model_config=model_config or {},
    )
    return train(dataset, spec)


def load(model_path: str, model_type: str, input_dim: int | None = None) -> object:
    from MLModels import train_models

    return train_models.load_model(
        model_path=model_path,
        model_type=model_type,
        input_dim=input_dim,
    )


def run_explainability(
    *,
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    task_type: str = "regression",
) -> None:
    from MLModels import train_models

    train_models.run_explainability(
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        output_dir=output_dir,
        background_samples=background_samples,
        task_type=task_type,
    )
