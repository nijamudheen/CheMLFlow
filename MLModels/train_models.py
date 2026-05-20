from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd

from MLModels.training import config as training_config
from MLModels.training import chemprop_models as training_chemprop_models
from MLModels.training import explainability as training_explainability
from MLModels.training import model_loader as training_model_loader
from MLModels.training import model_factory as training_model_factory
from MLModels.training import metrics as training_metrics
from MLModels.training import orchestrator as training_orchestrator
from MLModels.training import persistence as training_persistence
from MLModels.training import plots as training_plots
from MLModels.training import train_helpers as training_train_helpers
from MLModels.training import torch_models as training_torch_models

_ROW_INDEX_COL = "__row_index"


@dataclass
class TrainResult:
    model_path: str
    params_path: str
    metrics_path: str

@dataclass
class DLSearchConfig:
    model_class: Callable
    search_space: Dict[str, Any]
    default_params: Dict[str, Any]

def _ensure_dir(path: str) -> None:
    training_persistence.ensure_dir(path)


def _sanitize_xgboost_feature_columns(columns: pd.Index) -> tuple[list[str], list[dict[str, Any]], int]:
    return training_train_helpers.sanitize_xgboost_feature_columns(columns)


def _assign_feature_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame | None:
    return training_train_helpers.assign_feature_columns(df, columns)


def _sanitize_xgboost_feature_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    return training_train_helpers.sanitize_xgboost_feature_frames(X_train, X_test, X_val)


def _maybe_sanitize_xgboost_feature_frames(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str | None]:
    return training_train_helpers.maybe_sanitize_xgboost_feature_frames(
        model_type=model_type,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        output_dir=output_dir,
    )


def _maybe_apply_xgboost_feature_map_for_explain(
    model_type: str,
    output_dir: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return training_train_helpers.maybe_apply_xgboost_feature_map_for_explain(
        model_type=model_type,
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
    )

def _resolve_n_jobs(model_config: Dict[str, Any] | None = None) -> int:
    return training_config.resolve_n_jobs(model_config)


def _validate_classification_score_values(
    y_score: Any,
    *,
    context: str,
) -> np.ndarray:
    return training_metrics.validate_classification_score_values(y_score, context=context)


def _safe_auc(y_true, y_score) -> float | None:
    return training_metrics.safe_auc(y_true, y_score)


def _safe_auprc(y_true, y_score) -> float | None:
    return training_metrics.safe_auprc(y_true, y_score)


def _save_roc_curve(output_dir: str, model_type: str, y_true, y_score) -> str | None:
    return training_plots.save_roc_curve(output_dir, model_type, y_true, y_score)


def _sigmoid(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    return training_train_helpers.sigmoid(values)


def _predict_classification_outputs(
    estimator: object,
    model_type: str,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return training_train_helpers.predict_classification_outputs(
        estimator=estimator,
        model_type=model_type,
        X=X,
        predict_dl=lambda model, values: _predict_dl(model, values),
        validate_classification_score_values=_validate_classification_score_values,
        ensure_binary_labels=_ensure_binary_labels,
        sigmoid_fn=_sigmoid,
    )


def _save_pr_curve(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true,
    y_score,
) -> str | None:
    return training_plots.save_pr_curve(output_dir, model_type, split_name, y_true, y_score)


def _save_confusion_matrix_plot(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true,
    y_pred,
) -> str | None:
    return training_plots.save_confusion_matrix_plot(output_dir, model_type, split_name, y_true, y_pred)


def _save_classification_split_plots(
    output_dir: str,
    model_type: str,
    split_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    return training_plots.save_classification_split_plots(output_dir, model_type, split_outputs)


def _as_bool(value: Any) -> bool:
    return training_config.as_bool(value)


def _parse_runtime_training_options(model_config: Dict[str, Any] | None):
    return training_config.parse_runtime_training_options(model_config)


def _validate_regression_metric_inputs(
    y_true: Any,
    y_pred: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    return training_metrics.validate_regression_metric_inputs(y_true, y_pred, context=context)


def _validate_classification_metric_inputs(
    y_true: Any,
    y_score: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    return training_metrics.validate_classification_metric_inputs(y_true, y_score, context=context)


def _classification_metrics_from_outputs(
    y_true: Any,
    y_score: Any,
    y_pred: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]:
    return training_metrics.classification_metrics_from_outputs(
        y_true,
        y_score,
        y_pred,
        context=context,
        ensure_binary_labels=_ensure_binary_labels,
    )


def _safe_r2(y_true, y_pred) -> float | None:
    return training_metrics.safe_r2(y_true, y_pred)


def _safe_mae(y_true, y_pred) -> float | None:
    return training_metrics.safe_mae(y_true, y_pred)


def _save_split_metrics_artifacts(
    output_dir: str,
    model_type: str,
    split_metrics: Dict[str, Dict[str, float | None]],
) -> tuple[str | None, str | None]:
    return training_plots.save_split_metrics_artifacts(output_dir, model_type, split_metrics)


def _save_regression_parity_plots(
    output_dir: str,
    model_type: str,
    split_predictions: Dict[str, tuple[Any, Any]],
) -> Dict[str, str]:
    return training_plots.save_regression_parity_plots(output_dir, model_type, split_predictions)


def _ensure_binary_labels(series: pd.Series) -> pd.Series:
    return training_train_helpers.ensure_binary_labels(series)


def _require_chemprop() -> None:
    training_train_helpers.require_chemprop()


def _resolve_chemprop_foundation_config(model_config: Dict[str, Any]) -> tuple[str, str | None, bool]:
    return training_config.resolve_chemprop_foundation_config(model_config)


def _resolve_chemprop_split_positions(
    curated_df: pd.DataFrame,
    split_indices: Dict[str, Any],
    row_index_col: str = _ROW_INDEX_COL,
    allow_legacy_positions: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    return training_train_helpers.resolve_chemprop_split_positions(
        curated_df=curated_df,
        split_indices=split_indices,
        row_index_col=row_index_col,
        allow_legacy_positions=allow_legacy_positions,
    )


def _resolve_chemprop_predictor_ctor(nn_module, task_type: str):
    return training_train_helpers.resolve_chemprop_predictor_ctor(nn_module, task_type)


def train_chemprop_model(
    curated_df: pd.DataFrame,
    target_column: str,
    split_indices: Dict[str, Any],
    output_dir: str,
    random_state: int = 42,
    task_type: str = "classification",
    model_config: Dict[str, Any] | None = None,
) -> Tuple[object, TrainResult]:
    return training_chemprop_models.train_chemprop_model(
        curated_df=curated_df,
        target_column=target_column,
        split_indices=split_indices,
        output_dir=output_dir,
        random_state=random_state,
        task_type=task_type,
        model_config=model_config,
        row_index_col=_ROW_INDEX_COL,
        ensure_dir=_ensure_dir,
        require_chemprop=_require_chemprop,
        resolve_chemprop_foundation_config=_resolve_chemprop_foundation_config,
        as_bool=_as_bool,
        resolve_chemprop_split_positions=_resolve_chemprop_split_positions,
        ensure_binary_labels=_ensure_binary_labels,
        resolve_chemprop_predictor_ctor=_resolve_chemprop_predictor_ctor,
        validate_classification_score_values=_validate_classification_score_values,
        sigmoid=_sigmoid,
        classification_metrics_from_outputs=_classification_metrics_from_outputs,
        validate_regression_metric_inputs=_validate_regression_metric_inputs,
        safe_r2=_safe_r2,
        safe_mae=_safe_mae,
        save_split_metrics_artifacts=_save_split_metrics_artifacts,
        save_classification_split_plots=_save_classification_split_plots,
        save_regression_parity_plots=_save_regression_parity_plots,
        save_params=training_persistence.save_params,
        save_metrics_json=lambda metrics, path: training_persistence.save_metrics_json(metrics, path, indent=2),
        train_result_cls=TrainResult,
    )

def _run_optuna(
    config: DLSearchConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_evals: int,
    random_state: int,
    patience: int,
    task_type: str = "regression",
) -> Tuple[object, dict]:
    return training_torch_models.run_optuna(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evals=max_evals,
        random_state=random_state,
        patience=patience,
        task_type=task_type,
        seed_fn=_seed_dl_runtime,
        train_fn=_train_dl,
        predict_fn=_predict_dl,
    )

def _initialize_model(
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    input_dim: int = None,
    n_jobs: int = -1,
    tuning_method: str = "fixed",
    model_params: Dict[str, Any] | None = None,
    task_type: str = "regression",
):
    return training_model_factory.initialize_model(
        model_type=model_type,
        random_state=random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        input_dim=input_dim,
        n_jobs=n_jobs,
        tuning_method=tuning_method,
        model_params=model_params,
        task_type=task_type,
        dl_search_config_cls=DLSearchConfig,
    )

# DL training helper functions
def _get_device():
    return training_torch_models.get_device()


def _seed_dl_runtime(seed: int) -> None:
    training_torch_models.seed_dl_runtime(seed)


def _is_dl_model(model_type: str) -> bool:
    return training_model_factory.is_dl_model(model_type)
def _train_dl(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    random_state: int = 42,
    task_type: str = "regression",
    ) -> Dict[str, Any]:
    return training_torch_models.train_dl(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        random_state=random_state,
        task_type=task_type,
    )

def _predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    return training_torch_models.predict_dl(model=model, X=X, batch_size=batch_size)


def train_model(
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
    model_config: Dict[str, Any] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[object, TrainResult]:
    return training_orchestrator.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        output_dir=output_dir,
        random_state=random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        use_hpo=use_hpo,
        hpo_trials=hpo_trials,
        patience=patience,
        task_type=task_type,
        model_config=model_config,
        X_val=X_val,
        y_val=y_val,
        dl_search_config_cls=DLSearchConfig,
        train_result_cls=TrainResult,
        ensure_dir=_ensure_dir,
        is_dl_model=_is_dl_model,
        parse_runtime_training_options=_parse_runtime_training_options,
        maybe_sanitize_xgboost_feature_frames=_maybe_sanitize_xgboost_feature_frames,
        ensure_binary_labels=_ensure_binary_labels,
        initialize_model=_initialize_model,
        run_optuna=_run_optuna,
        seed_dl_runtime=_seed_dl_runtime,
        train_dl=_train_dl,
        predict_dl=_predict_dl,
        classification_metrics_from_outputs=_classification_metrics_from_outputs,
        save_split_metrics_artifacts=_save_split_metrics_artifacts,
        save_classification_split_plots=_save_classification_split_plots,
        save_regression_parity_plots=_save_regression_parity_plots,
        save_roc_curve=_save_roc_curve,
        predict_classification_outputs=_predict_classification_outputs,
        validate_regression_metric_inputs=_validate_regression_metric_inputs,
        safe_r2=_safe_r2,
        safe_mae=_safe_mae,
        save_model_pickle=training_persistence.save_model_pickle,
        save_torch_state_dict=training_persistence.save_torch_state_dict,
        save_params=training_persistence.save_params,
        save_metrics_series=training_persistence.save_metrics_series,
    )

def load_model(model_path: str, model_type: str, input_dim: int | None = None) -> object:
    return training_model_loader.load_model(
        model_path=model_path,
        model_type=model_type,
        input_dim=input_dim,
        is_dl_model=_is_dl_model,
        initialize_model=_initialize_model,
        load_pickle=training_persistence.load_pickle,
    )

def run_explainability(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    task_type: str = "regression",
) -> None:
    training_explainability.run_explainability(
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        output_dir=output_dir,
        background_samples=background_samples,
        task_type=task_type,
        ensure_dir=_ensure_dir,
        ensure_binary_labels=_ensure_binary_labels,
        resolve_n_jobs=_resolve_n_jobs,
        maybe_apply_xgboost_feature_map_for_explain=_maybe_apply_xgboost_feature_map_for_explain,
        get_device=_get_device,
        predict_dl=_predict_dl,
        validate_classification_score_values=_validate_classification_score_values,
        sigmoid=_sigmoid,
        safe_auc=_safe_auc,
        validate_regression_metric_inputs=_validate_regression_metric_inputs,
    )
