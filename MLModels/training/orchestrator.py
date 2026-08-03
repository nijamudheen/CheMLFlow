from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .progress import emit, opaque_fit_finished, opaque_fit_started

_PARENT_LEVEL_MODEL_SEARCH_MESSAGE = (
    "Runtime child-level hyperparameter search is disabled. Use DOE model_search "
    "to create parent-level fixed hyperparameter cases that fan out across CV folds."
)


def _predict_dl_with_batch(
    predict_dl: Callable[..., np.ndarray],
    estimator: object,
    X: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    try:
        signature = inspect.signature(predict_dl)
    except (TypeError, ValueError):
        return predict_dl(estimator, X, batch_size=batch_size)

    parameters = signature.parameters.values()
    supports_batch_size = "batch_size" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if supports_batch_size:
        return predict_dl(estimator, X, batch_size=batch_size)
    return predict_dl(estimator, X)


def _train_dl_with_progress(
    train_dl: Callable[..., dict[str, Any]],
    *args: Any,
    progress_reporter: Any | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pass telemetry only when an injected trainer accepts the new keyword."""

    try:
        signature = inspect.signature(train_dl)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = signature.parameters.values()
        if "progress_reporter" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ):
            kwargs["progress_reporter"] = progress_reporter
    return train_dl(*args, **kwargs)


def _fit_with_opaque_progress(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_type: str,
    progress_reporter: Any | None,
    fit_kwargs: dict[str, Any] | None = None,
) -> None:
    opaque_fit_started(progress_reporter, model_type)
    try:
        estimator.fit(X_train, y_train, **(fit_kwargs or {}))
    except BaseException:
        emit(
            progress_reporter,
            "training_scope_finished",
            "fit",
            status="failed",
            message=f"{model_type} fit failed.",
        )
        raise
    opaque_fit_finished(progress_reporter, model_type)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(container: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [part for part in str(dotted).split(".") if part]
    if not parts:
        return
    current = container
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _suggest_optuna_param(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    if not isinstance(spec, dict):
        raise ValueError(f"Optuna search-space spec for {name!r} must be a mapping.")
    stype = str(spec.get("type", "categorical")).strip().lower()
    if stype == "categorical":
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError(f"Optuna search-space spec for {name!r} must define non-empty choices.")
        return trial.suggest_categorical(name, choices)
    if stype == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        return trial.suggest_int(name, low, high, step=step, log=log)
    if stype == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        log = bool(spec.get("log", False))
        if "step" in spec and spec.get("step") is not None:
            return trial.suggest_float(name, low, high, step=float(spec["step"]), log=log)
        return trial.suggest_float(name, low, high, log=log)
    raise ValueError(f"Unknown Optuna search-space spec type for {name!r}: {stype!r}")


def _sampled_params_from_trial(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for name, spec in search_space.items():
        _set_dotted(sampled, name, _suggest_optuna_param(trial, name, spec))
    return sampled


def _grid_sampler_space(search_space: dict[str, Any]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, dict) or str(spec.get("type", "categorical")).strip().lower() != "categorical":
            raise ValueError("train.tuning.sampler=grid requires categorical Optuna params.")
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError(f"Optuna grid search-space spec for {name!r} must define non-empty choices.")
        grid[name] = choices
    return grid


def _runtime_optuna_sampler(optuna: Any, tuning_cfg: dict[str, Any], search_space: dict[str, Any], seed: int):
    sampler_name = str(tuning_cfg.get("sampler", "tpe")).strip().lower() or "tpe"
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        return optuna.samplers.GridSampler(_grid_sampler_space(search_space))
    raise ValueError("train.tuning.sampler must be one of: tpe, random, grid.")


def _default_runtime_optuna_metric(task_type: str) -> tuple[str, str]:
    if str(task_type).strip().lower() == "classification":
        return "val_auc", "maximize"
    return "val_rmse", "minimize"


def _runtime_optuna_metric_and_direction(tuning_cfg: dict[str, Any], task_type: str) -> tuple[str, str]:
    default_metric, default_direction = _default_runtime_optuna_metric(task_type)
    metric = str(tuning_cfg.get("metric", default_metric)).strip().lower() or default_metric
    direction = str(tuning_cfg.get("direction", default_direction)).strip().lower() or default_direction
    if direction not in {"minimize", "maximize"}:
        raise ValueError("train.tuning.direction must be 'minimize' or 'maximize'.")
    return metric, direction


def _regression_validation_metrics(y_true: Any, y_pred: Any, validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    y_true_arr, y_pred_arr = validate_regression_metric_inputs(
        y_true,
        y_pred,
        context="runtime Optuna validation scoring",
    )
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    return {
        "val_rmse": rmse,
        "val_mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "val_r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def _runtime_metric_value(metrics: dict[str, float | None], metric: str) -> float:
    normalized = metric.strip().lower()
    aliases = {
        "rmse": "val_rmse",
        "mae": "val_mae",
        "r2": "val_r2",
        "auc": "val_auc",
        "auroc": "val_auc",
        "accuracy": "val_accuracy",
        "f1": "val_f1",
        "auprc": "val_auprc",
    }
    key = aliases.get(normalized, normalized)
    value = metrics.get(key)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Runtime Optuna metric {metric!r} is unavailable for this validation split.")
    return float(value)


def _trial_rows(study: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in getattr(study, "trials", []):
        row: dict[str, Any] = {
            "number": getattr(trial, "number", None),
            "state": getattr(getattr(trial, "state", None), "name", str(getattr(trial, "state", ""))),
            "value": getattr(trial, "value", None),
        }
        for key, value in dict(getattr(trial, "params", {}) or {}).items():
            row[f"param_{key}"] = value
        rows.append(row)
    return rows


def _build_catboost_classifier(
    *,
    random_state: int,
    debug_logging: bool,
    model_params: dict[str, Any],
):
    from catboost import CatBoostClassifier

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "verbose": False,
    }
    params.update(model_params or {})
    if not debug_logging:
        if any(key in params for key in ("verbose", "verbose_eval", "logging_level", "silent")):
            logging.info("Global debug logging is off; forcing quiet CatBoost training output.")
        params.pop("verbose_eval", None)
        params.pop("logging_level", None)
        params.pop("silent", None)
        params["verbose"] = False
    return CatBoostClassifier(**params)


def _fit_candidate_model(
    *,
    model_type: str,
    task_type: str,
    is_dl: bool,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    n_jobs: int,
    patience: int,
    debug_logging: bool,
    dl_search_config_cls: Any,
    initialize_model: Callable[..., Any],
    seed_dl_runtime: Callable[[int], None],
    train_dl: Callable[..., dict[str, Any]],
    progress_reporter: Any | None = None,
):
    if task_type == "classification" and model_type == "catboost_classifier":
        estimator = _build_catboost_classifier(
            random_state=random_state,
            debug_logging=debug_logging,
            model_params=params,
        )
        fit_kwargs = {"eval_set": (X_val, y_val), "use_best_model": True}
        _fit_with_opaque_progress(
            estimator,
            X_train,
            y_train,
            model_type=model_type,
            progress_reporter=progress_reporter,
            fit_kwargs=fit_kwargs,
        )
        return estimator, params

    model = initialize_model(
        model_type,
        random_state,
        cv_folds,
        search_iters,
        input_dim=X_train.shape[1] if is_dl else None,
        n_jobs=n_jobs,
        tuning_method="fixed",
        model_params=params,
        task_type=task_type,
    )
    if isinstance(model, dl_search_config_cls):
        effective_params = {**model.default_params, **params}
        seed_dl_runtime(int(random_state))
        nn_model = model.model_class(effective_params)
        result = _train_dl_with_progress(
            train_dl,
            nn_model,
            X_train.values,
            y_train.values,
            X_val.values,
            y_val.values,
            epochs=effective_params["epochs"],
            batch_size=effective_params["batch_size"],
            learning_rate=effective_params["learning_rate"],
            patience=patience,
            random_state=random_state,
            task_type=task_type,
            progress_reporter=progress_reporter,
        )
        return result["model"], {**effective_params, **result["best_params"]}

    _fit_with_opaque_progress(
        model,
        X_train,
        y_train,
        model_type=model_type,
        progress_reporter=progress_reporter,
    )
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    best_params = model.best_params_ if hasattr(model, "best_params_") else params
    return estimator, best_params


def _score_candidate_on_validation(
    *,
    estimator: object,
    model_type: str,
    task_type: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selected_params: dict[str, Any],
    predict_dl: Callable[..., np.ndarray],
    predict_classification_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    classification_metrics_from_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]],
    validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | None]:
    if task_type == "classification":
        y_pred_proba, y_pred_label, _ = predict_classification_outputs(
            estimator=estimator,
            model_type=model_type,
            X=X_val,
        )
        _, _, _, metrics = classification_metrics_from_outputs(
            y_val,
            y_pred_proba,
            y_pred_label,
            context=f"{model_type} runtime Optuna validation scoring",
        )
        return {f"val_{key}": value for key, value in metrics.items()}

    if str(model_type).startswith("dl_"):
        batch_size = max(1, int(selected_params.get("batch_size", 64)))
        y_val_pred = _predict_dl_with_batch(
            predict_dl,
            estimator,
            X_val.values,
            batch_size=batch_size,
        )
    else:
        y_val_pred = estimator.predict(X_val)
    return _regression_validation_metrics(y_val, y_val_pred, validate_regression_metric_inputs)


def _run_runtime_optuna(
    *,
    model_type: str,
    task_type: str,
    is_dl: bool,
    base_model_params: dict[str, Any],
    tuning_cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    n_jobs: int,
    patience: int,
    debug_logging: bool,
    dl_search_config_cls: Any,
    initialize_model: Callable[..., Any],
    seed_dl_runtime: Callable[[int], None],
    train_dl: Callable[..., dict[str, Any]],
    predict_dl: Callable[..., np.ndarray],
    predict_classification_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    classification_metrics_from_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]],
    validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]],
    progress_reporter: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise ValueError(
            "train.tuning.method=optuna requires the optuna package. "
            "Install optuna or use train.tuning.method=fixed."
        ) from exc

    search_space = tuning_cfg.get("params", {})
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("train.tuning.params must be a non-empty mapping for runtime Optuna.")

    n_trials = int(tuning_cfg.get("n_trials", tuning_cfg.get("hpo_trials", 30)))
    if n_trials <= 0:
        raise ValueError("train.tuning.n_trials must be > 0 for runtime Optuna.")
    sampler_seed = int(tuning_cfg.get("seed", random_state))
    metric, direction = _runtime_optuna_metric_and_direction(tuning_cfg, task_type)
    sampler = _runtime_optuna_sampler(optuna, tuning_cfg, search_space, sampler_seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass

    emit(
        progress_reporter,
        "training_started",
        "trial",
        unit="trial",
        total=n_trials,
        phase="hyperparameter_search",
        message=f"Runtime Optuna search started for {model_type}.",
    )

    def objective(trial: Any) -> float:
        sampled_params = _sampled_params_from_trial(trial, search_space)
        candidate_params = _deep_merge_dicts(base_model_params, sampled_params)
        logging.info(
            "[optuna trial %d/%d] start params=%s",
            int(getattr(trial, "number", 0)) + 1,
            n_trials,
            sampled_params,
        )
        try:
            estimator, selected_params = _fit_candidate_model(
                model_type=model_type,
                task_type=task_type,
                is_dl=is_dl,
                params=candidate_params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                random_state=random_state,
                cv_folds=cv_folds,
                search_iters=search_iters,
                n_jobs=n_jobs,
                patience=patience,
                debug_logging=debug_logging,
                dl_search_config_cls=dl_search_config_cls,
                initialize_model=initialize_model,
                seed_dl_runtime=seed_dl_runtime,
                train_dl=train_dl,
                progress_reporter=progress_reporter,
            )
            val_metrics = _score_candidate_on_validation(
                estimator=estimator,
                model_type=model_type,
                task_type=task_type,
                X_val=X_val,
                y_val=y_val,
                selected_params=selected_params,
                predict_dl=predict_dl,
                predict_classification_outputs=predict_classification_outputs,
                classification_metrics_from_outputs=classification_metrics_from_outputs,
                validate_regression_metric_inputs=validate_regression_metric_inputs,
            )
            score = _runtime_metric_value(val_metrics, metric)
        except Exception as exc:
            logging.warning("Runtime Optuna trial failed during validation scoring: %s", exc)
            emit(
                progress_reporter,
                "training_update",
                "trial",
                current=int(getattr(trial, "number", 0)) + 1,
                total=n_trials,
                phase="hyperparameter_search",
                message=f"Optuna trial pruned: {exc}",
            )
            raise optuna.exceptions.TrialPruned(str(exc)) from exc
        logging.info(
            "[optuna trial %d/%d] complete: validation %s=%.6f",
            int(getattr(trial, "number", 0)) + 1,
            n_trials,
            metric,
            score,
        )
        emit(
            progress_reporter,
            "training_update",
            "trial",
            current=int(getattr(trial, "number", 0)) + 1,
            total=n_trials,
            phase="hyperparameter_search",
            message=f"Optuna trial completed: {metric}={score:.6f}.",
            metrics={metric: score},
        )
        return score

    logging.info(
        "Starting runtime Optuna HPO for %s: n_trials=%d metric=%s direction=%s axes=%s",
        model_type,
        n_trials,
        metric,
        direction,
        sorted(search_space),
    )
    try:
        study.optimize(objective, n_trials=n_trials)
    except BaseException:
        emit(
            progress_reporter,
            "training_scope_finished",
            "trial",
            status="failed",
            message=f"Runtime Optuna search failed for {model_type}.",
        )
        raise
    emit(
        progress_reporter,
        "training_scope_finished",
        "trial",
        message=f"Runtime Optuna search completed for {model_type}.",
    )
    try:
        best_trial = study.best_trial
        best_value = study.best_value
    except ValueError:
        raise ValueError(f"Runtime Optuna search for {model_type} produced no valid trials.")

    trial_rows = _trial_rows(study)
    trials_path = os.path.join(output_dir, f"{model_type}_optuna_trials.csv")
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)

    best_sampled_flat = dict(study.best_trial.params)
    best_sampled: dict[str, Any] = {}
    for key, value in best_sampled_flat.items():
        _set_dotted(best_sampled, key, value)
    best_params = _deep_merge_dicts(base_model_params, best_sampled)
    summary = {
        "method": "optuna",
        "metric": metric,
        "direction": direction,
        "n_trials": n_trials,
        "sampler": str(tuning_cfg.get("sampler", "tpe")).strip().lower() or "tpe",
        "seed": sampler_seed,
        "best_trial": int(best_trial.number),
        "best_value": float(best_value),
        "best_params": best_params,
        "best_sampled_params": best_sampled,
        "trials_path": trials_path,
    }
    summary_path = os.path.join(output_dir, f"{model_type}_optuna_summary.json")
    summary["summary_path"] = summary_path
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logging.info(
        "Runtime Optuna HPO complete for %s: best validation %s=%.6f params=%s",
        model_type,
        metric,
        float(best_value),
        best_params,
    )
    return best_params, summary


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
    model_config: dict[str, Any] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    progress_reporter: Any | None = None,
    *,
    dl_search_config_cls: Any,
    train_result_cls: Callable[[str, str, str], Any],
    ensure_dir: Callable[[str], None],
    is_dl_model: Callable[[str], bool],
    parse_runtime_training_options: Callable[[dict[str, Any] | None], Any],
    maybe_sanitize_xgboost_feature_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str | None]],
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
    initialize_model: Callable[..., Any],
    seed_dl_runtime: Callable[[int], None],
    train_dl: Callable[..., dict[str, Any]],
    predict_dl: Callable[..., np.ndarray],
    classification_metrics_from_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]],
    save_split_metrics_artifacts: Callable[[str, str, dict[str, dict[str, float | None]]], tuple[str | None, str | None]],
    save_classification_split_plots: Callable[[str, str, dict[str, dict[str, Any]]], dict[str, str]],
    save_regression_parity_plots: Callable[[str, str, dict[str, tuple[Any, Any]]], dict[str, str]],
    save_roc_curve: Callable[[str, str, Any, Any], str | None],
    predict_classification_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]],
    safe_r2: Callable[[Any, Any], float | None],
    safe_mae: Callable[[Any, Any], float | None],
    save_model_pickle: Callable[[Any, str], None],
    save_torch_state_dict: Callable[[Any, str], None],
    save_params: Callable[[dict[str, Any], str], None],
    save_metrics_series: Callable[[dict[str, Any], str], None],
) -> tuple[object, Any]:
    ensure_dir(output_dir)
    is_dl = is_dl_model(model_type)
    runtime_options = parse_runtime_training_options(model_config)
    model_config = runtime_options.model_config or {}
    plot_split_performance = runtime_options.plot_split_performance
    debug_logging = runtime_options.debug_logging
    n_jobs = runtime_options.n_jobs
    tuning_method = runtime_options.tuning_method
    tuning_cfg = runtime_options.tuning_config
    model_params = runtime_options.model_params
    if use_hpo:
        raise ValueError(_PARENT_LEVEL_MODEL_SEARCH_MESSAGE)
    optuna_summary: dict[str, Any] | None = None

    logging.info(
        "Training start: model=%s task=%s tuning=%s X_train=%s X_test=%s",
        model_type,
        task_type,
        tuning_method,
        X_train.shape,
        X_test.shape,
    )
    X_train, X_test, X_val, feature_name_map_path = maybe_sanitize_xgboost_feature_frames(
        model_type=model_type,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        output_dir=output_dir,
    )

    if task_type == "classification":
        y_train = ensure_binary_labels(y_train)
        y_test = ensure_binary_labels(y_test)
        if y_val is not None:
            y_val = ensure_binary_labels(y_val)

    if task_type == "regression" and model_type == "catboost_classifier":
        raise ValueError("Model type 'catboost_classifier' only supports classification tasks.")

    if tuning_method == "optuna":
        if X_val is None or y_val is None or len(y_val) == 0:
            raise ValueError(
                "train.tuning.method=optuna requires a validation split. "
                "Ensure the pipeline includes split and set split.val_size > 0."
            )
        model_params, optuna_summary = _run_runtime_optuna(
            model_type=model_type,
            task_type=task_type,
            is_dl=is_dl,
            base_model_params=model_params,
            tuning_cfg=tuning_cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=output_dir,
            random_state=random_state,
            cv_folds=cv_folds,
            search_iters=search_iters,
            n_jobs=n_jobs,
            patience=patience,
            debug_logging=debug_logging,
            dl_search_config_cls=dl_search_config_cls,
            initialize_model=initialize_model,
            seed_dl_runtime=seed_dl_runtime,
            train_dl=train_dl,
            predict_dl=predict_dl,
            predict_classification_outputs=predict_classification_outputs,
            classification_metrics_from_outputs=classification_metrics_from_outputs,
            validate_regression_metric_inputs=validate_regression_metric_inputs,
            progress_reporter=progress_reporter,
        )
        tuning_method = "fixed"
    elif tuning_method != "fixed":
        raise ValueError(
            f"Unsupported runtime tuning method {tuning_method!r}. "
            + _PARENT_LEVEL_MODEL_SEARCH_MESSAGE
        )

    if task_type == "classification" and model_type == "catboost_classifier":
        estimator = _build_catboost_classifier(
            random_state=random_state,
            debug_logging=debug_logging,
            model_params=model_params,
        )
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = (X_val, y_val)
        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True
        _fit_with_opaque_progress(
            estimator,
            X_train,
            y_train,
            model_type=model_type,
            progress_reporter=progress_reporter,
            fit_kwargs=fit_kwargs,
        )
        y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        estimator.save_model(model_path)
        best_params = model_params if optuna_summary is not None else estimator.get_params()
        y_test_arr, y_pred_proba, y_pred, metrics = classification_metrics_from_outputs(
            y_test,
            y_pred_proba,
            y_pred,
            context=f"{model_type} test scoring",
        )
        if plot_split_performance:
            train_proba = estimator.predict_proba(X_train)[:, 1]
            train_pred = estimator.predict(X_train)
            y_train_arr, train_proba, train_pred, train_metrics = classification_metrics_from_outputs(
                y_train,
                train_proba,
                train_pred,
                context=f"{model_type} train scoring",
            )
            split_metrics: dict[str, dict[str, float | None]] = {
                "train": train_metrics,
                "test": metrics.copy(),
            }
            split_outputs: dict[str, dict[str, Any]] = {
                "train": {
                    "y_true": y_train_arr,
                    "y_proba": train_proba,
                    "y_pred": train_pred,
                },
                "test": {
                    "y_true": y_test_arr,
                    "y_proba": y_pred_proba,
                    "y_pred": y_pred,
                },
            }
            if X_val is not None and y_val is not None and len(y_val) > 0:
                y_val_proba = estimator.predict_proba(X_val)[:, 1]
                y_val_pred = estimator.predict(X_val)
                y_val_arr, y_val_proba, y_val_pred, val_metrics = classification_metrics_from_outputs(
                    y_val,
                    y_val_proba,
                    y_val_pred,
                    context=f"{model_type} val scoring",
                )
                split_metrics["val"] = val_metrics
                split_outputs["val"] = {
                    "y_true": y_val_arr,
                    "y_proba": y_val_proba,
                    "y_pred": y_val_pred,
                }
            split_metrics_path, split_plot_path = save_split_metrics_artifacts(
                output_dir,
                model_type,
                split_metrics,
            )
            if split_metrics_path:
                metrics["split_metrics_path"] = split_metrics_path
            if split_plot_path:
                metrics["split_metrics_plot_path"] = split_plot_path
            metrics.update(save_classification_split_plots(output_dir, model_type, split_outputs))

        roc_path = save_roc_curve(output_dir, model_type, y_test, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path
        if optuna_summary is not None:
            metrics["tuning"] = optuna_summary
        params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
        save_params(best_params, params_path)
        save_metrics_series(metrics, metrics_path)
        logging.info("Training complete (classification): metrics=%s", metrics)
        logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)
        return estimator, train_result_cls(model_path, params_path, metrics_path)

    classification_model_types = {
        "catboost_classifier",
        "random_forest",
        "decision_tree",
        "xgboost",
        "svm",
        "ensemble",
        "tabpfn",
    }

    if task_type == "classification" and not (model_type in classification_model_types or is_dl):
        raise ValueError(f"Unsupported classification model type: {model_type}")

    model = initialize_model(
        model_type,
        random_state,
        cv_folds,
        search_iters,
        input_dim=X_train.shape[1] if is_dl else None,
        n_jobs=n_jobs,
        tuning_method=tuning_method,
        model_params=model_params,
        task_type=task_type,
    )
    if isinstance(model, dl_search_config_cls):
        if X_val is None or y_val is None or len(y_val) == 0:
            raise ValueError(
                "DL models require a validation split for early stopping. "
                "Ensure the pipeline includes the split node and set split.val_size > 0."
            )
        effective_params = {**model.default_params, **model_params}
        logging.info("Training DL model: %s (fixed params)", model_type)
        seed_dl_runtime(int(random_state))
        nn_model = model.model_class(effective_params)
        result = _train_dl_with_progress(
            train_dl,
            nn_model,
            X_train.values,
            y_train.values,
            X_val.values,
            y_val.values,
            epochs=effective_params["epochs"],
            batch_size=effective_params["batch_size"],
            learning_rate=effective_params["learning_rate"],
            patience=patience,
            random_state=random_state,
            task_type=task_type,
            progress_reporter=progress_reporter,
        )
        estimator = result["model"]
        best_params = {**effective_params, **result["best_params"]}

        model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        dl_batch_size = max(1, int(best_params.get("batch_size", 64)))
        y_pred = _predict_dl_with_batch(
            predict_dl,
            estimator,
            X_test.values,
            batch_size=dl_batch_size,
        )
    else:
        logging.info("Training ML model: %s", model_type)
        _fit_with_opaque_progress(
            model,
            X_train,
            y_train,
            model_type=model_type,
            progress_reporter=progress_reporter,
        )

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = estimator.predict(X_test)
        if model_type == "tabpfn":
            model_path = os.path.join(output_dir, f"{model_type}_best_model.tabpfn_fit")
            estimator.save_fit_state(model_path)
        else:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
            save_model_pickle(estimator, model_path)
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        if optuna_summary is not None:
            best_params = model_params if not best_params else _deep_merge_dicts(model_params, best_params)

    if task_type == "classification":
        y_pred_proba, y_pred_label, y_pred_score = predict_classification_outputs(
            estimator=estimator,
            model_type=model_type,
            X=X_test,
        )
        y_true, y_pred_proba, y_pred_label, metrics = classification_metrics_from_outputs(
            y_test,
            y_pred_proba,
            y_pred_label,
            context=f"{model_type} test scoring",
        )
        roc_path = save_roc_curve(output_dir, model_type, y_true, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path

        pred_path = os.path.join(output_dir, f"{model_type}_predictions.csv")
        pd.DataFrame(
            {
                "y_true": y_true,
                "y_score": np.asarray(y_pred_score).reshape(-1),
                "y_proba": y_pred_proba,
                "y_pred": y_pred_label,
            }
        ).to_csv(pred_path, index=False)
    else:
        y_true, y_pred = validate_regression_metric_inputs(
            y_test,
            y_pred,
            context=f"{model_type} test scoring",
        )
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    if feature_name_map_path:
        metrics["feature_name_map_path"] = feature_name_map_path
    if optuna_summary is not None:
        metrics["tuning"] = optuna_summary

    if plot_split_performance:
        split_metrics: dict[str, dict[str, float | None]] = {}

        if task_type == "classification":
            split_outputs: dict[str, dict[str, Any]] = {}
            tr_proba, tr_pred, _ = predict_classification_outputs(
                estimator=estimator,
                model_type=model_type,
                X=X_train,
            )
            y_tr, tr_proba, tr_pred, train_metrics = classification_metrics_from_outputs(
                y_train,
                tr_proba,
                tr_pred,
                context=f"{model_type} train scoring",
            )
            split_metrics["train"] = train_metrics
            split_outputs["train"] = {
                "y_true": y_tr,
                "y_proba": tr_proba,
                "y_pred": tr_pred,
            }

            te_proba, te_pred, _ = predict_classification_outputs(
                estimator=estimator,
                model_type=model_type,
                X=X_test,
            )
            y_te, te_proba, te_pred, test_metrics = classification_metrics_from_outputs(
                y_test,
                te_proba,
                te_pred,
                context=f"{model_type} test scoring",
            )
            split_metrics["test"] = test_metrics
            split_outputs["test"] = {
                "y_true": y_te,
                "y_proba": te_proba,
                "y_pred": te_pred,
            }

            if X_val is not None and y_val is not None and len(y_val) > 0:
                va_proba, va_pred, _ = predict_classification_outputs(
                    estimator=estimator,
                    model_type=model_type,
                    X=X_val,
                )
                y_va, va_proba, va_pred, val_metrics = classification_metrics_from_outputs(
                    y_val,
                    va_proba,
                    va_pred,
                    context=f"{model_type} val scoring",
                )
                split_metrics["val"] = val_metrics
                split_outputs["val"] = {
                    "y_true": y_va,
                    "y_proba": va_proba,
                    "y_pred": va_pred,
                }

        else:
            if is_dl:
                dl_batch_size = max(1, int(best_params.get("batch_size", 64)))
                y_train_pred = _predict_dl_with_batch(
                    predict_dl,
                    estimator,
                    X_train.values,
                    batch_size=dl_batch_size,
                )
                y_test_pred = _predict_dl_with_batch(
                    predict_dl,
                    estimator,
                    X_test.values,
                    batch_size=dl_batch_size,
                )
                y_val_pred = (
                    _predict_dl_with_batch(
                        predict_dl,
                        estimator,
                        X_val.values,
                        batch_size=dl_batch_size,
                    )
                    if X_val is not None
                    else None
                )
            else:
                y_train_pred = estimator.predict(X_train)
                y_test_pred = estimator.predict(X_test)
                y_val_pred = estimator.predict(X_val) if X_val is not None else None

            split_metrics = {
                "train": {"r2": safe_r2(y_train, y_train_pred), "mae": safe_mae(y_train, y_train_pred)},
                "test": {"r2": safe_r2(y_test, y_test_pred), "mae": safe_mae(y_test, y_test_pred)},
            }
            if X_val is not None and y_val is not None and len(y_val) > 0 and y_val_pred is not None:
                split_metrics["val"] = {"r2": safe_r2(y_val, y_val_pred), "mae": safe_mae(y_val, y_val_pred)}

        split_metrics_path, split_plot_path = save_split_metrics_artifacts(output_dir, model_type, split_metrics)
        if split_metrics_path:
            metrics["split_metrics_path"] = split_metrics_path
        if split_plot_path:
            metrics["split_metrics_plot_path"] = split_plot_path

        if task_type == "classification":
            metrics.update(save_classification_split_plots(output_dir, model_type, split_outputs))
        else:
            parity_paths = save_regression_parity_plots(
                output_dir,
                model_type,
                {
                    "train": (y_train, y_train_pred),
                    "test": (y_test, y_test_pred),
                    "val": (y_val, y_val_pred),
                },
            )
            for split_name, path in parity_paths.items():
                metrics[f"parity_plot_{split_name}_path"] = path

    params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    if is_dl:
        save_torch_state_dict(estimator, model_path)
    save_params(best_params, params_path)
    save_metrics_series(metrics, metrics_path)
    logging.info("Training complete (%s): metrics=%s", task_type, metrics)
    logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)

    return estimator, train_result_cls(model_path, params_path, metrics_path)
