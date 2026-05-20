from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


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
    *,
    dl_search_config_cls: Any,
    train_result_cls: Callable[[str, str, str], Any],
    ensure_dir: Callable[[str], None],
    is_dl_model: Callable[[str], bool],
    parse_runtime_training_options: Callable[[dict[str, Any] | None], Any],
    maybe_sanitize_xgboost_feature_frames: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str | None]],
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
    initialize_model: Callable[..., Any],
    run_optuna: Callable[..., tuple[object, dict[str, Any]]],
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
    model_params = runtime_options.model_params

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

    if task_type == "classification" and model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": random_state,
            "verbose": False,
        }
        params.update(model_config.get("params", {}))
        if not debug_logging:
            if any(key in params for key in ("verbose", "verbose_eval", "logging_level", "silent")):
                logging.info("Global debug logging is off; forcing quiet CatBoost training output.")
            params.pop("verbose_eval", None)
            params.pop("logging_level", None)
            params.pop("silent", None)
            params["verbose"] = False
        estimator = CatBoostClassifier(**params)
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = (X_val, y_val)
        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True
        estimator.fit(X_train, y_train, **fit_kwargs)
        y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        estimator.save_model(model_path)
        best_params = estimator.get_params()
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
                "DL models require a validation split for early stopping/HPO. "
                "Ensure the pipeline includes the split node and set split.val_size > 0."
            )
        if use_hpo:
            logging.info("Running optuna for %s (%s evals)", model_type, hpo_trials)
            estimator, best_params = run_optuna(
                model,
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values,
                hpo_trials,
                random_state,
                patience,
                task_type=task_type,
            )
        else:
            effective_params = {**model.default_params, **model_params}
            logging.info("Training DL model: %s (fixed params)", model_type)
            seed_dl_runtime(int(random_state))
            nn_model = model.model_class(effective_params)
            result = train_dl(
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
        model.fit(X_train, y_train)

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        save_model_pickle(estimator, model_path)
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}

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
