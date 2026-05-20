from __future__ import annotations

import logging
import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score


def run_explainability(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    task_type: str = "regression",
    *,
    ensure_dir: Callable[[str], None],
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
    resolve_n_jobs: Callable[..., int],
    maybe_apply_xgboost_feature_map_for_explain: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
    get_device: Callable[[], Any],
    predict_dl: Callable[[object, np.ndarray, int], np.ndarray] | Callable[[object, np.ndarray], np.ndarray],
    validate_classification_score_values: Callable[..., np.ndarray],
    sigmoid: Callable[[np.ndarray | pd.Series | list[float]], np.ndarray],
    safe_auc: Callable[[Any, Any], float | None],
    validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    ensure_dir(output_dir)
    if task_type == "classification":
        y_test = ensure_binary_labels(y_test)
    is_dl = model_type.startswith("dl_")
    n_jobs = resolve_n_jobs()
    logging.info(
        "Explainability start: model=%s task=%s X_train=%s X_test=%s n_jobs=%s output_dir=%s",
        model_type,
        task_type,
        X_train.shape,
        X_test.shape,
        n_jobs,
        output_dir,
    )
    X_train, X_test = maybe_apply_xgboost_feature_map_for_explain(
        model_type=model_type,
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
    )
    logging.info(
        "Explainability feature matrix ready: model=%s X_train=%s X_test=%s",
        model_type,
        X_train.shape,
        X_test.shape,
    )

    try:
        import shap
    except Exception as exc:
        logging.warning("SHAP is not available; skipping SHAP explainability. %s", exc)
        return

    if is_dl:
        class _SklearnWrapper:
            def __init__(self, model, task_type: str):
                self.model = model
                self.task_type = task_type

            def fit(self, X, y):
                return self

            def predict(self, X):
                y_out = predict_dl(self.model, X.values if hasattr(X, "values") else X)
                if self.task_type == "classification":
                    scores = validate_classification_score_values(
                        y_out,
                        context="dl explainability prediction",
                    )
                    proba = sigmoid(scores)
                    return (proba >= 0.5).astype(int)
                return y_out

            def predict_proba(self, X):
                y_out = predict_dl(self.model, X.values if hasattr(X, "values") else X)
                scores = validate_classification_score_values(
                    y_out,
                    context="dl explainability prediction",
                )
                proba = sigmoid(scores)
                return np.vstack([1.0 - proba, proba]).T

            def score(self, X, y):
                y_true = np.asarray(y).reshape(-1)
                if self.task_type == "classification":
                    y_true = ensure_binary_labels(pd.Series(y_true)).to_numpy(dtype=int)
                    proba = self.predict_proba(X)[:, 1]
                    auc = safe_auc(y_true, proba)
                    if auc is None:
                        pred = (proba >= 0.5).astype(int)
                        return float(accuracy_score(y_true, pred))
                    return float(auc)
                y_pred = predict_dl(self.model, X.values if hasattr(X, "values") else X)
                y_true, y_pred = validate_regression_metric_inputs(
                    y_true,
                    y_pred,
                    context="dl permutation importance scoring",
                )
                return float(r2_score(y_true, y_pred))

        wrapped_estimator = _SklearnWrapper(estimator, task_type=task_type)
        logging.info(
            "Permutation importance start: model=%s X_test=%s n_repeats=10 n_jobs=1 scoring=default",
            model_type,
            X_test.shape,
        )
        result = permutation_importance(
            wrapped_estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
        )
    else:
        scoring = "roc_auc" if task_type == "classification" else None
        logging.info(
            "Permutation importance start: model=%s X_test=%s n_repeats=10 n_jobs=%s scoring=%s",
            model_type,
            X_test.shape,
            n_jobs,
            scoring or "default",
        )
        result = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=n_jobs, scoring=scoring
        )
    logging.info("Permutation importance complete: model=%s", model_type)

    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance": result.importances_mean}
    ).sort_values(by="importance", ascending=False)
    importance_path = os.path.join(output_dir, f"{model_type}_permutation_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    logging.info("Permutation importance CSV written: %s", importance_path)

    plt.figure(figsize=(10, 6))
    importance_df.head(20).plot.bar(x="feature", y="importance")
    plt.tight_layout()
    importance_plot_path = os.path.join(output_dir, f"{model_type}_permutation_importance.png")
    plt.savefig(importance_plot_path)
    plt.close()
    logging.info("Permutation importance plot written: %s", importance_plot_path)

    try:
        if model_type in ["random_forest", "decision_tree", "xgboost", "catboost_classifier"]:
            logging.info("SHAP start: model=%s method=tree X_test=%s", model_type, X_test.shape)
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            if task_type == "classification" and isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]

        elif model_type == "ensemble":
            logging.info(
                "SHAP start: model=%s method=explainer background=%s X_test=%s",
                model_type,
                min(background_samples, len(X_test)),
                X_test.shape,
            )
            explainer = shap.Explainer(estimator.predict, X_test.iloc[:background_samples])
            shap_values = explainer(X_test)

        elif model_type == "svm":
            background = shap.sample(X_test, min(background_samples, len(X_test)))
            explainer = shap.KernelExplainer(estimator.predict, background)
            X_explain = X_test.iloc[: min(100, len(X_test))]
            logging.info(
                "SHAP start: model=%s method=kernel background=%s X_explain=%s",
                model_type,
                len(background),
                X_explain.shape,
            )
            shap_values = explainer.shap_values(X_explain)
            X_test = X_explain

        elif model_type.startswith("dl_"):
            device = get_device()
            estimator.to(device).eval()

            n_bg = min(background_samples, len(X_train))
            n_ex = min(100, len(X_test))

            X_bg_np = X_train.sample(n=n_bg, random_state=42).values.astype(np.float32)
            X_ex_np = X_test.iloc[:n_ex].values.astype(np.float32)
            logging.info(
                "SHAP start: model=%s method=kernel device=%s background=%s X_explain=%s",
                model_type,
                device,
                X_bg_np.shape,
                X_ex_np.shape,
            )

            def model_predict(X):
                out = predict_dl(estimator, X)
                out = np.asarray(out).reshape(-1)
                if task_type == "classification":
                    return out
                return out

            explainer = shap.KernelExplainer(model_predict, X_bg_np)
            shap_values = explainer.shap_values(X_ex_np, nsamples=100)

            X_test = X_test.iloc[:n_ex]

        logging.info("SHAP values computed: model=%s", model_type)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        shap_plot_path = os.path.join(output_dir, f"{model_type}_shap_summary.png")
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        logging.info("SHAP summary plot written: %s", shap_plot_path)
    except Exception as exc:
        logging.warning("SHAP explainability failed: %s", exc)
    logging.info("Explainability complete: model=%s", model_type)
