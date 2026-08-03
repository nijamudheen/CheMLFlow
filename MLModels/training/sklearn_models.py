from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

_TABULAR_MODEL_TYPES = {
    "random_forest",
    "svm",
    "decision_tree",
    "xgboost",
    "ensemble",
    "tabpfn",
}
_PARENT_LEVEL_MODEL_SEARCH_MESSAGE = (
    "Runtime child-level hyperparameter search is disabled. Use DOE model_search "
    "to create parent-level fixed hyperparameter cases that fan out across CV folds."
)


def is_tabular_model(model_type: str) -> bool:
    return str(model_type).strip().lower() in _TABULAR_MODEL_TYPES


def build_tabular_model(
    *,
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    n_jobs: int,
    tuning_method: str,
    model_params: dict[str, Any] | None,
    task_type: str,
):
    model_type = str(model_type).strip().lower()
    tuning_method = str(tuning_method or "fixed").strip().lower()
    is_classification = str(task_type or "").strip().lower() == "classification"
    model_params = model_params or {}
    if tuning_method != "fixed":
        raise ValueError(
            f"Unsupported model.tuning.method={tuning_method!r}. "
            + _PARENT_LEVEL_MODEL_SEARCH_MESSAGE
        )
    _ = (cv_folds, search_iters)

    if model_type == "tabpfn":
        if not is_classification:
            raise ValueError("model.type=tabpfn currently supports classification tasks only.")
        params = dict(model_params)
        model_version = str(params.pop("model_version", "2.6")).strip().lower()
        if model_version not in {"2.6", "v2.6", "v2_6"}:
            raise ValueError(
                "CheMLFlow currently pins model.type=tabpfn to TabPFN 2.6; "
                f"got model_version={model_version!r}."
            )
        if str(params.get("fit_mode", "")).strip().lower() == "fit_with_cache":
            raise ValueError(
                "TabPFN fit_mode='fit_with_cache' cannot be persisted as a .tabpfn_fit artifact."
            )
        params.setdefault("random_state", random_state)
        params.setdefault("ignore_pretraining_limits", True)
        params.setdefault("n_estimators", 8)
        params.setdefault("n_preprocessing_jobs", max(1, int(n_jobs)))
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise ImportError(
                "model.type=tabpfn requires the optional TabPFN dependency. "
                "Install CheMLFlow with the tabpfn extra and accept access to the "
                "TabPFN 2.6 model weights before training."
            ) from exc
        version_2_6 = getattr(ModelVersion, "V2_6", None)
        if version_2_6 is None:
            raise ImportError(
                "This TabPFN installation does not expose model version 2.6. "
                "Install the CheMLFlow tabpfn extra (tabpfn==7.0.1)."
            )
        return TabPFNClassifier.create_default_for_version(
            version_2_6,
            **params,
        )

    if model_type == "random_forest":
        if is_classification:
            params = {"random_state": random_state, **model_params}
            params.setdefault("n_jobs", n_jobs)
            return RandomForestClassifier(**params)
        params = {"random_state": random_state, **model_params}
        params.setdefault("n_jobs", n_jobs)
        return RandomForestRegressor(**params)

    if model_type == "svm":
        if is_classification:
            params = {"probability": True, **model_params}
            return SVC(**params)
        return SVR(**model_params)

    if model_type == "decision_tree":
        if is_classification:
            params = {"random_state": random_state, **model_params}
            return DecisionTreeClassifier(**params)
        params = {"random_state": random_state, **model_params}
        return DecisionTreeRegressor(**params)

    if model_type == "xgboost":
        if is_classification:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": n_jobs,
                **model_params,
            }
            return XGBClassifier(**params)
        params = {
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": n_jobs,
            **model_params,
        }
        return XGBRegressor(**params)

    if model_type == "ensemble":
        ensemble_cfg = model_params if isinstance(model_params, dict) else {}
        rf_override = (
            ensemble_cfg.get("rf_params")
            if isinstance(ensemble_cfg.get("rf_params"), dict)
            else {}
        )
        xgb_override = (
            ensemble_cfg.get("xgb_params")
            if isinstance(ensemble_cfg.get("xgb_params"), dict)
            else {}
        )
        voting = str(ensemble_cfg.get("voting", "soft")).strip().lower() or "soft"

        if is_classification:
            rf_params = {
                "n_estimators": 200,
                "max_depth": 30,
                "max_features": "sqrt",
                "bootstrap": False,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "random_state": random_state,
                "n_jobs": n_jobs,
                **rf_override,
            }
            xgb_params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": n_jobs,
                **xgb_override,
            }
            rf = RandomForestClassifier(**rf_params)
            xgb = XGBClassifier(**xgb_params)
            return VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)],
                voting=voting,
                n_jobs=n_jobs,
            )

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            max_features="sqrt",
            bootstrap=False,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=random_state,
            **rf_override,
        )
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state,
            n_jobs=n_jobs,
            **xgb_override,
        )
        return VotingRegressor([("rf", rf), ("xgb", xgb)], n_jobs=n_jobs)

    raise ValueError(f"Unsupported tabular model type: {model_type}")
