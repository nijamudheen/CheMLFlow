from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

_TABULAR_MODEL_TYPES = {"random_forest", "svm", "decision_tree", "xgboost", "ensemble"}


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

    if model_type == "random_forest":
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {"random_state": random_state, **model_params}
                params.setdefault("n_jobs", n_jobs)
                return RandomForestClassifier(**params)
            base_rf_cls = RandomForestClassifier(random_state=random_state)
            return RandomizedSearchCV(
                estimator=base_rf_cls,
                param_distributions=param_dist,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {"random_state": random_state, **model_params}
            params.setdefault("n_jobs", n_jobs)
            return RandomForestRegressor(**params)
        base_rf = RandomForestRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_type == "svm":
        if is_classification:
            param_grid_svc = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.1, 0.01],
            }
            if tuning_method == "fixed":
                params = {"probability": True, **model_params}
                return SVC(**params)
            return GridSearchCV(
                estimator=SVC(kernel="rbf", probability=True),
                param_grid=param_grid_svc,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
            )
        param_grid_svm = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5],
        }
        if tuning_method == "fixed":
            return SVR(**model_params)
        return GridSearchCV(
            estimator=SVR(kernel="rbf"),
            param_grid=param_grid_svm,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
        )

    if model_type == "decision_tree":
        param_dist_dt = {
            "max_depth": [int(x) for x in np.linspace(5, 50, num=10)] + [None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {"random_state": random_state, **model_params}
                return DecisionTreeClassifier(**params)
            base_dt_cls = DecisionTreeClassifier(random_state=random_state)
            return RandomizedSearchCV(
                estimator=base_dt_cls,
                param_distributions=param_dist_dt,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {"random_state": random_state, **model_params}
            return DecisionTreeRegressor(**params)
        base_dt = DecisionTreeRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_dt,
            param_distributions=param_dist_dt,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_type == "xgboost":
        param_dist_xgb = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.1, 1, 10, 100],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    **model_params,
                }
                return XGBClassifier(**params)
            base_xgb_cls = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=n_jobs,
            )
            return RandomizedSearchCV(
                estimator=base_xgb_cls,
                param_distributions=param_dist_xgb,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {
                "objective": "reg:squarederror",
                "random_state": random_state,
                "n_jobs": n_jobs,
                **model_params,
            }
            return XGBRegressor(**params)
        base_xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist_xgb,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )

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
