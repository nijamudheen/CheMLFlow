from __future__ import annotations

from typing import Any, Callable

from . import dl_registry, sklearn_models


def is_dl_model(model_type: str) -> bool:
    return str(model_type).strip().lower().startswith("dl_")


def initialize_model(
    *,
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    input_dim: int | None = None,
    n_jobs: int = -1,
    tuning_method: str = "fixed",
    model_params: dict[str, Any] | None = None,
    task_type: str = "regression",
    dl_search_config_cls: Callable[..., Any],
):
    tuning_method = str(tuning_method or "fixed").strip().lower()
    if tuning_method not in {"train_cv", "fixed"}:
        raise ValueError(f"Unsupported model.tuning.method={tuning_method!r}; expected 'train_cv' or 'fixed'.")

    if sklearn_models.is_tabular_model(model_type):
        return sklearn_models.build_tabular_model(
            model_type=model_type,
            random_state=random_state,
            cv_folds=cv_folds,
            search_iters=search_iters,
            n_jobs=n_jobs,
            tuning_method=tuning_method,
            model_params=model_params,
            task_type=task_type,
        )

    if is_dl_model(model_type):
        return dl_registry.build_dl_search_config(
            model_type=model_type,
            input_dim=input_dim,
            dl_search_config_cls=dl_search_config_cls,
        )

    raise ValueError(f"Unsupported model type: {model_type}")
