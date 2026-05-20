from __future__ import annotations

import os
from typing import Any, Callable


def load_model(
    model_path: str,
    model_type: str,
    input_dim: int | None = None,
    *,
    is_dl_model: Callable[[str], bool],
    initialize_model: Callable[..., Any],
    load_pickle: Callable[[str], Any],
) -> object:
    is_dl = is_dl_model(model_type)

    if is_dl:
        import torch

        if input_dim is None:
            raise ValueError("input_dim required to load DL models")

        params_path = model_path.replace("_best_model.pth", "_best_params.pkl")
        if os.path.exists(params_path):
            saved_params = load_pickle(params_path)
        else:
            saved_params = {}

        config = initialize_model(
            model_type,
            random_state=42,
            cv_folds=5,
            search_iters=100,
            input_dim=input_dim,
        )

        model_params = {**config.default_params, **saved_params}

        model = config.model_class(model_params)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    if model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    return load_pickle(model_path)
