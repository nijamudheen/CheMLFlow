from __future__ import annotations

import inspect
from typing import Any, Callable


def build_dl_search_config(
    *,
    model_type: str,
    input_dim: int | None,
    dl_search_config_cls: Callable[..., Any],
):
    model_type = str(model_type).strip().lower()
    if input_dim is None:
        raise ValueError("input_dim required for DL models")

    if model_type == "dl_simple":
        try:
            from DLModels.simplenn import SimpleNN
        except Exception:
            # Backward compatibility for repos that still expose the legacy module/class.
            from DLModels.simpleregressionnn import SimpleRegressionNN as SimpleNN

        simple_params = inspect.signature(SimpleNN).parameters

        def _make_simple_model(params: dict[str, Any]):
            kwargs = {
                "input_dim": input_dim,
                "hidden_dim": params.get("hidden_dim", 256),
            }
            if "use_tropical" in simple_params:
                kwargs["use_tropical"] = params.get("use_tropical", False)
            return SimpleNN(**kwargs)

        return dl_search_config_cls(
            model_class=_make_simple_model,
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                **(
                    {"use_tropical": {"type": "categorical", "choices": [True, False]}}
                    if "use_tropical" in simple_params
                    else {}
                ),
            },
            default_params={
                "hidden_dim": 256,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type == "dl_deep":
        from DLModels.deepregressionnn import DeepRegressionNN

        return dl_search_config_cls(
            model_class=lambda params: DeepRegressionNN(
                input_dim=input_dim,
                hidden_dims=[params.get("hidden_dim", 128)] * params.get("num_layers", 3),
                dropout_rate=params.get("dropout_rate", 0.2),
                use_tropical=params.get("use_tropical", False),
            ),
            search_space={
                "num_layers": {"type": "categorical", "choices": [2, 3, 4, 5]},
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                "use_tropical": {"type": "categorical", "choices": [True, False]},
            },
            default_params={
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout_rate": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type == "dl_gru":
        from DLModels.gruregressor import GRURegressor

        return dl_search_config_cls(
            model_class=lambda params: GRURegressor(
                seq_len=input_dim,
                input_size=params.get("input_size", 1),
                hidden_size=params.get("hidden_size", 128),
                num_layers=params.get("num_layers", 2),
                bidirectional=params.get("bidirectional", True),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "num_layers": {"type": "categorical", "choices": [1, 2]},
                "bidirectional": {"type": "categorical", "choices": [True, False]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "epochs": 200,
            },
        )

    if model_type == "dl_resmlp":
        from DLModels.resmlp import ResMLP

        return dl_search_config_cls(
            model_class=lambda params: ResMLP(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_dim", 512),
                n_blocks=params.get("n_blocks", 4),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [128, 256, 512, 1024]},
                "n_blocks": {"type": "categorical", "choices": [2, 3, 4, 6, 8]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "hidden_dim": 512,
                "n_blocks": 4,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type == "dl_tabtransformer":
        from DLModels.tabtransformer import TabTransformer

        return dl_search_config_cls(
            model_class=lambda params: TabTransformer(
                input_dim=input_dim,
                embed_dim=params.get("embed_dim", 128),
                n_heads=params.get("n_heads", 4),
                n_layers=params.get("n_layers", 2),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "embed_dim": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
                "n_layers": {"type": "categorical", "choices": [2, 3, 4]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.4, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "embed_dim": 128,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "epochs": 200,
            },
        )

    if model_type == "dl_aereg":
        from DLModels.aeregressor import AERegressor, Autoencoder

        return dl_search_config_cls(
            model_class=lambda params: AERegressor(
                pretrained_encoder=Autoencoder(
                    input_dim=input_dim,
                    bottleneck=params.get("bottleneck", 64),
                ).encoder,
                bottleneck=params.get("bottleneck", 64),
                dropout=params.get("dropout", 0.1),
            ),
            search_space={
                "bottleneck": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "bottleneck": 64,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type in {"dl_adaptive_nvar", "dl_connectome_nvar"}:
        raise ValueError(
            f"Model {model_type!r} is a time-series model and cannot be used "
            "with the standard tabular `train` node. "
            "Use the `train.timeseries` pipeline node and pipeline_type=timeseries; "
            "see config/timeseries_mg_demo.yaml. The matching search space lives "
            "in build_timeseries_dl_search_config(...) in this same module."
        )

    raise ValueError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------------
# Time-series DL search configs
# ---------------------------------------------------------------------------
#
# The tabular DL search above is reached via `MLModels.training.torch_models.
# run_optuna`, which assumes (input_dim -> 1) regression on a tabular X, y.
# Time-series models — Adaptive NVAR and Connectome NVAR — train differently
# (Adam->L-BFGS on residual targets, scored by autoregressive rollout), so
# they get a parallel entry point. The search space dict format is identical
# to the tabular one (`type` in {"categorical","float","int"}, etc.), and the
# time-series trainer's Optuna driver translates each spec to the same
# trial.suggest_* call. Keeping the format identical lets users reason about
# search spaces uniformly across all dl_* models.
#
# Each architecture gets its own registry entry so that:
#   1. Adaptive NVAR (MLP feature block) and Connectome NVAR (fixed-graph
#      feature block) have independent search spaces — they don't share the
#      hidden_dim vs n_connectome axis, the input_scaling axis, etc.
#   2. New time-series models can be added without disturbing existing ones.


def build_timeseries_dl_search_config(
    *,
    model_type: str,
    dl_search_config_cls: Callable[..., Any],
):
    """Return a search_config for a time-series DL model.

    Parameters
    ----------
    model_type:
        One of `dl_adaptive_nvar`, `dl_connectome_nvar`.
    dl_search_config_cls:
        The same DLSearchConfig dataclass used by the tabular path. We pass a
        placeholder model_class because building an actual nn.Module requires
        runtime data (delay-embedding dimension `dk = d * k`, connectome
        adjacency, etc.) that only exists inside the trainer. The time-series
        Optuna driver constructs models directly from sampled params rather
        than going through `config.model_class`.

    Returns
    -------
    A DLSearchConfig with `search_space` and `default_params` populated.
    The `model_class` is set to a no-op factory; callers should consult
    `default_params` and `search_space` and instantiate the model themselves.
    """
    model_type = str(model_type).strip().lower()

    def _unused_model_class(params: dict[str, Any]):  # pragma: no cover - placeholder
        raise RuntimeError(
            "Time-series DL search configs do not build models via "
            "config.model_class; the time-series trainer constructs models "
            "directly from sampled hyperparameters. See "
            "MLModels.training.timeseries_nvar.run_optuna_timeseries."
        )

    if model_type == "dl_adaptive_nvar":
        # Mirrors `Optuna_MG_Adaptive_NVAR_0percent_noise.ipynb` (notebook 2).
        # Architectural axes:
        #   - k:          delay-embedding length. Small k underfits chaotic
        #                 dynamics; large k inflates dk = d*k.
        #   - hidden_dim: MLP feature width. Bigger -> more capacity but
        #                 longer L-BFGS phase.
        # Optimization axes:
        #   - lr_adam:    log-scale; Adam pre-trains the MLP before L-BFGS.
        #   - lr_lbfgs:   linear scale, narrow band; full L-BFGS step size.
        return dl_search_config_cls(
            model_class=_unused_model_class,
            search_space={
                # Notebook-faithful values from
                # Optuna_MG_Adaptive_NVAR_0percent_noise.ipynb:
                #   k_grid         = [2, 10, 30, 50]
                #   hidden_dim_grid= [10, 20, 50, 100, 200, 500, 1000]
                #   adam_lr_grid   = [1e-4, 1e-3, 1e-2]
                #   lbfgs_lr_grid  = [1.0, 0.5, 0.1]
                "k": {"type": "categorical", "choices": [2, 10, 30, 50]},
                "hidden_dim": {
                    "type": "categorical",
                    "choices": [10, 20, 50, 100, 200, 500, 1000],
                },
                "lr_adam": {
                    "type": "categorical",
                    "choices": [1e-4, 1e-3, 1e-2],
                },
                "lr_lbfgs": {"type": "categorical", "choices": [1.0, 0.5, 0.1]},
            },
            default_params={
                "k": 5,
                "hidden_dim": 200,
                "lr_adam": 1e-3,
                "lr_lbfgs": 1.0,
                # The remaining knobs are not searched by default; they come
                # from train.model.params in the runtime config.
                "max_epochs_adam": 5000,
                "adam_patience": 200,
                "num_epochs_lbfgs": 50000,
                "lbfgs_patience": 200,
                "weight_decay": 0.0,
                "train_noise_scale": 0.05,
                "dataset_noise_scale": 0.0,
                "horizons": [25, 50, 75, 100],
                "num_windows": 10,
                "optuna_num_runs": 5,
            },
        )

    if model_type == "dl_connectome_nvar":
        # Mirrors `Adaptive_NVAR_optuna_connectome.ipynb` (notebook 1).
        # Architectural axes specific to the connectome variant:
        #   - n_connectome:    subgraph size selected from the full adjacency.
        #                      Decoupled from `hidden_dim` (which AdaptiveNVAR
        #                      uses) on purpose — they are not interchangeable.
        #   - input_scaling:   initial scale of the input projection weights;
        #                      the connectome layer is sensitive to this.
        #   - weight_decay:    log-scale, includes 0 via the "0.0" choice.
        return dl_search_config_cls(
            model_class=_unused_model_class,
            search_space={
                "k": {"type": "categorical", "choices": [2, 5, 10, 20, 30, 40, 50]},
                "n_connectome": {
                    "type": "categorical",
                    "choices": [25, 50, 75, 100, 150, 200, 250, 300],
                },
                "input_scaling": {
                    "type": "categorical",
                    "choices": [0.02, 0.05, 0.10, 0.20, 0.50],
                },
                "lr_adam": {
                    "type": "categorical",
                    "choices": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                },
                "lr_lbfgs": {"type": "categorical", "choices": [1.0, 0.5, 0.1]},
                "weight_decay": {
                    "type": "categorical",
                    "choices": [0.0, 1e-8, 1e-6, 1e-4],
                },
            },
            default_params={
                "k": 5,
                "n_connectome": 100,
                "input_scaling": 0.10,
                "lr_adam": 1e-3,
                "lr_lbfgs": 1.0,
                "weight_decay": 0.0,
                "max_epochs_adam": 5000,
                "adam_patience": 200,
                "num_epochs_lbfgs": 50000,
                "lbfgs_patience": 200,
                "train_noise_scale": 0.05,
                "dataset_noise_scale": 0.0,
                "horizons": [25, 50, 75, 100],
                "num_windows": 10,
                # Connectome-specific defaults; these are not searched. To vary
                # them across cases, put them in DOE search_space (e.g.
                # connectome_mode: [connectome, connectome_randomized]).
                "connectome_mode": "connectome",
                "connectome_selection_mode": "top_degree",
                "connectome_normalization": "maxabs",
                "connectome_binarize": False,
            },
        )

    raise ValueError(
        f"Unsupported time-series model_type: {model_type!r}. "
        "Expected one of: dl_adaptive_nvar, dl_connectome_nvar."
    )
