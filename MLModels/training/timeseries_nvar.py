"""
Time-series Adaptive NVAR trainer for CheMLFlow.

This module owns the `train.timeseries` pipeline node's contract:

  inputs:
    * a raw [T, d] series (loaded from the canonical .npz)
    * a runtime config block describing model, training, and rollout
  outputs (in the per-case run_dir):
    * <model_type>_metrics.json          — primary scalar + diagnostics
    * <model_type>_split_metrics.json    — train/val/test horizon RMSEs
    * <model_type>_rollout_per_window_per_horizon.csv  — rich table
    * <model_type>_predictions.npz       — per-window predictions/truth/noisy
    * <model_type>_best_model.pth        — torch state_dict
    * <model_type>_best_params.pkl       — joblib-serialized hyperparams

The two-phase Adam → L-BFGS training and the windowed autoregressive rollout
are ported from the user's notebook reference implementation, but parameterized
so the trainer is dataset-agnostic. The Optuna search lives in the dl_registry
machinery; this module is invoked once per (case, fold) by the training node.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import joblib
import numpy as np

from utilities import connectome_loader, timeseries_io


LOGGER = logging.getLogger(__name__)

# Public model-type strings recognized by this trainer. Both start with `dl_`
# so the existing DOE/profile machinery accepts them automatically.
ADAPTIVE_NVAR = "dl_adaptive_nvar"
CONNECTOME_NVAR = "dl_connectome_nvar"
SUPPORTED_MODEL_TYPES = (ADAPTIVE_NVAR, CONNECTOME_NVAR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Resolved hyperparameters for a single train+rollout call."""

    # Architectural
    k: int = 5                       # delay-embedding length
    hidden_dim: int = 200            # MLP hidden (AdaptiveNVAR only)
    feature_dim_m: Optional[int] = None  # output of MLP feature block; defaults to dk*(dk+1)/2
    n_connectome: int = 100          # connectome subgraph size (Connectome only)
    input_scaling: float = 0.10      # input projection init scale

    # Training (Adam phase)
    lr_adam: float = 1e-3
    max_epochs_adam: int = 5000
    adam_patience: int = 200
    weight_decay: float = 0.0

    # Training (L-BFGS phase)
    lr_lbfgs: float = 1.0
    num_epochs_lbfgs: int = 50000
    lbfgs_patience: int = 200
    lbfgs_max_iter: int = 50
    lbfgs_history_size: int = 50
    tolerance: float = 1e-20

    # Noise
    train_noise_scale: float = 0.05  # added to training segment only
    dataset_noise_scale: float = 0.0 # added globally (then val_true is the clean series)

    # Rollout
    horizons: tuple[int, ...] = (25, 50, 75, 100)
    num_windows: int = 10            # windows per evaluation segment

    # Repeated stochastic runs
    optuna_num_runs: int = 5          # notebook-compatible repeated validation runs per Optuna trial
    test_num_runs: int = 25           # final independent test trainings after Optuna/fixed params

    # Device selection. Values:
    #   "auto" - use CUDA if torch.cuda.is_available(), else CPU. No raise.
    #   "cuda" - REQUIRE CUDA. Raise RuntimeError if torch.cuda.is_available()
    #            is False or device_count is 0. This is the recommended HPCC
    #            setting: it converts silent CPU fallback (which can waste
    #            hours of wall-clock) into a fast, loud failure.
    #   "cpu"  - force CPU even if CUDA is available. Useful for debugging.
    device: str = "auto"

    # Reproducibility
    base_seed: int = 2025

    # Connectome-specific
    connectome_xlsx: Optional[str] = None
    connectome_sheet: Optional[str] = None
    connectome_mode: str = "connectome"  # "connectome" or "connectome_randomized"
    connectome_selection_mode: str = "top_degree"  # or "random"
    connectome_selection_seed: int = 2025
    connectome_binarize: bool = False
    connectome_normalization: str = "maxabs"
    connectome_swap_factor: int = 10


def parse_training_config(
    *,
    model_type: str,
    model_params: dict[str, Any],
    train_block: dict[str, Any],
    global_random_state: int,
) -> TrainingConfig:
    """Merge model.params + train.timeseries.* into a TrainingConfig.

    The split-len keys (warmup_len/train_len/val_len/test_len) live in a
    separate `split` block parsed by `timeseries_io.parse_split_config`.
    """
    cfg = TrainingConfig()

    # base_seed comes from global random_state unless explicitly overridden.
    cfg.base_seed = int(model_params.get("base_seed", global_random_state))

    # Device knob. Accepts "auto"|"cuda"|"cpu" (case-insensitive). Anything
    # else is rejected loudly so a typo in YAML doesn't silently fall back.
    if "device" in model_params:
        dev = str(model_params["device"]).strip().lower()
        if dev not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                f"train.model.params.device must be one of "
                f"'auto', 'cuda', 'cpu'; got {model_params['device']!r}."
            )
        cfg.device = dev

    # Architectural ----------------------------------------------------------
    if "k" in model_params:
        cfg.k = int(model_params["k"])
    if "hidden_dim" in model_params:
        cfg.hidden_dim = int(model_params["hidden_dim"])
    if "feature_dim_m" in model_params and model_params["feature_dim_m"] is not None:
        cfg.feature_dim_m = int(model_params["feature_dim_m"])
    if "n_connectome" in model_params:
        cfg.n_connectome = int(model_params["n_connectome"])
    if "input_scaling" in model_params:
        cfg.input_scaling = float(model_params["input_scaling"])

    # Training ---------------------------------------------------------------
    for key in (
        "lr_adam",
        "lr_lbfgs",
        "weight_decay",
        "tolerance",
        "train_noise_scale",
        "dataset_noise_scale",
    ):
        if key in model_params:
            setattr(cfg, key, float(model_params[key]))
    for key in (
        "max_epochs_adam",
        "adam_patience",
        "num_epochs_lbfgs",
        "lbfgs_patience",
        "lbfgs_max_iter",
        "lbfgs_history_size",
        "optuna_num_runs",
        "test_num_runs",
    ):
        if key in model_params:
            setattr(cfg, key, int(model_params[key]))

    # Rollout ----------------------------------------------------------------
    horizons = model_params.get("horizons")
    if horizons is None:
        # Allow horizons to live under the train.timeseries block too.
        ts_block = train_block.get("timeseries", {}) if isinstance(train_block, dict) else {}
        horizons = ts_block.get("horizons")
    if horizons is not None:
        cfg.horizons = tuple(int(h) for h in horizons if int(h) > 0)
        if not cfg.horizons:
            raise ValueError("horizons must be a non-empty list of positive ints.")
    if "num_windows" in model_params:
        cfg.num_windows = int(model_params["num_windows"])

    # Connectome -------------------------------------------------------------
    if model_type == CONNECTOME_NVAR:
        if "connectome_xlsx" not in model_params or not model_params["connectome_xlsx"]:
            raise ValueError(
                f"{CONNECTOME_NVAR} requires train.model.params.connectome_xlsx (path to .xlsx)."
            )
        cfg.connectome_xlsx = str(model_params["connectome_xlsx"])
        cfg.connectome_sheet = model_params.get("connectome_sheet")
        if "connectome_mode" in model_params:
            mode = str(model_params["connectome_mode"]).strip().lower()
            if mode not in {"connectome", "connectome_randomized"}:
                raise ValueError(
                    "connectome_mode must be 'connectome' or 'connectome_randomized'."
                )
            cfg.connectome_mode = mode
        if "connectome_selection_mode" in model_params:
            sel = str(model_params["connectome_selection_mode"]).strip().lower()
            if sel not in {"top_degree", "random"}:
                raise ValueError(
                    "connectome_selection_mode must be 'top_degree' or 'random'."
                )
            cfg.connectome_selection_mode = sel
        if "connectome_selection_seed" in model_params:
            cfg.connectome_selection_seed = int(model_params["connectome_selection_seed"])
        if "connectome_binarize" in model_params:
            cfg.connectome_binarize = bool(model_params["connectome_binarize"])
        if "connectome_normalization" in model_params:
            norm = str(model_params["connectome_normalization"]).strip().lower()
            if norm not in {"none", "maxabs", "spectral"}:
                raise ValueError(
                    "connectome_normalization must be 'none', 'maxabs', or 'spectral'."
                )
            cfg.connectome_normalization = norm
        if "connectome_swap_factor" in model_params:
            cfg.connectome_swap_factor = int(model_params["connectome_swap_factor"])

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection (strict + diagnostic)
# ---------------------------------------------------------------------------
#
# `train.model.params.device` (auto|cuda|cpu) controls how the trainer selects
# a torch device. The default `auto` matches the older behavior; `cuda` is
# strict — it raises if CUDA isn't actually available, converting a silent
# 5-hour-on-CPU surprise into a fast, loud failure. The first call also
# emits a one-time diagnostic so misconfigurations are visible in the log
# (PyTorch CUDA build, CUDA_VISIBLE_DEVICES env, device_count, chosen device).
#
# We deliberately keep the diagnostic in the trainer rather than in
# run_node_train_timeseries so it always fires together with the actual
# device-selection decision, even in standalone (non-CheMLFlow-pipeline)
# uses of train_timeseries_nvar.

_DEVICE_DIAGNOSTIC_EMITTED = False


def _log_device_diagnostics(requested: str, resolved) -> None:
    """One-shot dump of torch CUDA visibility info. Fires once per process."""
    global _DEVICE_DIAGNOSTIC_EMITTED
    if _DEVICE_DIAGNOSTIC_EMITTED:
        return
    _DEVICE_DIAGNOSTIC_EMITTED = True

    import torch

    LOGGER.info("=" * 60)
    LOGGER.info("Device diagnostic (emitted once per process):")
    LOGGER.info("  torch.__version__         = %s", torch.__version__)
    LOGGER.info("  torch.version.cuda        = %s", getattr(torch.version, "cuda", None))
    LOGGER.info("  torch.cuda.is_available() = %s", torch.cuda.is_available())
    try:
        LOGGER.info("  torch.cuda.device_count() = %d", torch.cuda.device_count())
    except Exception as exc:
        LOGGER.info("  torch.cuda.device_count() raised: %s", exc)
    LOGGER.info(
        "  CUDA_VISIBLE_DEVICES env  = %s",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    LOGGER.info("  requested device          = %s", requested)
    LOGGER.info("  resolved device           = %s", resolved)
    LOGGER.info("=" * 60)


def _resolve_device(requested: str):
    """Return a torch.device honoring `requested` (auto|cuda|cpu).

    - "cuda": requires torch.cuda.is_available() and device_count >= 1, else raises.
    - "cpu":  always returns torch.device("cpu").
    - "auto": CUDA if available, else CPU. No raise.

    Also emits a one-time process-wide diagnostic via _log_device_diagnostics.
    """
    import torch

    req = str(requested or "auto").strip().lower()
    if req == "cpu":
        device = torch.device("cpu")
    elif req == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            _log_device_diagnostics(req, "REQUESTED CUDA BUT UNAVAILABLE")
            raise RuntimeError(
                "train.model.params.device='cuda' was requested but no CUDA "
                "device is visible to PyTorch. See the device diagnostic "
                "above. Common causes: (1) PyTorch was installed without "
                "CUDA support — `python -c 'import torch; print(torch.version.cuda)'` "
                "shows None; (2) Slurm did not allocate a GPU — `nvidia-smi` "
                "shows no devices, or CUDA_VISIBLE_DEVICES is empty; (3) the "
                "job ran on a CPU-only node. Re-submit with --gpus=1 and a "
                "CUDA-enabled PyTorch in the env, or set device: cpu/auto to "
                "fall back."
            )
        device = torch.device("cuda")
    elif req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # pragma: no cover - parse_training_config already validates
        raise ValueError(f"unknown device {requested!r}")

    _log_device_diagnostics(req, device)
    return device


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_model(
    *,
    model_type: str,
    cfg: TrainingConfig,
    d: int,
    connectome_adj: Optional[np.ndarray],
):
    from DLModels.adaptive_nvar import (
        AdaptiveConnectomeNVARModel,
        AdaptiveNVARModel,
        init_weights_stable,
    )

    dk = int(d * cfg.k)
    if model_type == ADAPTIVE_NVAR:
        m = cfg.feature_dim_m if cfg.feature_dim_m is not None else int(dk * (dk + 1) // 2)
        model = AdaptiveNVARModel(dk=dk, m=m, d=d, hidden_dim=cfg.hidden_dim)
        model.apply(init_weights_stable)
        return model
    if model_type == CONNECTOME_NVAR:
        if connectome_adj is None:
            raise ValueError(f"{CONNECTOME_NVAR} requires a connectome_adj matrix.")
        return AdaptiveConnectomeNVARModel(
            dk=dk,
            d=d,
            n_connectome=int(connectome_adj.shape[0]),
            connectome_adj=connectome_adj,
            input_scale=cfg.input_scaling,
        )
    raise ValueError(f"Unsupported model_type for timeseries trainer: {model_type!r}")


# ---------------------------------------------------------------------------
# Training: Adam -> L-BFGS
# ---------------------------------------------------------------------------


def _train_adaptive_nvar(
    *,
    model,
    X_train,
    cfg: TrainingConfig,
    save_path: str,
    device,
):
    """Two-phase training: Adam with early stopping, then L-BFGS refinement."""
    import torch
    import torch.nn.functional as F

    from DLModels.adaptive_nvar import construct_H_lin

    model = model.to(device)
    X_train = X_train.to(device)
    H_lin = construct_H_lin(X_train, cfg.k)
    Y = X_train[cfg.k :] - X_train[cfg.k - 1 : -1]

    # ---- Phase 1: Adam ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr_adam, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )
    best_loss = float("inf")
    epochs_no_improve = 0
    adam_epochs_run = 0

    for epoch in range(cfg.max_epochs_adam):
        model.train()
        Y_hat = model(H_lin)
        loss = F.mse_loss(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        adam_epochs_run = epoch + 1
        if best_loss - loss.item() > cfg.tolerance:
            best_loss = float(loss.item())
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg.adam_patience:
            break

    # If no checkpoint was written (rare but possible), persist current state.
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device))
    adam_best_loss = best_loss

    # ---- Phase 2: L-BFGS ----
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.lr_lbfgs,
        max_iter=cfg.lbfgs_max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    best_loss = float("inf")
    epochs_no_improve = 0
    lbfgs_epochs_run = 0

    def closure():
        optimizer.zero_grad()
        Y_hat = model(H_lin)
        loss = F.mse_loss(Y_hat, Y)
        loss.backward()
        return loss

    for epoch in range(cfg.num_epochs_lbfgs):
        model.train()
        loss = optimizer.step(closure)
        current_loss = float(loss.item()) if loss is not None else float("inf")
        lbfgs_epochs_run = epoch + 1
        if best_loss - current_loss > cfg.tolerance:
            best_loss = current_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg.lbfgs_patience:
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, {
        "adam_epochs_run": int(adam_epochs_run),
        "adam_best_loss": float(adam_best_loss),
        "lbfgs_epochs_run": int(lbfgs_epochs_run),
        "lbfgs_best_loss": float(best_loss),
    }


# ---------------------------------------------------------------------------
# Windowed autoregressive rollout + multi-horizon RMSE
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _rollout_segment(
    *,
    model,
    X_full,                # full noisy series, torch tensor [T_full, d]
    seg_offset: int,       # starting index of this segment in X_full
    seg_true,              # numpy array [seg_len, d] — clean truth
    seg_noisy,             # numpy array [seg_len, d] — noisy observed
    cfg: TrainingConfig,
    device,
):
    """Run cfg.num_windows non-overlapping rollouts over a contiguous segment.

    Returns:
        per_window: list of dicts with keys
            window_index, start, end, h{H}_rmse for each H in cfg.horizons
        concat_pred:  np.ndarray [num_windows*window_size, d]
        concat_true:  np.ndarray [num_windows*window_size, d]
        concat_noisy: np.ndarray [num_windows*window_size, d]
    """
    import torch

    model.eval()
    seg_len = int(seg_true.shape[0])
    if cfg.num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    window_size = seg_len // cfg.num_windows
    if window_size <= 0:
        raise ValueError(
            f"Segment of length {seg_len} cannot be divided into "
            f"{cfg.num_windows} non-empty windows."
        )

    per_window: list[dict[str, Any]] = []
    all_preds, all_true, all_noisy = [], [], []

    for w in range(cfg.num_windows):
        start = w * window_size
        end = start + window_size
        true_window = seg_true[start:end]
        noisy_window = seg_noisy[start:end]

        # Initial context: the k timesteps right before the window.
        init_start = seg_offset + start - cfg.k
        init_end = seg_offset + start
        if init_start < 0:
            raise ValueError(
                "Insufficient history before evaluation segment for delay-embedding "
                f"k={cfg.k}; need offset >= k, got offset={seg_offset}, start={start}."
            )
        X_init = X_full[init_start:init_end].to(device)
        if X_init.shape[0] != cfg.k:
            raise ValueError(
                f"Context window mismatch: got {X_init.shape[0]} rows, expected k={cfg.k}."
            )

        x_t = [x.clone() for x in X_init.unbind(0)]
        H_lin = torch.cat(x_t, dim=-1).unsqueeze(0)
        predictions = []
        for _ in range(window_size):
            with torch.no_grad():
                delta_x = model(H_lin)
            x_next = x_t[-1] + delta_x.squeeze(0)
            predictions.append(x_next)
            x_t = x_t[1:] + [x_next]
            H_lin = torch.cat(x_t, dim=-1).unsqueeze(0)

        pred_np = torch.stack(predictions).cpu().numpy()
        all_preds.append(pred_np)
        all_true.append(true_window)
        all_noisy.append(noisy_window)

        row: dict[str, Any] = {
            "window_index": int(w),
            "start": int(start),
            "end": int(end),
            "window_size": int(window_size),
        }
        for h in cfg.horizons:
            if h <= window_size:
                row[f"rmse_h{h}"] = _rmse(true_window[:h], pred_np[:h])
            else:
                row[f"rmse_h{h}"] = None
        per_window.append(row)

    concat_pred = np.concatenate(all_preds, axis=0)
    concat_true = np.concatenate(all_true, axis=0)
    concat_noisy = np.concatenate(all_noisy, axis=0)
    return per_window, concat_pred, concat_true, concat_noisy


def _aggregate_horizon_rmse(
    per_window: list[dict[str, Any]], horizons: tuple[int, ...]
) -> dict[str, float]:
    """Mean RMSE per horizon across all windows that produced a value."""
    out: dict[str, float] = {}
    for h in horizons:
        key = f"rmse_h{h}"
        vals = [row[key] for row in per_window if row.get(key) is not None]
        if vals:
            out[key] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _add_relative_gaussian_noise(X_clean: np.ndarray, noise_scale: float) -> np.ndarray:
    if noise_scale <= 0:
        return X_clean.astype(np.float32, copy=True)
    signal_std = np.std(X_clean, axis=0, keepdims=True)
    noise = np.random.normal(0.0, noise_scale * signal_std, X_clean.shape)
    return (X_clean + noise).astype(np.float32, copy=False)


def _torchify(arr: np.ndarray, device, dtype="float32"):
    import torch

    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@dataclass
class TimeSeriesTrainResult:
    """Lightweight echo of MLModels.train_models.TrainResult — same field names."""

    model_path: str
    params_path: str
    metrics_path: str


# ---------------------------------------------------------------------------
# Optuna search (in-loop, per case)
# ---------------------------------------------------------------------------
#
# Each trial:
#   1. Sample a params dict from the search space defined in dl_registry.
#   2. Build a fresh TrainingConfig by merging:
#         dl_registry.default_params  <- baseline
#         user's YAML model_params    <- per-case overrides (DOE)
#         sampled trial params        <- this trial's choices
#   3. Train Adam->L-BFGS on the train segment.
#   4. Run windowed rollout on val.
#   5. Return val RMSE at the largest horizon.
#
# The DOE varies scientific factors (noise, connectome_mode, etc.) across cases.
# Within each case, this Optuna loop varies architectural/optimization knobs
# (k, hidden_dim, lr_adam, ...). Those two layers stay strictly separated.


def _trial_seed(base_seed: int, trial_number: int) -> int:
    return int(base_seed) + int(trial_number) + 1


def _train_and_score_val(
    *,
    model_type: str,
    cfg: TrainingConfig,
    sliced_clean: timeseries_io.TimeSeriesSplit,
    sliced_noisy: timeseries_io.TimeSeriesSplit,
    data_noisy: np.ndarray,
    split_cfg: timeseries_io.TimeSeriesSplitConfig,
    connectome_adj: Optional[np.ndarray],
    seed: int,
    tmp_save_path: str,
    num_runs: int = 1,
) -> Optional[float]:
    """Train once, run val rollout, return val RMSE at largest horizon.

    Returns None on infeasible configs (e.g. k too large for the segment).
    The temporary checkpoint at `tmp_save_path` is overwritten on each call.
    """
    import torch

    d = int(sliced_clean.d)
    device = _resolve_device(cfg.device)
    primary_h = max(cfg.horizons)
    primary_key = f"rmse_h{primary_h}"

    # Match the notebook protocol: repeat each Optuna trial over several
    # stochastic initializations, but keep the training-time noise realization
    # fixed for the case. In the notebook, X_train is created once before
    # Optuna and only torch seeds change inside the run loop.
    np.random.seed(int(cfg.base_seed) + 1)
    train_with_noise = _add_relative_gaussian_noise(sliced_noisy.train, cfg.train_noise_scale)

    scores: list[float] = []
    for run in range(max(1, int(num_runs))):
        run_seed = int(seed) + int(run)
        _seed_everything(run_seed)
        run_save_path = (
            tmp_save_path
            if int(num_runs) <= 1
            else f"{os.path.splitext(tmp_save_path)[0]}_run{run:02d}.pth"
        )

        try:
            model = _build_model(
                model_type=model_type,
                cfg=cfg,
                d=d,
                connectome_adj=connectome_adj,
            )
            X_train_t = _torchify(train_with_noise, device=device)
            model, _ = _train_adaptive_nvar(
                model=model,
                X_train=X_train_t,
                cfg=cfg,
                save_path=run_save_path,
                device=device,
            )
            X_full_t = _torchify(data_noisy, device=device)
            val_offset = split_cfg.warmup_len + split_cfg.train_len
            val_per_window, _, _, _ = _rollout_segment(
                model=model,
                X_full=X_full_t,
                seg_offset=val_offset,
                seg_true=sliced_clean.val,
                seg_noisy=sliced_noisy.val,
                cfg=cfg,
                device=device,
            )
            val_rmse_map = _aggregate_horizon_rmse(val_per_window, cfg.horizons)
            score = val_rmse_map.get(primary_key)
            if score is not None and np.isfinite(score):
                scores.append(float(score))
        except ValueError as exc:
            # Misconfigured trial (e.g. k > train_len). Return None -> Optuna prunes.
            LOGGER.warning("Optuna trial pruned (build/train): %s", exc)
            return None
        except Exception as exc:
            LOGGER.warning("Optuna trial errored (build/train): %s", exc)
            return None

    if not scores:
        return None
    return float(np.mean(scores))


def _run_optuna_timeseries(
    *,
    model_type: str,
    raw_path: str,
    user_model_params: dict[str, Any],
    train_block: dict[str, Any],
    split_block: dict[str, Any],
    tuning_block: dict[str, Any],
    global_random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run an in-loop Optuna search; return (best_params, study_summary).

    `best_params` contains only the searched axes. Callers merge it into the
    user's model_params and re-run the standard fixed train to produce the
    final on-disk artifacts.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "tuning.method=optuna requires the optuna package. "
            "Install with `pip install optuna`."
        ) from exc

    # Lazy import to avoid circulars; dl_registry imports nothing from this module.
    from MLModels.train_models import DLSearchConfig
    from MLModels.training.dl_registry import build_timeseries_dl_search_config

    search_cfg = build_timeseries_dl_search_config(
        model_type=model_type,
        dl_search_config_cls=DLSearchConfig,
    )
    search_space: dict[str, dict[str, Any]] = dict(search_cfg.search_space)
    registry_defaults: dict[str, Any] = dict(search_cfg.default_params)

    # Tuning controls ---------------------------------------------------------
    n_trials = int(tuning_block.get("n_trials", 30))
    timeout = tuning_block.get("timeout", None)
    timeout = float(timeout) if timeout is not None else None

    # Optional knob: shrink the workload per trial so a 30-trial sweep finishes
    # in reasonable wall-time. If the user sets `tuning.trial_epoch_cap`, every
    # trial uses min(cap, cfg.max_epochs_adam) for both phases. Final retrain
    # (run after the search by the caller) uses the full budget.
    trial_epoch_cap_raw = tuning_block.get("trial_epoch_cap")
    trial_epoch_cap = int(trial_epoch_cap_raw) if trial_epoch_cap_raw is not None else None

    # Pre-load data + (optionally) connectome once for the whole search.
    data_clean, _raw_meta = timeseries_io.load_raw_timeseries(raw_path)
    split_cfg = timeseries_io.parse_split_config(split_block)
    np.random.seed(int(global_random_state))
    dataset_noise_scale = float(user_model_params.get("dataset_noise_scale", 0.0))
    data_noisy = _add_relative_gaussian_noise(data_clean, dataset_noise_scale)
    sliced_clean = timeseries_io.slice_time_series(data_clean, split_cfg)
    sliced_noisy = timeseries_io.slice_time_series(data_noisy, split_cfg)

    connectome_adj: Optional[np.ndarray] = None
    bundle = None
    if model_type == CONNECTOME_NVAR:
        # n_connectome may be in the search space; we'll rebuild the bundle
        # per trial only when it actually changes.
        pass  # built inside the objective

    # Optuna driver -----------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=int(global_random_state))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Suppress Optuna's per-trial INFO chatter unless the user opted in.
    if str(tuning_block.get("verbose", "false")).strip().lower() not in {"true", "1", "yes", "on"}:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    last_bundle_cache_key: Optional[tuple] = None
    last_bundle: Optional[connectome_loader.ConnectomeBundle] = None
    last_bundle_lock: dict[str, Any] = {}

    def _resolve_connectome_for_trial(trial_params: dict[str, Any]) -> Optional[np.ndarray]:
        nonlocal last_bundle_cache_key, last_bundle
        if model_type != CONNECTOME_NVAR:
            return None
        # Effective params for connectome bundle construction come from user
        # model_params + registry defaults + this trial's sampled overrides.
        effective = {**registry_defaults, **user_model_params, **trial_params}
        if "connectome_xlsx" not in effective or not effective["connectome_xlsx"]:
            raise ValueError(
                f"{CONNECTOME_NVAR} requires train.model.params.connectome_xlsx."
            )
        key = (
            effective["connectome_xlsx"],
            effective.get("connectome_sheet"),
            int(effective.get("n_connectome", 100)),
            effective.get("connectome_selection_mode", "top_degree"),
            int(effective.get("connectome_selection_seed", 2025)),
            bool(effective.get("connectome_binarize", False)),
            effective.get("connectome_mode", "connectome"),
            int(effective.get("connectome_swap_factor", 10)),
            int(effective.get("base_seed", global_random_state)),
            str(effective.get("connectome_normalization", "maxabs")),
        )
        if key == last_bundle_cache_key and last_bundle is not None:
            return last_bundle.adjacency
        last_bundle = connectome_loader.build_connectome(
            xlsx_path=str(effective["connectome_xlsx"]),
            sheet_name=effective.get("connectome_sheet"),
            n_select=int(effective.get("n_connectome", 100)),
            selection_mode=str(effective.get("connectome_selection_mode", "top_degree")),
            selection_seed=int(effective.get("connectome_selection_seed", 2025)),
            binarize=bool(effective.get("connectome_binarize", False)),
            randomize=(str(effective.get("connectome_mode", "connectome")) == "connectome_randomized"),
            randomize_swap_factor=int(effective.get("connectome_swap_factor", 10)),
            randomize_seed=int(effective.get("base_seed", global_random_state)),
            normalization=str(effective.get("connectome_normalization", "maxabs")),
        )
        last_bundle_cache_key = key
        return last_bundle.adjacency

    tmp_dir = tempfile.mkdtemp(prefix="timeseries_optuna_")
    tmp_save_path = os.path.join(tmp_dir, f"{model_type}_trial.pth")

    def objective(trial: "optuna.Trial") -> float:
        trial_params: dict[str, Any] = {}
        for name, spec in search_space.items():
            stype = spec["type"]
            if stype == "categorical":
                trial_params[name] = trial.suggest_categorical(name, spec["choices"])
            elif stype == "float":
                trial_params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=bool(spec.get("log", False))
                )
            elif stype == "int":
                trial_params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=bool(spec.get("log", False))
                )
            else:
                raise ValueError(f"Unknown search-space spec type: {stype!r}")

        # Effective params: registry defaults <- user YAML <- trial sample.
        effective_params = {**registry_defaults, **user_model_params, **trial_params}
        if trial_epoch_cap is not None:
            effective_params["max_epochs_adam"] = min(
                int(effective_params.get("max_epochs_adam", trial_epoch_cap)),
                trial_epoch_cap,
            )
            effective_params["num_epochs_lbfgs"] = min(
                int(effective_params.get("num_epochs_lbfgs", trial_epoch_cap)),
                trial_epoch_cap,
            )

        cfg = parse_training_config(
            model_type=model_type,
            model_params=effective_params,
            train_block=train_block,
            global_random_state=global_random_state,
        )
        connectome_adj = _resolve_connectome_for_trial(trial_params)
        score = _train_and_score_val(
            model_type=model_type,
            cfg=cfg,
            sliced_clean=sliced_clean,
            sliced_noisy=sliced_noisy,
            data_noisy=data_noisy,
            split_cfg=split_cfg,
            connectome_adj=connectome_adj,
            seed=int(global_random_state),
            tmp_save_path=tmp_save_path,
            num_runs=int(effective_params.get("optuna_num_runs", 5)),
        )
        if score is None or not np.isfinite(score):
            raise optuna.exceptions.TrialPruned("invalid score")
        return float(score)

    LOGGER.info(
        "Starting Optuna search for %s: n_trials=%d, search axes=%s",
        model_type,
        n_trials,
        sorted(search_space.keys()),
    )
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    if not completed:
        raise RuntimeError(
            f"Optuna search for {model_type} produced no valid trials; "
            "every trial was pruned or errored. Inspect logs for details."
        )
    best = study.best_trial
    summary = {
        "method": "optuna",
        "n_trials_requested": int(n_trials),
        "n_trials_completed": int(len(completed)),
        "best_trial_number": int(best.number),
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "search_axes": sorted(search_space.keys()),
        "trial_epoch_cap": trial_epoch_cap,
        "optuna_num_runs": int(user_model_params.get("optuna_num_runs", registry_defaults.get("optuna_num_runs", 5))),
    }
    LOGGER.info(
        "Optuna search complete: best val RMSE=%.6f, best_params=%s",
        best.value,
        best.params,
    )
    return dict(best.params), summary


def train_timeseries_nvar(
    *,
    model_type: str,
    raw_path: str,
    output_dir: str,
    model_params: dict[str, Any],
    train_block: dict[str, Any],
    split_block: dict[str, Any],
    global_random_state: int = 42,
) -> TimeSeriesTrainResult:
    """Train an Adaptive NVAR model and write all artifacts under output_dir.

    Dispatch:
      * `train.tuning.method == "fixed"` (default): train once with the
        hyperparameters in `model_params`, fall back to dl_registry defaults
        for anything not provided.
      * `train.tuning.method == "optuna"`: in-loop hyperparameter search.
        Sample from the search space in `dl_registry.build_timeseries_dl_search_config`,
        train each trial, score by validation rollout RMSE at the largest
        horizon, then retrain the best trial's params and write the
        usual artifacts. The number of trials is `train.tuning.n_trials`
        (default 30). The search axes are defined in `dl_registry`, not in
        the YAML — to vary scientific factors across experiments, use the DOE.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}."
        )

    os.makedirs(output_dir, exist_ok=True)

    tuning_block = train_block.get("tuning", {}) if isinstance(train_block, dict) else {}
    tuning_method = str(tuning_block.get("method", "fixed")).strip().lower()

    if tuning_method == "optuna":
        best_params, study_summary = _run_optuna_timeseries(
            model_type=model_type,
            raw_path=raw_path,
            user_model_params=dict(model_params or {}),
            train_block=train_block,
            split_block=split_block,
            tuning_block=tuning_block,
            global_random_state=int(global_random_state),
        )
        # Merge searched best_params into the user's fixed params and rerun
        # the final train+test protocol. User-supplied YAML values for searched
        # axes are intentionally overridden by best_params here; that's the
        # whole point of running the search.
        merged_params = {**model_params, **best_params}
        train_block_for_final = dict(train_block or {})
        train_block_for_final["tuning"] = {"method": "fixed"}
        return _train_timeseries_nvar_repeated_final(
            model_type=model_type,
            raw_path=raw_path,
            output_dir=output_dir,
            model_params=merged_params,
            train_block=train_block_for_final,
            split_block=split_block,
            global_random_state=int(global_random_state),
            optuna_summary=study_summary,
        )

    if tuning_method not in {"fixed", ""}:
        raise ValueError(
            f"Unsupported tuning.method={tuning_method!r} for train.timeseries; "
            "expected 'fixed' or 'optuna'."
        )

    return _train_timeseries_nvar_repeated_final(
        model_type=model_type,
        raw_path=raw_path,
        output_dir=output_dir,
        model_params=model_params,
        train_block=train_block,
        split_block=split_block,
        global_random_state=int(global_random_state),
        optuna_summary=None,
    )


def _horizon_mean_std(rows: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    """Return mean/std over repeated final runs for each horizon key.

    `rows` is intentionally a flat list with keys such as `rmse_h25` or
    `val_rmse_h25`, so the same helper can summarize test, validation, and
    train repetitions by passing the corresponding row subset.
    """
    out: dict[str, Any] = {}
    for h in horizons:
        key = f"rmse_h{h}"
        vals = [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None and np.isfinite(row[key])
        ]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    out["num_runs"] = int(len(rows))
    out["runs"] = rows
    return out


def _prefixed_horizon_mean_std(
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Summarize repeated horizon metrics stored with a prefix.

    Example: prefix="val_" consumes keys like `val_rmse_h25` and writes
    canonical keys like `rmse_h25_mean`, making the output shape identical for
    train/val/test repeated summaries.
    """
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {
            "run_index": row.get("run_index"),
            "seed": row.get("seed"),
        }
        for h in horizons:
            src = f"{prefix}rmse_h{h}"
            if src in row:
                normalized[f"rmse_h{h}"] = row[src]
        normalized_rows.append(normalized)
    return _horizon_mean_std(normalized_rows, horizons)


def _train_timeseries_nvar_repeated_final(
    *,
    model_type: str,
    raw_path: str,
    output_dir: str,
    model_params: dict[str, Any],
    train_block: dict[str, Any],
    split_block: dict[str, Any],
    global_random_state: int,
    optuna_summary: Optional[dict[str, Any]],
) -> TimeSeriesTrainResult:
    """Run the final training/testing protocol, optionally repeated 25 times.

    Optuna is already finished before this function is called. Therefore these
    repetitions are not hyperparameter trials; they are independent final test
    trainings with the selected fixed hyperparameters.
    """
    cfg0 = parse_training_config(
        model_type=model_type,
        model_params=model_params,
        train_block=train_block,
        global_random_state=global_random_state,
    )
    num_runs = max(1, int(cfg0.test_num_runs))

    if num_runs <= 1:
        return _train_timeseries_nvar_once(
            model_type=model_type,
            raw_path=raw_path,
            output_dir=output_dir,
            model_params=model_params,
            train_block=train_block,
            split_block=split_block,
            global_random_state=int(global_random_state),
            optuna_summary=optuna_summary,
        )

    LOGGER.info(
        "Final test protocol: %d independent retrain+evaluate runs for %s "
        "(Optuna already chose hyperparameters; these runs measure training "
        "stochasticity at the chosen point).",
        num_runs,
        model_type,
    )
    first_result: Optional[TimeSeriesTrainResult] = None
    run_rows: list[dict[str, Any]] = []
    primary_key = f"rmse_h{max(cfg0.horizons)}"

    # Streaming progress file: a CSV row appended after each test-run finishes,
    # so a user can `tail -f` it and see results land in real time. This is
    # important on HPCC where a 25-run sweep takes hours; without it, the
    # only on-disk evidence is per-run subdirs, and metrics.json carries
    # stale single-run numbers until the very end.
    progress_csv = os.path.join(output_dir, f"{model_type}_test_runs_progress.csv")
    progress_fieldnames = (
        ["run_index", "seed", "wall_seconds"]
        + [f"rmse_h{h}" for h in cfg0.horizons]
        + [f"val_rmse_h{h}" for h in cfg0.horizons]
        + [f"train_rmse_h{h}" for h in cfg0.horizons]
    )

    import time as _time

    with open(progress_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=progress_fieldnames)
        writer.writeheader()

    for run_idx in range(num_runs):
        run_seed = int(cfg0.base_seed) + run_idx
        run_params = dict(model_params or {})
        run_params["base_seed"] = run_seed
        # Avoid recursive repetition inside each individual final run.
        run_params["test_num_runs"] = 1

        run_output_dir = output_dir if run_idx == 0 else os.path.join(output_dir, f"test_run_{run_idx:02d}")
        os.makedirs(run_output_dir, exist_ok=True)

        LOGGER.info(
            "[final test %d/%d] seed=%d output=%s",
            run_idx + 1,
            num_runs,
            run_seed,
            run_output_dir,
        )
        t_start = _time.time()

        result = _train_timeseries_nvar_once(
            model_type=model_type,
            raw_path=raw_path,
            output_dir=run_output_dir,
            model_params=run_params,
            train_block=train_block,
            split_block=split_block,
            global_random_state=run_seed,
            optuna_summary=optuna_summary if run_idx == 0 else None,
        )
        wall_seconds = _time.time() - t_start
        if first_result is None:
            first_result = result

        with open(result.metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        test_metrics = payload.get("test_rmse_horizons", {}) or {}
        val_metrics = payload.get("val_rmse_horizons", {}) or {}
        train_metrics = payload.get("train_rmse_horizons", {}) or {}
        row = {
            "run_index": int(run_idx),
            "seed": int(run_seed),
            **{k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
            **{f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
            **{f"train_{k}": float(v) for k, v in train_metrics.items() if isinstance(v, (int, float))},
        }
        run_rows.append(row)

        # Append a CSV row so external tools / `tail -f` can see progress.
        csv_row = {key: "" for key in progress_fieldnames}
        csv_row["run_index"] = int(run_idx)
        csv_row["seed"] = int(run_seed)
        csv_row["wall_seconds"] = f"{wall_seconds:.2f}"
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)) and k in csv_row:
                csv_row[k] = float(v)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                key = f"val_{k}"
                if key in csv_row:
                    csv_row[key] = float(v)
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                key = f"train_{k}"
                if key in csv_row:
                    csv_row[key] = float(v)
        with open(progress_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=progress_fieldnames).writerow(csv_row)

        # Update first_result.metrics_path with a partial-stats snapshot so
        # downstream tooling sees a meaningful aggregate even if the job is
        # killed mid-sweep. `partial: true` flags this clearly; the post-loop
        # rewrite below sets it to false when num_runs runs have all landed.
        try:
            partial_test_stats = _horizon_mean_std(run_rows, cfg0.horizons)
            partial_val_stats = _prefixed_horizon_mean_std(run_rows, cfg0.horizons, prefix="val_")
            partial_train_stats = _prefixed_horizon_mean_std(run_rows, cfg0.horizons, prefix="train_")
            with open(first_result.metrics_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            snapshot["rmse_single_run"] = snapshot.get(
                "rmse_single_run", snapshot.get("rmse")
            )
            snapshot["rmse"] = partial_test_stats.get(
                f"{primary_key}_mean", snapshot.get("rmse")
            )
            snapshot["rmse_std"] = partial_test_stats.get(f"{primary_key}_std")
            snapshot["test_num_runs"] = int(run_idx + 1)
            snapshot["test_num_runs_target"] = int(num_runs)
            snapshot["test_rmse_horizons_repeated"] = partial_test_stats
            snapshot["val_rmse_horizons_repeated"] = partial_val_stats
            snapshot["train_rmse_horizons_repeated"] = partial_train_stats
            # Convenience flat fields for benchmarking scripts.
            for h in cfg0.horizons:
                for suffix in ("mean", "std"):
                    k = f"rmse_h{h}_{suffix}"
                    if k in partial_test_stats:
                        snapshot[f"test_{k}"] = partial_test_stats[k]
                    if k in partial_val_stats:
                        snapshot[f"val_{k}"] = partial_val_stats[k]
                    if k in partial_train_stats:
                        snapshot[f"train_{k}"] = partial_train_stats[k]
            snapshot["partial"] = True
            if optuna_summary is not None:
                snapshot["tuning"] = optuna_summary
            with open(first_result.metrics_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=_json_default)
        except Exception as exc:
            LOGGER.warning(
                "[final test %d/%d] could not update partial metrics.json: %s",
                run_idx + 1,
                num_runs,
                exc,
            )

        LOGGER.info(
            "[final test %d/%d] done in %.1fs; %s=%.6f (running mean=%.6f, std=%.6f)",
            run_idx + 1,
            num_runs,
            wall_seconds,
            primary_key,
            row.get(primary_key, float("nan")),
            partial_test_stats.get(f"{primary_key}_mean", float("nan")),
            partial_test_stats.get(f"{primary_key}_std", float("nan")),
        )

    assert first_result is not None

    test_stats = _horizon_mean_std(run_rows, cfg0.horizons)
    val_stats = _prefixed_horizon_mean_std(run_rows, cfg0.horizons, prefix="val_")
    train_stats = _prefixed_horizon_mean_std(run_rows, cfg0.horizons, prefix="train_")
    rmse_mean_key = f"{primary_key}_mean"
    rmse_std_key = f"{primary_key}_std"

    # Update the main run's metrics JSON so downstream analysis sees the 25-run mean.
    with open(first_result.metrics_path, "r", encoding="utf-8") as f:
        main_metrics = json.load(f)
    # `rmse_single_run` was set on the first iteration; preserve it. If absent
    # (e.g. partial snapshots were skipped), fall back to current `rmse`.
    main_metrics["rmse_single_run"] = main_metrics.get("rmse_single_run", main_metrics.get("rmse"))
    main_metrics["rmse"] = test_stats.get(rmse_mean_key, main_metrics.get("rmse"))
    main_metrics["rmse_std"] = test_stats.get(rmse_std_key)
    main_metrics["test_num_runs"] = int(num_runs)
    main_metrics["test_num_runs_target"] = int(num_runs)
    main_metrics["test_rmse_horizons_repeated"] = test_stats
    main_metrics["val_rmse_horizons_repeated"] = val_stats
    main_metrics["train_rmse_horizons_repeated"] = train_stats
    for h in cfg0.horizons:
        for suffix in ("mean", "std"):
            k = f"rmse_h{h}_{suffix}"
            if k in test_stats:
                main_metrics[f"test_{k}"] = test_stats[k]
            if k in val_stats:
                main_metrics[f"val_{k}"] = val_stats[k]
            if k in train_stats:
                main_metrics[f"train_{k}"] = train_stats[k]
    main_metrics["partial"] = False
    if optuna_summary is not None:
        main_metrics["tuning"] = optuna_summary
    with open(first_result.metrics_path, "w", encoding="utf-8") as f:
        json.dump(main_metrics, f, indent=2, default=_json_default)

    # Also update split_metrics.json while preserving the usual train/val/test keys.
    split_metrics_path = main_metrics.get("split_metrics_path")
    if split_metrics_path and os.path.exists(split_metrics_path):
        with open(split_metrics_path, "r", encoding="utf-8") as f:
            split_metrics = json.load(f)
        split_metrics["test_single_run"] = split_metrics.get("test", {})
        split_metrics["val_single_run"] = split_metrics.get("val", {})
        split_metrics["train_single_run"] = split_metrics.get("train", {})
        split_metrics["test_repeated"] = test_stats
        split_metrics["val_repeated"] = val_stats
        split_metrics["train_repeated"] = train_stats
        # Use repeated means as the canonical benchmark values while retaining
        # the original first-run values above.
        split_metrics["test"] = {
            f"rmse_h{h}": test_stats.get(f"rmse_h{h}_mean")
            for h in cfg0.horizons
            if test_stats.get(f"rmse_h{h}_mean") is not None
        }
        split_metrics["val"] = {
            f"rmse_h{h}": val_stats.get(f"rmse_h{h}_mean")
            for h in cfg0.horizons
            if val_stats.get(f"rmse_h{h}_mean") is not None
        }
        split_metrics["train"] = {
            f"rmse_h{h}": train_stats.get(f"rmse_h{h}_mean")
            for h in cfg0.horizons
            if train_stats.get(f"rmse_h{h}_mean") is not None
        }
        with open(split_metrics_path, "w", encoding="utf-8") as f:
            json.dump(split_metrics, f, indent=2, default=_json_default)

    LOGGER.info(
        "Repeated final test complete: %s mean=%s std=%s over %d runs",
        primary_key,
        test_stats.get(rmse_mean_key),
        test_stats.get(rmse_std_key),
        num_runs,
    )
    return first_result


def _train_timeseries_nvar_once(
    *,
    model_type: str,
    raw_path: str,
    output_dir: str,
    model_params: dict[str, Any],
    train_block: dict[str, Any],
    split_block: dict[str, Any],
    global_random_state: int,
    optuna_summary: Optional[dict[str, Any]],
) -> TimeSeriesTrainResult:
    """Single-shot train + evaluate + persist. Used by both fixed and the
    post-Optuna retrain step.
    """

    # Resolve config (training + rollout) and split lengths.
    cfg = parse_training_config(
        model_type=model_type,
        model_params=model_params,
        train_block=train_block,
        global_random_state=global_random_state,
    )
    split_cfg = timeseries_io.parse_split_config(split_block)

    # Load raw [T, d] series and slice contiguous segments.
    data_clean, raw_meta = timeseries_io.load_raw_timeseries(raw_path)
    if data_clean.ndim != 2:
        raise ValueError(f"Loaded series must be 2-D, got {data_clean.shape}.")
    LOGGER.info(
        "Loaded raw time-series: shape=%s, dtype=%s, source=%s",
        data_clean.shape,
        data_clean.dtype,
        raw_meta,
    )

    # Reproducibility
    _seed_everything(cfg.base_seed)

    # Apply dataset-wide measurement noise (shared across train/val/test windows).
    np.random.seed(int(cfg.base_seed))  # noise is deterministic per seed
    data_noisy = _add_relative_gaussian_noise(data_clean, cfg.dataset_noise_scale)

    sliced_clean = timeseries_io.slice_time_series(data_clean, split_cfg)
    sliced_noisy = timeseries_io.slice_time_series(data_noisy, split_cfg)
    d = int(sliced_clean.d)

    # Add training-time noise on top of the (possibly already-noisy) train segment.
    np.random.seed(int(cfg.base_seed) + 1)
    train_with_noise = _add_relative_gaussian_noise(
        sliced_noisy.train, cfg.train_noise_scale
    )

    # Connectome (lazy: only when model_type == CONNECTOME_NVAR)
    bundle = None
    if model_type == CONNECTOME_NVAR:
        bundle = connectome_loader.build_connectome(
            xlsx_path=cfg.connectome_xlsx,
            sheet_name=cfg.connectome_sheet,
            n_select=cfg.n_connectome,
            selection_mode=cfg.connectome_selection_mode,
            selection_seed=cfg.connectome_selection_seed,
            binarize=cfg.connectome_binarize,
            randomize=(cfg.connectome_mode == "connectome_randomized"),
            randomize_swap_factor=cfg.connectome_swap_factor,
            randomize_seed=cfg.base_seed,
            normalization=cfg.connectome_normalization,
        )

    # Build & train ----------------------------------------------------------
    import torch

    device = _resolve_device(cfg.device)
    LOGGER.info("Training on device: %s", device)

    model = _build_model(
        model_type=model_type,
        cfg=cfg,
        d=d,
        connectome_adj=bundle.adjacency if bundle is not None else None,
    )

    model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
    X_train_t = _torchify(train_with_noise, device=device)
    model, train_diag = _train_adaptive_nvar(
        model=model,
        X_train=X_train_t,
        cfg=cfg,
        save_path=model_path,
        device=device,
    )

    # Rollout ----------------------------------------------------------------
    # The full noisy series provides the warmup context for both eval segments.
    X_full_t = _torchify(data_noisy, device=device)

    val_offset = split_cfg.warmup_len + split_cfg.train_len
    test_offset = val_offset + split_cfg.val_len

    val_per_window, val_pred, val_true, val_noisy = _rollout_segment(
        model=model,
        X_full=X_full_t,
        seg_offset=val_offset,
        seg_true=sliced_clean.val,
        seg_noisy=sliced_noisy.val,
        cfg=cfg,
        device=device,
    )
    test_per_window: list[dict[str, Any]] = []
    test_pred = test_true = test_noisy = None
    if split_cfg.test_len > 0:
        test_per_window, test_pred, test_true, test_noisy = _rollout_segment(
            model=model,
            X_full=X_full_t,
            seg_offset=test_offset,
            seg_true=sliced_clean.test,
            seg_noisy=sliced_noisy.test,
            cfg=cfg,
            device=device,
        )

    # Train-segment in-sample horizon RMSE: optional, but cheap and useful.
    # Use the *training* tail as a within-segment rollout: predict the last
    # `min(num_windows*max_horizon, train_len)` steps from the trained delta.
    train_per_window: list[dict[str, Any]] = []
    train_pred = train_true = train_noisy = None
    max_h = max(cfg.horizons)
    train_eval_len = min(cfg.num_windows * max_h, split_cfg.train_len - cfg.k - 1)
    if train_eval_len >= cfg.num_windows and train_eval_len > 0:
        train_seg_true = sliced_clean.train[-train_eval_len:]
        train_seg_noisy = sliced_noisy.train[-train_eval_len:]
        train_seg_offset = split_cfg.warmup_len + split_cfg.train_len - train_eval_len
        try:
            tcfg = TrainingConfig(**asdict(cfg))
            tcfg.num_windows = cfg.num_windows
            train_per_window, train_pred, train_true, train_noisy = _rollout_segment(
                model=model,
                X_full=X_full_t,
                seg_offset=train_seg_offset,
                seg_true=train_seg_true,
                seg_noisy=train_seg_noisy,
                cfg=tcfg,
                device=device,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping in-sample train rollout: %s", exc)

    # Aggregate metrics ------------------------------------------------------
    val_rmse = _aggregate_horizon_rmse(val_per_window, cfg.horizons)
    test_rmse = (
        _aggregate_horizon_rmse(test_per_window, cfg.horizons)
        if test_per_window
        else {}
    )
    train_rmse = (
        _aggregate_horizon_rmse(train_per_window, cfg.horizons)
        if train_per_window
        else {}
    )

    # Primary metric: RMSE at the largest horizon, on val if test missing.
    primary_h = max(cfg.horizons)
    primary_key = f"rmse_h{primary_h}"
    primary_target = test_rmse if test_rmse else val_rmse
    primary_value = primary_target.get(primary_key)

    # Persist metrics.json (CheMLFlow-shape, includes split_metrics_path) -----
    split_metrics: dict[str, dict[str, Any]] = {
        "train": train_rmse,
        "val": val_rmse,
        "test": test_rmse,
    }
    split_metrics_path = os.path.join(output_dir, f"{model_type}_split_metrics.json")
    with open(split_metrics_path, "w", encoding="utf-8") as f:
        json.dump(split_metrics, f, indent=2)

    metrics: dict[str, Any] = {
        "model_type": model_type,
        # Mirror the fields downstream tools look for first.
        "rmse": float(primary_value) if primary_value is not None else None,
        "primary_horizon": int(primary_h),
        "primary_metric": primary_key,
        "split_metrics_path": split_metrics_path,
        "train_rmse_horizons": train_rmse,
        "val_rmse_horizons": val_rmse,
        "test_rmse_horizons": test_rmse,
        "training": train_diag,
        "config": _serializable_config(cfg, bundle, raw_meta),
        "split": {
            "warmup_len": split_cfg.warmup_len,
            "train_len": split_cfg.train_len,
            "val_len": split_cfg.val_len,
            "test_len": split_cfg.test_len,
        },
        "data_dim": int(d),
    }
    if optuna_summary is not None:
        metrics["tuning"] = optuna_summary
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # Per-window-per-horizon CSV --------------------------------------------
    rich_csv_path = os.path.join(
        output_dir, f"{model_type}_rollout_per_window_per_horizon.csv"
    )
    horizon_keys = [f"rmse_h{h}" for h in cfg.horizons]
    fieldnames = ["segment", "window_index", "start", "end", "window_size"] + horizon_keys
    with open(rich_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for segment, rows in (
            ("train", train_per_window),
            ("val", val_per_window),
            ("test", test_per_window),
        ):
            for row in rows:
                payload = {"segment": segment}
                for key in fieldnames:
                    if key == "segment":
                        continue
                    payload[key] = row.get(key)
                writer.writerow(payload)

    # Predictions npz --------------------------------------------------------
    predictions_path = os.path.join(output_dir, f"{model_type}_predictions.npz")
    npz_payload: dict[str, np.ndarray] = {}
    if val_pred is not None:
        npz_payload["val_pred"] = val_pred
        npz_payload["val_true"] = val_true
        npz_payload["val_noisy"] = val_noisy
    if test_pred is not None:
        npz_payload["test_pred"] = test_pred
        npz_payload["test_true"] = test_true
        npz_payload["test_noisy"] = test_noisy
    if train_pred is not None:
        npz_payload["train_pred"] = train_pred
        npz_payload["train_true"] = train_true
        npz_payload["train_noisy"] = train_noisy
    if npz_payload:
        np.savez(predictions_path, **npz_payload)

    # Hyperparameters --------------------------------------------------------
    params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
    joblib.dump(
        {
            "model_type": model_type,
            "cfg": asdict(cfg),
            "split": {
                "warmup_len": split_cfg.warmup_len,
                "train_len": split_cfg.train_len,
                "val_len": split_cfg.val_len,
                "test_len": split_cfg.test_len,
            },
            "connectome_meta": (
                {
                    "n_nodes": bundle.n_nodes,
                    "sheet_name": bundle.sheet_name,
                    "node_names_first10": bundle.node_names[:10],
                }
                if bundle is not None
                else None
            ),
        },
        params_path,
    )

    LOGGER.info(
        "timeseries training complete: %s = %s (config=%s)",
        primary_key,
        metrics["rmse"],
        {"k": cfg.k, "num_windows": cfg.num_windows, "horizons": cfg.horizons},
    )

    return TimeSeriesTrainResult(
        model_path=model_path,
        params_path=params_path,
        metrics_path=metrics_path,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serializable_config(
    cfg: TrainingConfig,
    bundle: Optional[connectome_loader.ConnectomeBundle],
    raw_meta: dict,
) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["horizons"] = list(cfg.horizons)
    payload["raw_meta"] = raw_meta
    if bundle is not None:
        payload["connectome_resolved"] = {
            "n_nodes": bundle.n_nodes,
            "sheet_name": bundle.sheet_name,
        }
    return payload


def _json_default(obj):  # pragma: no cover - tiny utility
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON-serializable")
