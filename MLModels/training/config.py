from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeTrainingOptions:
    model_config: dict[str, Any]
    plot_split_performance: bool
    debug_logging: bool
    n_jobs: int
    tuning_method: str
    model_params: dict[str, Any]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_n_jobs(model_config: dict[str, Any] | None = None) -> int:
    """Resolve parallelism level for scikit-learn/joblib.

    Default to single-thread under pytest (subprocesses inherit PYTEST_CURRENT_TEST),
    since some sandboxes/CI environments disallow the syscalls loky uses to size its pool.
    """
    model_config = model_config or {}

    value = model_config.get("n_jobs")
    if value is None:
        env = os.environ.get("CHEMLFLOW_N_JOBS")
        if env:
            try:
                value = int(env)
            except ValueError:
                logging.warning("Invalid CHEMLFLOW_N_JOBS=%r; falling back to defaults.", env)

    if value is None:
        value = 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1

    try:
        value_int = int(value)
    except (TypeError, ValueError):
        logging.warning("Invalid n_jobs=%r; using 1.", value)
        return 1
    if value_int == 0:
        logging.warning("n_jobs=0 is invalid; using 1.")
        return 1

    # Some environments (notably sandboxed macOS setups) raise PermissionError on
    # os.sysconf("SC_SEM_NSEMS_MAX"), which joblib/loky calls when starting a
    # process pool. Fall back to single-thread execution to keep pipelines runnable.
    if value_int != 1:
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            logging.warning(
                "Parallel joblib backend is not permitted in this environment; forcing n_jobs=1."
            )
            return 1
        except Exception:
            # If sysconf is unavailable/unsupported, let joblib decide.
            pass
    return value_int


def parse_runtime_training_options(model_config: dict[str, Any] | None) -> RuntimeTrainingOptions:
    normalized = model_config if isinstance(model_config, dict) else {}
    plot_split_performance = as_bool(normalized.get("plot_split_performance", False))
    debug_logging = as_bool(normalized.get("_debug_logging", False))
    n_jobs = resolve_n_jobs(normalized)
    tuning_cfg = normalized.get("tuning", {}) if isinstance(normalized.get("tuning"), dict) else {}
    tuning_method = str(
        tuning_cfg.get(
            "method",
            normalized.get("tuning_method", "fixed"),
        )
    ).strip().lower() or "fixed"
    model_params = normalized.get("params", {}) if isinstance(normalized.get("params"), dict) else {}
    return RuntimeTrainingOptions(
        model_config=normalized,
        plot_split_performance=plot_split_performance,
        debug_logging=debug_logging,
        n_jobs=n_jobs,
        tuning_method=tuning_method,
        model_params=model_params,
    )


def resolve_chemprop_foundation_config(model_config: dict[str, Any]) -> tuple[str, str | None, bool]:
    foundation = str(model_config.get("foundation", "none")).strip().lower()
    if foundation in {"", "none"}:
        foundation = "none"
    if foundation not in {"none", "chemeleon"}:
        raise ValueError(
            "model.foundation must be one of: 'none', 'chemeleon'."
        )

    freeze_encoder = as_bool(model_config.get("freeze_encoder", False))
    checkpoint_path: str | None = None
    if foundation == "chemeleon":
        raw_path = model_config.get("foundation_checkpoint")
        if not raw_path or not str(raw_path).strip():
            raise ValueError(
                "model.foundation_checkpoint is required when model.foundation='chemeleon'."
            )
        checkpoint_path = os.path.expanduser(str(raw_path).strip())
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"model.foundation_checkpoint does not exist or is not a file: {checkpoint_path}"
            )
    elif freeze_encoder:
        raise ValueError(
            "model.freeze_encoder=true requires model.foundation='chemeleon'."
        )

    return foundation, checkpoint_path, freeze_encoder
