"""Smoke test for the time-series Adaptive NVAR pipeline.

Builds a tiny synthetic signal, runs train_timeseries_nvar end-to-end with a
miniature config (1 epoch each phase, 2 windows, single horizon), and asserts
all expected artifacts are written with valid JSON shapes.

Kept fast (< 5s on CPU) so it runs in CI without optional GPU dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Make the repo root importable even when this test runs in isolation.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# torch is optional in some CheMLFlow CI environments; gate the test on it.
torch = pytest.importorskip("torch")

from utilities import timeseries_io  # noqa: E402


def _make_synthetic_series(T: int = 200, d: int = 1, seed: int = 0) -> np.ndarray:
    """A short pseudo-Mackey-Glass-like series; just structure for the test."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(0.05 * t) + 0.5 * np.sin(0.13 * t + 1.0)
    noise = rng.normal(0.0, 0.05, T)
    series = (base + noise).astype(np.float32)
    if d > 1:
        # Add weakly correlated extra channels so [T, d] has the right shape.
        extra = np.stack(
            [series + 0.1 * rng.normal(size=T).astype(np.float32) for _ in range(d - 1)],
            axis=1,
        )
        return np.concatenate([series.reshape(-1, 1), extra], axis=1)
    return series.reshape(-1, 1)


def test_train_timeseries_nvar_writes_expected_artifacts(tmp_path):
    """End-to-end smoke: small Adaptive NVAR run produces all expected files."""
    from MLModels.training import timeseries_nvar

    # Fabricate a [T=200, d=1] series and write the canonical raw .npz.
    series = _make_synthetic_series(T=200, d=1)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(
        str(raw_path), series, source_meta={"source": "synthetic"}
    )

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    # Use a tiny budget so the test runs in seconds even on CPU.
    model_params = {
        "k": 3,
        "hidden_dim": 8,
        "feature_dim_m": 12,
        "lr_adam": 1e-2,
        "max_epochs_adam": 2,
        "adam_patience": 2,
        "lr_lbfgs": 0.5,
        "num_epochs_lbfgs": 2,
        "lbfgs_patience": 2,
        "lbfgs_max_iter": 4,
        "lbfgs_history_size": 4,
        "train_noise_scale": 0.0,
        "dataset_noise_scale": 0.0,
        "num_windows": 2,
        "horizons": [5, 10],
        "weight_decay": 0.0,
    }
    split_block = {
        "warmup_len": 20,
        "train_len": 100,
        "val_len": 40,
        "test_len": 40,
    }

    result = timeseries_nvar.train_timeseries_nvar(
        model_type="dl_adaptive_nvar",
        raw_path=str(raw_path),
        output_dir=str(output_dir),
        model_params=model_params,
        train_block={},
        split_block=split_block,
        global_random_state=0,
    )

    # All artifact paths must exist.
    assert os.path.isfile(result.model_path), "model state_dict missing"
    assert os.path.isfile(result.metrics_path), "metrics.json missing"
    assert os.path.isfile(result.params_path), "params.pkl missing"

    rich_csv = output_dir / "dl_adaptive_nvar_rollout_per_window_per_horizon.csv"
    assert rich_csv.exists(), "rich rollout CSV missing"

    split_metrics_path = output_dir / "dl_adaptive_nvar_split_metrics.json"
    assert split_metrics_path.exists(), "split_metrics.json missing"

    # metrics.json shape: top-level rmse + per-horizon dicts + path link.
    payload = json.loads(open(result.metrics_path).read())
    assert payload["model_type"] == "dl_adaptive_nvar"
    assert "rmse" in payload and payload["rmse"] is not None
    assert payload["primary_metric"] == "rmse_h10"
    assert payload["primary_horizon"] == 10
    assert "val_rmse_horizons" in payload
    assert "test_rmse_horizons" in payload
    assert payload["split_metrics_path"].endswith("dl_adaptive_nvar_split_metrics.json")

    # split_metrics shape: train/val/test each contain rmse_h{H}.
    split_metrics = json.loads(open(split_metrics_path).read())
    for segment in ("train", "val", "test"):
        assert segment in split_metrics, f"{segment} segment missing"
    assert "rmse_h5" in split_metrics["val"]
    assert "rmse_h10" in split_metrics["val"]

    # Rich CSV must have a row per (segment, window) and the horizon columns.
    with open(rich_csv) as f:
        header = f.readline().strip().split(",")
    for col in ("segment", "window_index", "rmse_h5", "rmse_h10"):
        assert col in header, f"rich CSV missing column {col}"


def test_train_timeseries_nvar_rejects_wrong_model_type(tmp_path):
    """An unknown model_type fails fast with a clear ValueError."""
    from MLModels.training import timeseries_nvar

    series = _make_synthetic_series(T=80, d=1)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(str(raw_path), series)

    with pytest.raises(ValueError, match="Unsupported model_type"):
        timeseries_nvar.train_timeseries_nvar(
            model_type="dl_resmlp",
            raw_path=str(raw_path),
            output_dir=str(tmp_path / "run"),
            model_params={"k": 2},
            train_block={},
            split_block={"warmup_len": 5, "train_len": 30, "val_len": 20, "test_len": 20},
        )


def test_parse_split_config_validates_required_keys():
    """Missing/zero split keys raise descriptive ValueError."""
    with pytest.raises(ValueError, match="missing required keys"):
        timeseries_io.parse_split_config({"warmup_len": 1, "train_len": 1})

    with pytest.raises(ValueError, match="train_len must be > 0"):
        timeseries_io.parse_split_config(
            {"warmup_len": 1, "train_len": 0, "val_len": 1, "test_len": 1}
        )


def test_slice_time_series_orientation():
    """Slicing preserves order and returns float32 arrays."""
    data = np.arange(60, dtype=np.float32).reshape(60, 1)
    cfg = timeseries_io.parse_split_config(
        {"warmup_len": 5, "train_len": 30, "val_len": 10, "test_len": 10}
    )
    sliced = timeseries_io.slice_time_series(data, cfg)
    assert sliced.warmup.shape == (5, 1)
    assert sliced.train.shape == (30, 1)
    assert sliced.val.shape == (10, 1)
    assert sliced.test.shape == (10, 1)
    assert float(sliced.train[0, 0]) == 5.0
    assert float(sliced.val[0, 0]) == 35.0
    assert float(sliced.test[0, 0]) == 45.0
    assert sliced.train.dtype == np.float32


def test_dl_registry_timeseries_search_configs():
    """Both architectures expose distinct, well-formed search configs."""
    from MLModels.train_models import DLSearchConfig
    from MLModels.training.dl_registry import build_timeseries_dl_search_config

    adaptive = build_timeseries_dl_search_config(
        model_type="dl_adaptive_nvar", dl_search_config_cls=DLSearchConfig
    )
    connectome = build_timeseries_dl_search_config(
        model_type="dl_connectome_nvar", dl_search_config_cls=DLSearchConfig
    )

    # Adaptive has hidden_dim; connectome has n_connectome — they should not
    # be interchangeable.
    assert "hidden_dim" in adaptive.search_space
    assert "hidden_dim" not in connectome.search_space
    assert "n_connectome" in connectome.search_space
    assert "n_connectome" not in adaptive.search_space

    # Both share the optimizer axes.
    for axis in ("k", "lr_adam", "lr_lbfgs"):
        assert axis in adaptive.search_space
        assert axis in connectome.search_space

    # Every spec must declare a recognized type.
    for cfg in (adaptive, connectome):
        for name, spec in cfg.search_space.items():
            assert spec["type"] in {"categorical", "float", "int"}, (name, spec)

    # Unknown model_type raises clearly.
    with pytest.raises(ValueError, match="Unsupported time-series model_type"):
        build_timeseries_dl_search_config(
            model_type="dl_simple", dl_search_config_cls=DLSearchConfig
        )


def test_train_timeseries_optuna_runs_and_writes_artifacts(tmp_path):
    """tuning.method=optuna runs the search, picks a winner, retrains, writes artifacts."""
    pytest.importorskip("optuna")
    from MLModels.training import timeseries_nvar

    series = _make_synthetic_series(T=300, d=1, seed=0)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(str(raw_path), series)

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    # 3 trials, all with tiny epoch budgets, so the test still finishes quickly.
    model_params = {
        # Searched axes (k, hidden_dim, lr_adam, lr_lbfgs) come from the
        # registry's search space. We just pin the non-searched knobs.
        "max_epochs_adam": 3,
        "adam_patience": 2,
        "num_epochs_lbfgs": 3,
        "lbfgs_patience": 2,
        "lbfgs_max_iter": 4,
        "lbfgs_history_size": 4,
        "train_noise_scale": 0.0,
        "dataset_noise_scale": 0.0,
        "num_windows": 2,
        "horizons": [5, 10],
        "weight_decay": 0.0,
    }
    train_block = {
        "tuning": {
            "method": "optuna",
            "n_trials": 3,
            "trial_epoch_cap": 3,
        }
    }
    split_block = {"warmup_len": 30, "train_len": 150, "val_len": 60, "test_len": 60}

    result = timeseries_nvar.train_timeseries_nvar(
        model_type="dl_adaptive_nvar",
        raw_path=str(raw_path),
        output_dir=str(output_dir),
        model_params=model_params,
        train_block=train_block,
        split_block=split_block,
        global_random_state=0,
    )

    payload = json.loads(open(result.metrics_path).read())
    assert "tuning" in payload, "metrics.json must record Optuna summary when method=optuna"
    tuning = payload["tuning"]
    assert tuning["method"] == "optuna"
    assert tuning["n_trials_requested"] == 3
    assert tuning["n_trials_completed"] >= 1
    assert "best_params" in tuning and isinstance(tuning["best_params"], dict)
    assert set(tuning["best_params"]).issubset({"k", "hidden_dim", "lr_adam", "lr_lbfgs"})
    # The final retrain used best_params merged with user-supplied fixed knobs.
    cfg_snapshot = payload["config"]
    assert cfg_snapshot["k"] == tuning["best_params"]["k"]
    assert cfg_snapshot["hidden_dim"] == tuning["best_params"]["hidden_dim"]


# ---------------------------------------------------------------------------
# v6 additions: GPU strictness, all-categorical Optuna, repeated-final UX
# ---------------------------------------------------------------------------


def test_device_strict_cuda_raises_when_unavailable(tmp_path):
    """device='cuda' must raise a clear error when CUDA isn't available.

    In the CI/test environment there is no CUDA; the resolver should not
    silently fall back to CPU. This guards against the v3-era surprise
    of running 5 hours on CPU when the user expected GPU.
    """
    import torch
    from MLModels.training import timeseries_nvar

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        pytest.skip("CUDA is available in this environment; strict-cuda test N/A.")

    series = _make_synthetic_series(T=80, d=1)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(str(raw_path), series)

    model_params = {
        "k": 2,
        "hidden_dim": 4,
        "feature_dim_m": 4,
        "max_epochs_adam": 2,
        "adam_patience": 1,
        "num_epochs_lbfgs": 2,
        "lbfgs_patience": 1,
        "lbfgs_max_iter": 2,
        "num_windows": 1,
        "horizons": [5],
        "device": "cuda",   # strict — must raise
    }
    with pytest.raises(RuntimeError, match="cuda"):
        timeseries_nvar.train_timeseries_nvar(
            model_type="dl_adaptive_nvar",
            raw_path=str(raw_path),
            output_dir=str(tmp_path / "run"),
            model_params=model_params,
            train_block={"tuning": {"method": "fixed"}},
            split_block={"warmup_len": 5, "train_len": 40, "val_len": 20, "test_len": 10},
        )


def test_device_auto_falls_back_to_cpu_silently(tmp_path):
    """device='auto' uses CPU when CUDA is absent — no raise."""
    import torch
    from MLModels.training import timeseries_nvar

    series = _make_synthetic_series(T=80, d=1)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(str(raw_path), series)

    model_params = {
        "k": 2,
        "hidden_dim": 4,
        "feature_dim_m": 4,
        "max_epochs_adam": 2,
        "adam_patience": 1,
        "num_epochs_lbfgs": 2,
        "lbfgs_patience": 1,
        "lbfgs_max_iter": 2,
        "num_windows": 1,
        "horizons": [5],
        "device": "auto",
        # Disable the 25-run protocol for this smoke test.
        "test_num_runs": 1,
    }
    result = timeseries_nvar.train_timeseries_nvar(
        model_type="dl_adaptive_nvar",
        raw_path=str(raw_path),
        output_dir=str(tmp_path / "run"),
        model_params=model_params,
        train_block={"tuning": {"method": "fixed"}},
        split_block={"warmup_len": 5, "train_len": 40, "val_len": 20, "test_len": 10},
    )
    assert os.path.exists(result.metrics_path)


def test_device_param_rejects_typos():
    """Misspelled device values fail at parse time, not silently."""
    from MLModels.training.timeseries_nvar import parse_training_config

    with pytest.raises(ValueError, match="device"):
        parse_training_config(
            model_type="dl_adaptive_nvar",
            model_params={"device": "gpu"},  # not a valid value
            train_block={},
            global_random_state=0,
        )


def test_dl_registry_adaptive_nvar_lr_adam_is_categorical():
    """v6: lr_adam must be categorical (notebook-faithful), not continuous float."""
    from MLModels.train_models import DLSearchConfig
    from MLModels.training.dl_registry import build_timeseries_dl_search_config

    adaptive = build_timeseries_dl_search_config(
        model_type="dl_adaptive_nvar", dl_search_config_cls=DLSearchConfig
    )
    spec = adaptive.search_space["lr_adam"]
    assert spec["type"] == "categorical", (
        f"adaptive_nvar lr_adam should be categorical (notebook-faithful); got {spec}"
    )
    # Exact notebook-2 grid: adam_lr_grid = [1e-4, 1e-3, 1e-2]
    assert sorted(spec["choices"]) == [1e-4, 1e-3, 1e-2]


def test_repeated_final_writes_progress_csv_and_partial_flag(tmp_path):
    """Each test-run iteration appends a row to *_test_runs_progress.csv
    and updates metrics.json with partial: true."""
    from MLModels.training import timeseries_nvar

    series = _make_synthetic_series(T=200, d=1, seed=0)
    raw_path = tmp_path / "raw.npz"
    timeseries_io.save_raw_timeseries(str(raw_path), series)

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    # 3 test runs, tiny budget so the whole thing finishes in a few seconds.
    model_params = {
        "k": 2,
        "hidden_dim": 4,
        "feature_dim_m": 4,
        "max_epochs_adam": 2,
        "adam_patience": 1,
        "num_epochs_lbfgs": 2,
        "lbfgs_patience": 1,
        "lbfgs_max_iter": 2,
        "lbfgs_history_size": 2,
        "num_windows": 2,
        "horizons": [5, 10],
        "test_num_runs": 3,
        "device": "auto",
    }
    result = timeseries_nvar.train_timeseries_nvar(
        model_type="dl_adaptive_nvar",
        raw_path=str(raw_path),
        output_dir=str(output_dir),
        model_params=model_params,
        train_block={"tuning": {"method": "fixed"}},
        split_block={"warmup_len": 30, "train_len": 100, "val_len": 30, "test_len": 30},
        global_random_state=0,
    )

    # Progress CSV exists with header + 3 data rows.
    progress_csv = output_dir / "dl_adaptive_nvar_test_runs_progress.csv"
    assert progress_csv.exists(), "progress CSV must be written incrementally"
    with open(progress_csv) as f:
        lines = f.readlines()
    assert lines[0].startswith("run_index,seed,wall_seconds")
    assert len(lines) == 1 + 3, f"expected 1 header + 3 data rows, got {len(lines)} lines"

    # Final metrics.json carries partial: false and the right run count.
    payload = json.loads(open(result.metrics_path).read())
    assert payload["partial"] is False
    assert payload["test_num_runs"] == 3
    assert payload["test_num_runs_target"] == 3
    assert payload["rmse_std"] is not None  # mean/std written
    # Per-run details preserved.
    stats = payload["test_rmse_horizons_repeated"]
    assert stats["num_runs"] == 3
    assert len(stats["runs"]) == 3
