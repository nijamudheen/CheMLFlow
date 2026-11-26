"""Device setting for neural models (train.model.params.device) and DOE GPU slots."""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from MLModels.training import torch_models

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, "auto"), ("", "auto"), ("AUTO", "auto"), (" cuda ", "cuda"), ("cpu", "cpu"), ("mps", "mps")],
)
def test_normalize_device_accepts_time_series_vocabulary(value, expected) -> None:
    assert torch_models.normalize_device(value) == expected


@pytest.mark.parametrize("value", ["gpu", "cuda:1", "0", "tpu"])
def test_normalize_device_rejects_other_values(value) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        torch_models.normalize_device(value)


def _no_accelerators(monkeypatch, torch, *, mps: bool = False) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_auto_order(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _no_accelerators(monkeypatch, torch)
    assert torch_models.resolve_device("auto").type == "cpu"
    _no_accelerators(monkeypatch, torch, mps=True)
    assert torch_models.resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_models.resolve_device("auto").type == "cuda"


def test_resolve_device_strict_values_raise_when_unavailable(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _no_accelerators(monkeypatch, torch)
    with pytest.raises(RuntimeError, match="cuda"):
        torch_models.resolve_device("cuda")
    with pytest.raises(RuntimeError, match="mps"):
        torch_models.resolve_device("mps")
    assert torch_models.resolve_device("cpu").type == "cpu"


# ---------------------------------------------------------------------------
# train_dl / predict_dl
# ---------------------------------------------------------------------------


def _tiny_data(n: int = 24, d: int = 3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = X.sum(axis=1)
    return X, y


def test_train_dl_reports_device_and_is_reproducible() -> None:
    torch = pytest.importorskip("torch")
    X, y = _tiny_data()

    def _run():
        torch_models.seed_dl_runtime(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
        result = torch_models.train_dl(
            model, X[:16], y[:16], X[16:], y[16:],
            epochs=3, batch_size=5, learning_rate=0.01, patience=5, random_state=7, device="cpu",
        )
        return result, torch_models.predict_dl(result["model"], X)

    first, preds_a = _run()
    _, preds_b = _run()
    assert first["device"] == "cpu"
    assert "device" not in first["best_params"]
    np.testing.assert_array_equal(preds_a, preds_b)


def test_predict_dl_uses_the_model_device_without_choosing_one(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("predict_dl must not pick a device")

    monkeypatch.setattr(torch_models, "resolve_device", _must_not_be_called)
    model = torch.nn.Linear(3, 1)
    preds = torch_models.predict_dl(model, np.ones((5, 3), dtype=np.float32), batch_size=2)
    assert preds.shape == (5,)
    assert torch_models.model_device(model).type == "cpu"


# ---------------------------------------------------------------------------
# train_model: params.device is a runtime setting, not a hyperparameter
# ---------------------------------------------------------------------------


def _split_frames():
    X, y = _tiny_data(n=32)
    X = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    y = pd.Series(y)
    return X.iloc[:20], y.iloc[:20], X.iloc[20:26], y.iloc[20:26], X.iloc[26:], y.iloc[26:]


def test_train_model_strips_device_from_params_and_records_it(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from MLModels import train_models

    seen: dict[str, object] = {}
    real_train_dl = train_models._train_dl

    def _spy_train_dl(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        return real_train_dl(*args, **kwargs)

    monkeypatch.setattr(train_models, "_train_dl", _spy_train_dl)
    X_train, y_train, X_val, y_val, X_test, y_test = _split_frames()
    _, result = train_models.train_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        model_type="dl_simple", output_dir=str(tmp_path), task_type="regression",
        model_config={"params": {"epochs": 2, "batch_size": 4, "device": "CPU"}},
        X_val=X_val, y_val=y_val,
    )

    assert seen["device"] == "cpu"
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["device"] == "cpu"
    with open(result.params_path, "rb") as fh:
        saved_params = pickle.load(fh)
    assert "device" not in saved_params

    reloaded = train_models.load_model(result.model_path, "dl_simple", input_dim=X_train.shape[1])
    assert next(reloaded.parameters()).device.type == "cpu"


def test_train_model_rejects_invalid_device_before_training(monkeypatch, tmp_path) -> None:
    from MLModels import train_models

    def _fail(*args, **kwargs):
        raise AssertionError("training must not start")

    monkeypatch.setattr(train_models, "_train_dl", _fail)
    X_train, y_train, X_val, y_val, X_test, y_test = _split_frames()
    with pytest.raises(ValueError, match="device must be one of"):
        train_models.train_model(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_type="dl_simple", output_dir=str(tmp_path), task_type="regression",
            model_config={"params": {"device": "cuda:1"}},
            X_val=X_val, y_val=y_val,
        )


def test_load_model_maps_weights_to_cpu(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from MLModels import train_models

    X_train, y_train, X_val, y_val, X_test, y_test = _split_frames()
    _, result = train_models.train_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        model_type="dl_simple", output_dir=str(tmp_path), task_type="regression",
        model_config={"params": {"epochs": 1, "batch_size": 4, "device": "cpu"}},
        X_val=X_val, y_val=y_val,
    )
    calls: list[dict] = []
    real_load = torch.load

    def _spy_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)
    train_models.load_model(result.model_path, "dl_simple", input_dim=X_train.shape[1])
    assert calls and calls[0].get("map_location") == "cpu"


# ---------------------------------------------------------------------------
# Runtime Optuna: out-of-memory trials are pruned but visible
# ---------------------------------------------------------------------------


def _optuna_config(batch_sizes):
    return {
        "params": {"epochs": 1, "device": "cpu"},
        "tuning": {
            "method": "optuna",
            "sampler": "grid",
            "n_trials": len(batch_sizes),
            "metric": "val_rmse",
            "direction": "minimize",
            "params": {"batch_size": {"type": "categorical", "choices": list(batch_sizes)}},
        },
    }


def _oom_for(monkeypatch, oom_batch_sizes):
    from MLModels import train_models

    real_train_dl = train_models._train_dl

    def _maybe_oom(*args, **kwargs):
        assert kwargs.get("device") == "cpu"
        if int(kwargs["batch_size"]) in oom_batch_sizes:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_train_dl(*args, **kwargs)

    monkeypatch.setattr(train_models, "_train_dl", _maybe_oom)
    return train_models


def test_runtime_optuna_marks_out_of_memory_trials(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    train_models = _oom_for(monkeypatch, {8})
    X_train, y_train, X_val, y_val, X_test, y_test = _split_frames()
    _, result = train_models.train_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        model_type="dl_simple", output_dir=str(tmp_path), task_type="regression",
        model_config=_optuna_config([4, 8]),
        X_val=X_val, y_val=y_val,
    )

    tuning = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))["tuning"]
    assert tuning["n_trials_pruned_out_of_memory"] == 1
    assert tuning["n_trials_pruned_error"] == 0
    assert "device" not in tuning["best_params"]
    trials = pd.read_csv(tuning["trials_path"])
    pruned = trials[trials["state"] == "PRUNED"]
    assert list(pruned["failure_reason"]) == ["out_of_memory"]
    assert list(pruned["param_batch_size"]) == [8]


def test_runtime_optuna_all_out_of_memory_says_so(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    train_models = _oom_for(monkeypatch, {4, 8})
    X_train, y_train, X_val, y_val, X_test, y_test = _split_frames()
    with pytest.raises(ValueError, match="2 ran out of memory"):
        train_models.train_model(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_type="dl_simple", output_dir=str(tmp_path), task_type="regression",
            model_config=_optuna_config([4, 8]),
            X_val=X_val, y_val=y_val,
        )


# ---------------------------------------------------------------------------
# run_doe_local.py GPU slots
# ---------------------------------------------------------------------------


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_doe_local_under_test", REPO_ROOT / "scripts" / "run_doe_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_doe(tmp_path: Path, n_cases: int) -> Path:
    doe_dir = tmp_path / "generated"
    doe_dir.mkdir(parents=True)
    records = []
    for i in range(n_cases):
        config_path = doe_dir / f"case_{i:04d}.yaml"
        config = {
            "global": {
                "base_dir": str(tmp_path / f"data_{i}"),
                "run_dir": str(tmp_path / "runs" / f"case_{i:04d}"),
            }
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        records.append(
            {"record_type": "execution_child", "status": "valid", "case_id": f"case_{i:04d}", "config_path": str(config_path)}
        )
    (doe_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return doe_dir


def test_gpu_slots_interleave_cards() -> None:
    runner = _load_runner()
    assert runner._gpu_slots("0,1", 2) == ["0", "1", "0", "1"]
    assert runner._gpu_slots(" 3 , ", 1) == ["3"]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--gpus", "0", "--max-workers", "2"],
        ["--workers-per-gpu", "2"],
        ["--gpus", "0", "--workers-per-gpu", "0"],
        ["--gpus", ","],
    ],
)
def test_runner_refuses_inconsistent_gpu_options(tmp_path, extra_args) -> None:
    runner = _load_runner()
    doe_dir = _write_doe(tmp_path, 1)
    assert runner.main(["--doe-dir", str(doe_dir), *extra_args]) == 2


def test_runner_gives_each_case_its_own_gpu_slot(monkeypatch, tmp_path) -> None:
    runner = _load_runner()
    doe_dir = _write_doe(tmp_path, 6)
    lock = threading.Lock()
    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    seen: list[str] = []

    def _fake_run(cmd, **kwargs):
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        with lock:
            seen.append(gpu)
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
        time.sleep(0.05)
        with lock:
            active[gpu] -= 1
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    rc = runner.main(["--doe-dir", str(doe_dir), "--gpus", "0,1", "--max-workers", "2"])

    assert rc == 0
    assert len(seen) == 6
    assert set(seen) == {"0", "1"}
    assert max(peak.values()) == 1
    log_text = (doe_dir / "local_logs").rglob("*")
    assert any("cuda_visible_devices=" in p.read_text(encoding="utf-8") for p in log_text if p.is_file())


def test_runner_leaves_cuda_visible_devices_alone_without_gpus(monkeypatch, tmp_path) -> None:
    runner = _load_runner()
    doe_dir = _write_doe(tmp_path, 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    seen: list[str | None] = []

    def _fake_run(cmd, **kwargs):
        seen.append(kwargs["env"].get("CUDA_VISIBLE_DEVICES"))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    assert runner.main(["--doe-dir", str(doe_dir)]) == 0
    assert seen == ["5"]
