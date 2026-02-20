import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List

import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "data"


def _run_pipeline(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["CHEMLFLOW_CONFIG"] = str(config_path)
    # Keep subprocess execution deterministic and avoid native thread/runtime flakiness
    # (notably with torch/chemprop in CI-like environments).
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("MPLBACKEND", "Agg")
    mpl_cfg = tmp_path / ".mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cfg))
    python = os.environ.get("CHEMLFLOW_PYTHON", sys.executable)
    result = subprocess.run(
        [python, "main.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Pipeline failed.\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return Path(config["global"]["run_dir"])


def _assert_metrics(run_dir: Path, model_type: str, keys: List[str]) -> None:
    metrics_path = run_dir / f"{model_type}_metrics.json"
    assert metrics_path.exists(), f"Missing metrics file: {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in keys:
        assert key in metrics, f"Missing metric '{key}' in {metrics_path}"


def test_e2e_qm9_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_qm9"
    config = {
        "global": {
            "pipeline_type": "qm9",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data_qm9"),
            "target_column": "gap",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {"nodes": ["get_data", "curate", "split", "featurize.rdkit", "train"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "qm9_small.csv")},
        },
        "curate": {"properties": "gap"},
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": False,
        },
        "train": {
            "model": {
                "type": "decision_tree",
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "decision_tree", ["r2", "mae"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_ara_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ara"
    config = {
        "global": {
            "pipeline_type": "ara",
            "task_type": "classification",
            "base_dir": str(tmp_path / "data_ara"),
            "target_column": "label",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "label.normalize", "split", "featurize.morgan", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "ara_small.csv")},
        },
        "curate": {
            "properties": "Activity",
            "smiles_column": "Smiles",
            "dedupe_strategy": "drop_conflicts",
            "label_column": "Activity",
            "prefer_largest_fragment": True,
        },
        "label": {
            "source_column": "Activity",
            "target_column": "label",
            "positive": ["active"],
            "negative": ["inactive"],
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": True,
        },
        "featurize": {"radius": 2, "n_bits": 128},
        "train": {
            "model": {
                "type": "catboost_classifier",
                "params": {
                    "depth": 4,
                    "learning_rate": 0.1,
                    "iterations": 10,
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "verbose": False,
                },
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "catboost_classifier", ["auc", "auprc", "accuracy", "f1"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_pgp_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_pgp"
    config = {
        "global": {
            "pipeline_type": "adme",
            "task_type": "classification",
            "base_dir": str(tmp_path / "data_pgp"),
            "target_column": "label",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "label.normalize", "split", "featurize.morgan", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "pgp_small.csv")},
        },
        "curate": {
            "properties": "Activity",
            "smiles_column": "SMILES",
            "dedupe_strategy": "drop_conflicts",
            "label_column": "Activity",
            "prefer_largest_fragment": True,
        },
        "label": {
            "source_column": "Activity",
            "target_column": "label",
            "positive": ["1", 1],
            "negative": ["0", 0],
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": True,
        },
        "featurize": {"radius": 2, "n_bits": 128},
        "train": {
            "model": {
                "type": "catboost_classifier",
                "params": {
                    "depth": 4,
                    "learning_rate": 0.1,
                    "iterations": 10,
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "verbose": False,
                },
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "catboost_classifier", ["auc", "auprc", "accuracy", "f1"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_pgp_chemprop_fast(tmp_path: Path) -> None:
    pytest.importorskip("chemprop")

    run_dir = tmp_path / "run_pgp_chemprop"
    config = {
        "global": {
            "pipeline_type": "adme",
            "task_type": "classification",
            "base_dir": str(tmp_path / "data_pgp"),
            "target_column": "label",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "label.normalize", "split", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "pgp_small.csv")},
        },
        "curate": {
            "properties": "Activity",
            "smiles_column": "SMILES",
            "dedupe_strategy": "drop_conflicts",
            "label_column": "Activity",
            "prefer_largest_fragment": True,
        },
        "label": {
            "source_column": "Activity",
            "target_column": "label",
            "positive": ["1", 1],
            "negative": ["0", 0],
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "train": {
            "model": {
                "type": "chemprop",
                "params": {
                    "max_epochs": 2,
                    "batch_size": 16,
                    "max_lr": 1e-3,
                    "mp_hidden_dim": 64,
                    "ffn_hidden_dim": 64,
                    "mp_depth": 2,
                },
            },
            "reporting": {
                "plot_split_performance": True,
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "chemprop", ["auc", "auprc", "accuracy", "f1"])
    metrics = json.loads((out_dir / "chemprop_metrics.json").read_text(encoding="utf-8"))
    split_metrics_path = metrics.get("split_metrics_path")
    split_plot_path = metrics.get("split_metrics_plot_path")
    assert split_metrics_path is not None
    assert split_plot_path is not None
    assert Path(split_metrics_path).exists()
    assert Path(split_plot_path).exists()
    split_metrics = json.loads(Path(split_metrics_path).read_text(encoding="utf-8"))
    assert set(split_metrics.keys()) == {"train", "val", "test"}
    assert {"auc", "auprc", "accuracy", "f1"}.issubset(split_metrics["train"].keys())
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_flash_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_flash"
    config = {
        "global": {
            "pipeline_type": "flash",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data_flash"),
            "target_column": "FP Exp.",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "split", "use.curated_features", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "flash_small.csv")},
        },
        "curate": {
            "properties": "FP Exp.",
            "smiles_column": "SMILES",
            "prefer_largest_fragment": True,
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": False,
        },
        "train": {
            "model": {
                "type": "random_forest",
            },
            "features": {
                "categorical_features": ["Family"],
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "random_forest", ["r2", "mae"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_ysi_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ysi"
    config = {
        "global": {
            "pipeline_type": "ysi",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data_ysi"),
            "target_column": "YSI",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {"nodes": ["get_data", "curate", "split", "featurize.rdkit", "train"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "ysi_small.csv")},
        },
        "curate": {
            "properties": "YSI",
            "smiles_column": "SMILES",
            "prefer_largest_fragment": True,
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": False,
        },
        "train": {
            "model": {
                "type": "decision_tree",
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "decision_tree", ["r2", "mae"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_pah_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_pah"
    config = {
        "global": {
            "pipeline_type": "pah",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data_pah"),
            "target_column": "log_p",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {"nodes": ["get_data", "curate", "split", "featurize.rdkit", "train"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "pah_small.csv")},
        },
        "curate": {
            "properties": "log_p",
            "smiles_column": "smiles",
            "prefer_largest_fragment": True,
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": False,
        },
        "train": {
            "model": {
                "type": "decision_tree",
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "decision_tree", ["r2", "mae"])
    assert (out_dir / "run_config.yaml").exists()
