from __future__ import annotations

import json
from pathlib import Path

import yaml

from utilities.doe import generate_doe


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pgp_dataset_path() -> str:
    return str(REPO_ROOT / "tests" / "fixtures" / "data" / "pgp_small.csv")


def _base_clf_doe(tmp_path: Path) -> dict:
    return {
        "version": 1,
        "dataset": {
            "profile": "clf_local_csv",
            "name": "pgp_small_doe",
            "task_type": "classification",
            "target_column": "label",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
            "smiles_column": "SMILES",
            "label_source_column": "Activity",
            "label_map": {
                "positive": [1, "1", "active"],
                "negative": [0, "0", "inactive"],
            },
            "curate": {
                "properties": "Activity",
                "smiles_column": "SMILES",
                "dedupe_strategy": "drop_conflicts",
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.morgan"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [True],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data"),
            "global.runs.enabled": False,
            "global.random_state": 42,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
            "split.require_disjoint": True,
            "split.require_full_test_coverage": True,
            "train.tuning.method": "fixed",
            "train.reporting.plot_split_performance": True,
        },
        "output": {"dir": str(tmp_path / "generated")},
    }


def test_generate_doe_skips_invalid_model_task_combos(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["catboost_classifier", "random_forest"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 2
    assert summary["valid_cases"] == 1
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_MODEL_TASK_MISMATCH", 0) == 1
    assert len(result["valid_cases"]) == 1
    assert Path(result["valid_cases"][0]["config_path"]).exists()


def test_generate_doe_requires_validation_split_for_chemprop(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["none"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_VALIDATION_SPLIT_REQUIRED", 0) == 1


def test_generate_doe_skips_chemprop_with_preprocess_and_no_features(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["none"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS", 0) == 1
    assert summary["issue_counts"].get("DOE_CHEMPROP_PREPROCESS_UNSUPPORTED", 0) == 1


def test_generate_doe_enforces_split_mode_strategy_compatibility(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["tdc_scaffold"]
    spec["search_space"]["split.cv.n_splits"] = [5]
    spec["search_space"]["split.cv.repeats"] = [1]
    spec["search_space"]["split.cv.fold_index"] = [0]
    spec["search_space"]["split.cv.repeat_index"] = [0]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_STRATEGY_MODE_INVALID", 0) == 1


def test_generate_doe_validates_cv_fold_index_bounds(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["random"]
    spec["search_space"]["split.cv.n_splits"] = [3]
    spec["search_space"]["split.cv.repeats"] = [1]
    spec["search_space"]["split.cv.fold_index"] = [3]
    spec["search_space"]["split.cv.repeat_index"] = [0]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_PARAM_INVALID", 0) >= 1


def test_generate_doe_supports_auto_task_with_confirmation(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated_auto"
    spec = {
        "version": 1,
        "dataset": {
            "name": "pgp_auto",
            "task_type": "auto",
            "auto_confirmed": True,
            "target_column": "Activity",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["use.curated_features"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "pipeline.label_normalize": [False],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data_auto"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
        },
        "output": {"dir": str(output_dir)},
    }

    result = generate_doe(spec)
    summary = result["summary"]
    assert summary["task_type"] == "classification"
    assert summary["profile"] == "clf_local_csv"
    assert summary["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["task_type"] == "classification"
    assert config["train"]["model"]["type"] == "catboost_classifier"

    manifest_lines = Path(result["manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    payload = json.loads(manifest_lines[0])
    assert payload["status"] == "valid"


def test_generate_doe_isolates_case_artifacts_by_default(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["catboost_classifier", "dl_simple"]
    spec["defaults"]["global.base_dir"] = str(tmp_path / "data_root")
    spec["defaults"]["global.run_dir"] = str(tmp_path / "runs_root")

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 2

    config_paths = [Path(case["config_path"]) for case in result["valid_cases"]]
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]

    base_dirs = [cfg["global"]["base_dir"] for cfg in configs]
    run_dirs = [cfg["global"]["run_dir"] for cfg in configs]
    run_ids = [cfg["global"]["runs"]["id"] for cfg in configs]

    assert len(set(base_dirs)) == 2
    assert len(set(run_dirs)) == 2
    assert len(set(run_ids)) == 2
    assert any(path.endswith("case_0001") for path in base_dirs)
    assert any(path.endswith("case_0002") for path in base_dirs)


def test_generate_doe_validates_dataset_columns(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["dataset"]["smiles_column"] = "DOES_NOT_EXIST"
    spec["dataset"]["label_source_column"] = "MISSING_LABEL_SOURCE"
    spec["search_space"]["pipeline.label_normalize"] = [True]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_DATASET_COLUMN_MISSING", 0) >= 1
