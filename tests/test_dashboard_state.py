from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chemlflow_dashboard.state import StudyStateCollector


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_config(path: Path, run_dir: Path, model_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "global:",
                "  task_type: classification",
                f"  run_dir: {run_dir}",
                "pipeline:",
                "  feature_input: featurize.morgan",
                "train:",
                "  model:",
                f"    type: {model_type}",
            ]
        ),
        encoding="utf-8",
    )


def _success_run(run_dir: Path, model_type: str, auc: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        run_dir / "run_status.json",
        {"status": "success", "start_time": now, "end_time": now},
    )
    _write_json(
        run_dir / "run_progress.json",
        {
            "state": "success",
            "start_time": now,
            "end_time": now,
            "updated_at": now,
            "last_heartbeat_at": now,
            "pipeline": {
                "total": 3,
                "completed": 3,
                "nodes": ["get_data", "split", "train"],
            },
            "training": {"active": False, "scopes": {}},
        },
    )
    _write_json(run_dir / f"{model_type}_metrics.json", {"auc": auc})
    (run_dir / "run.log").write_text("training complete\n", encoding="utf-8")


def test_doe_snapshot_separates_skips_and_ranks_only_complete_parents(
    tmp_path: Path,
) -> None:
    doe_dir = tmp_path / "generated"
    config_dir = doe_dir / "configs"
    runs = [tmp_path / "run_a", tmp_path / "run_b", tmp_path / "run_c"]
    configs = [config_dir / f"case_{letter}.yaml" for letter in ("a", "b", "c")]
    _write_config(configs[0], runs[0], "random_forest")
    _write_config(configs[1], runs[1], "random_forest")
    _write_config(configs[2], runs[2], "dl_simple")
    _success_run(runs[0], "random_forest", 0.8)
    _success_run(runs[1], "random_forest", 0.9)

    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        runs[2] / "run_progress.json",
        {
            "state": "running",
            "start_time": now,
            "updated_at": now,
            "last_heartbeat_at": now,
            "pipeline": {"total": 3, "completed": 2, "current": "train", "nodes": []},
            "training": {
                "active": True,
                "scopes": {
                    "fit": {
                        "status": "running",
                        "unit": "fit",
                        "current": None,
                        "total": None,
                        "indeterminate": True,
                    }
                },
            },
        },
    )
    manifest = [
        {
            "record_type": "execution_child",
            "case_id": "case_a",
            "parent_case_id": "parent_rf",
            "status": "valid",
            "config_path": str(configs[0]),
            "execution_label": "fold0",
            "execution_factors": {"split.cv.fold_index": 0},
            "factors": {"train.model.type": "random_forest"},
        },
        {
            "record_type": "execution_child",
            "case_id": "case_b",
            "parent_case_id": "parent_rf",
            "status": "valid",
            "config_path": str(configs[1]),
            "execution_label": "fold1",
            "execution_factors": {"split.cv.fold_index": 1},
            "factors": {"train.model.type": "random_forest"},
        },
        {
            "record_type": "execution_child",
            "case_id": "case_c",
            "parent_case_id": "parent_dl",
            "status": "valid",
            "config_path": str(configs[2]),
            "execution_label": "fold0",
            "execution_factors": {"split.cv.fold_index": 0},
            "factors": {"train.model.type": "dl_simple"},
        },
        {
            "record_type": "execution_child",
            "case_id": "case_skip",
            "parent_case_id": "parent_skip",
            "status": "skipped",
            "issues": [
                {"code": "DOE_COMBINATION_NOT_ALLOWED", "message": "incompatible"}
            ],
        },
    ]
    _write_jsonl(doe_dir / "manifest.jsonl", manifest)
    _write_jsonl(
        doe_dir / "parent_manifest.jsonl",
        [
            {
                "case_id": "parent_rf",
                "parent_case_id": "parent_rf",
                "model_type": "random_forest",
            },
            {
                "case_id": "parent_dl",
                "parent_case_id": "parent_dl",
                "model_type": "dl_simple",
            },
        ],
    )
    _write_jsonl(
        doe_dir / "execution_manifest.jsonl",
        [
            {
                "record_type": "execution_attempt",
                "case_id": "case_c",
                "state": "RUNNING",
                "start_time": now,
                "log_path": str(runs[2] / "run.log"),
            }
        ],
    )
    _write_json(
        doe_dir / "summary.json",
        {
            "task_type": "classification",
            "selection": {"primary_metric": "auc", "optimize": "max"},
        },
    )

    collector = StudyStateCollector(mode="doe", source_path=doe_dir, repo_root=tmp_path)
    snapshot = collector.collect()

    assert snapshot["summary"]["valid_cases"] == 3
    assert snapshot["summary"]["completed"] == 2
    assert snapshot["summary"]["running"] == 1
    assert snapshot["summary"]["progress_fraction"] == 2 / 3
    assert snapshot["timeline"]["started_at"] is not None
    assert snapshot["timeline"]["ended_at"] is None
    assert snapshot["skips"]["count"] == 1
    assert snapshot["skips"]["by_reason"] == {"DOE_COMBINATION_NOT_ALLOWED": 1}
    assert len(snapshot["leaderboard"]) == 1
    assert snapshot["leaderboard"][0]["parent_case_id"] == "parent_rf"
    assert abs(snapshot["leaderboard"][0]["metric_mean"] - 0.85) < 1e-12
    assert abs(snapshot["leaderboard"][0]["metric_std"] - 0.05) < 1e-12
    assert snapshot["leaderboard"][0]["provisional"] is True

    detail = collector.case_detail("case_a")
    assert detail is not None
    assert any(item["name"] == "run.log" for item in detail["artifacts"])
    assert collector.log_path("case_a") == runs[0] / "run.log"


def test_running_case_becomes_stale_only_after_heartbeat_threshold(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "case.yaml"
    run_dir = tmp_path / "run"
    _write_config(config_path, run_dir, "random_forest")
    old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    _write_json(
        run_dir / "run_progress.json",
        {
            "state": "running",
            "start_time": old,
            "updated_at": old,
            "last_heartbeat_at": old,
            "pipeline": {"total": 1, "completed": 0},
            "training": {"active": True, "scopes": {}},
        },
    )

    collector = StudyStateCollector(
        mode="single",
        source_path=config_path,
        repo_root=tmp_path,
        stale_after_seconds=20,
    )
    case = collector.collect()["cases"][0]
    assert case["status"] == "stale"
    assert case["freshness_seconds"] >= 100


def test_new_running_attempt_does_not_reuse_old_success_or_metric(
    tmp_path: Path,
) -> None:
    doe_dir = tmp_path / "generated"
    config_path = doe_dir / "configs" / "case.yaml"
    run_dir = tmp_path / "run"
    _write_config(config_path, run_dir, "random_forest")
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_json(
        run_dir / "run_status.json",
        {"status": "success", "start_time": old, "end_time": old},
    )
    _write_json(
        run_dir / "run_progress.json",
        {
            "state": "success",
            "start_time": old,
            "end_time": old,
            "updated_at": old,
            "last_heartbeat_at": old,
            "pipeline": {"total": 3, "completed": 3},
            "training": {"active": False, "scopes": {}},
        },
    )
    metrics_path = run_dir / "random_forest_metrics.json"
    _write_json(metrics_path, {"auc": 0.99})
    old_epoch = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
    os.utime(metrics_path, (old_epoch, old_epoch))
    now = datetime.now(timezone.utc).isoformat()
    _write_jsonl(
        doe_dir / "manifest.jsonl",
        [
            {
                "record_type": "execution_child",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "status": "valid",
                "config_path": str(config_path),
                "factors": {"train.model.type": "random_forest"},
            }
        ],
    )
    _write_jsonl(
        doe_dir / "execution_manifest.jsonl",
        [
            {
                "record_type": "execution_attempt",
                "case_id": "case_0001",
                "state": "RUNNING",
                "start_time": now,
            }
        ],
    )
    _write_json(
        doe_dir / "summary.json",
        {
            "task_type": "classification",
            "selection": {"primary_metric": "auc", "optimize": "max"},
        },
    )

    case = StudyStateCollector(
        mode="doe", source_path=doe_dir, repo_root=tmp_path
    ).collect()["cases"][0]
    assert case["status"] == "running"
    assert case["metric_value"] is None
    assert case["metrics"] == {}
    assert case["progress"] == {}
    assert case["run_status"] == {}
    assert case["start_time"] == now

    _write_jsonl(
        doe_dir / "execution_manifest.jsonl",
        [
            {
                "record_type": "execution_attempt",
                "case_id": "case_0001",
                "state": "SKIPPED",
                "start_time": now,
                "failure_reason": "already_successful",
            }
        ],
    )
    resumed = StudyStateCollector(
        mode="doe",
        source_path=doe_dir,
        repo_root=tmp_path,
    ).collect()["cases"][0]
    assert resumed["status"] == "completed"
    assert resumed["metric_value"] == 0.99
