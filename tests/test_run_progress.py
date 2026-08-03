from __future__ import annotations

import json
import time
from pathlib import Path

from utilities.run_progress import RunProgressReporter


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_progress_records_pipeline_and_known_training_scope(tmp_path: Path) -> None:
    reporter = RunProgressReporter(
        tmp_path,
        config_path="config/demo.yaml",
        pipeline_nodes=["get_data", "train"],
        heartbeat_interval_seconds=0.05,
    ).start()

    reporter.node_started("get_data")
    reporter.node_completed("get_data")
    reporter.node_started("train")
    reporter.training_started("epoch", unit="epoch", total=10, phase="training")
    reporter.training_update(
        "epoch",
        current=3,
        total=10,
        metrics={"val_loss": 0.25},
    )
    time.sleep(0.07)
    reporter.training_scope_finished("epoch", message="early stop")
    reporter.node_completed("train")
    reporter.success()

    payload = _load(tmp_path / "run_progress.json")
    assert payload["schema_version"] == 1
    assert payload["state"] == "success"
    assert payload["pipeline"]["completed"] == 2
    assert payload["pipeline"]["completed_nodes"] == ["get_data", "train"]
    assert payload["training"]["active"] is False
    scope = payload["training"]["scopes"]["epoch"]
    assert scope["current"] == 3
    assert scope["total"] == 10
    assert scope["indeterminate"] is False
    assert scope["metrics"]["val_loss"] == 0.25
    assert scope["status"] == "completed"
    assert payload["end_time"]


def test_run_progress_keeps_opaque_fit_indeterminate_and_records_failure(
    tmp_path: Path,
) -> None:
    reporter = RunProgressReporter(
        tmp_path,
        config_path="config/demo.yaml",
        pipeline_nodes=["train"],
        heartbeat_interval_seconds=60,
    ).start()
    reporter.node_started("train")
    reporter.training_indeterminate("fit", message="random forest fit active")
    reporter.failed(exception_type="RuntimeError", message="boom", failed_node="train")

    payload = _load(tmp_path / "run_progress.json")
    assert payload["state"] == "failed"
    assert payload["training"]["scopes"]["fit"]["total"] is None
    assert payload["training"]["scopes"]["fit"]["indeterminate"] is True
    assert payload["error"] == {
        "exception_type": "RuntimeError",
        "failed_node": "train",
        "message": "boom",
    }
