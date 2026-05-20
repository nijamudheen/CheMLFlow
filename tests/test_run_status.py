from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import (
    NODE_REGISTRY,
    _track_heavy_artifact,
    _track_heavy_artifact_glob,
    run_configured_pipeline_nodes,
)


def _base_config(tmp_path: Path) -> dict:
    return {
        "global": {
            "pipeline_type": "doe_status_test",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "run"),
            "target_column": "target",
            "thresholds": {"active": 1, "inactive": 2},
            "runs": {"enabled": True, "id": "status_test"},
        },
        "pipeline": {"nodes": ["get_data"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(tmp_path / "missing.csv")},
        },
    }


def test_run_status_written_on_node_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def _boom(_context: dict) -> None:
        raise RuntimeError("node exploded")

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _boom)

    with pytest.raises(RuntimeError, match="node exploded"):
        run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    run_dir = Path(config["global"]["run_dir"])
    run_config_path = run_dir / "run_config.yaml"
    status_path = run_dir / "run_status.json"
    assert run_config_path.exists()
    assert status_path.exists()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failed_node"] == "get_data"
    assert status["exception_type"] == "RuntimeError"
    assert "node exploded" in status["message"]
    assert status.get("traceback")


def test_run_status_written_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def _ok(_context: dict) -> None:
        return None

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _ok)

    assert run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    run_dir = Path(config["global"]["run_dir"])
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert "failed_node" not in status
    assert status["artifact_retention"] == "full"
    assert status["artifact_retention_summary"]["retention"] == "full"


def test_default_artifact_retention_keeps_heavy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path)

    def _ok(context: dict) -> None:
        for path in (
            context["paths"]["morgan_labeled"],
            str(Path(context["run_dir"]) / "random_forest_best_model.pkl"),
        ):
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("heavy", encoding="utf-8")

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _ok)

    assert run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    assert Path(config["global"]["base_dir"], "morgan_fingerprints_labeled.csv").exists()
    assert Path(config["global"]["run_dir"], "random_forest_best_model.pkl").exists()


def test_audit_light_artifact_retention_removes_heavy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path)
    config["global"]["artifact_retention"] = "audit_light"
    nested_model = Path(config["global"]["run_dir"]) / "seed_1" / "benchmark_a" / "chemprop-best-01.ckpt"
    nested_last = Path(config["global"]["run_dir"]) / "seed_1" / "benchmark_a" / "last.ckpt"

    def _ok(context: dict) -> None:
        for path in (
            context["paths"]["morgan_labeled"],
            context["paths"]["preprocessed_features"],
            context["paths"]["selected_features"],
            str(Path(context["run_dir"]) / "random_forest_best_model.pkl"),
            str(nested_model),
            str(nested_last),
        ):
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("heavy", encoding="utf-8")
            _track_heavy_artifact(context, str(file_path))
        _track_heavy_artifact_glob(context, str(Path(context["run_dir"]) / "**" / "chemprop-best-*.ckpt"))
        _track_heavy_artifact_glob(context, str(Path(context["run_dir"]) / "**" / "last.ckpt"))

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _ok)

    assert run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    assert not Path(config["global"]["base_dir"], "morgan_fingerprints_labeled.csv").exists()
    assert not Path(config["global"]["base_dir"], "preprocessed_features.csv").exists()
    assert not Path(config["global"]["base_dir"], "selected_features.csv").exists()
    assert not Path(config["global"]["run_dir"], "random_forest_best_model.pkl").exists()
    assert not nested_model.exists()
    assert not nested_last.exists()
    status = json.loads((Path(config["global"]["run_dir"]) / "run_status.json").read_text(encoding="utf-8"))
    summary = status["artifact_retention_summary"]
    assert summary["retention"] == "audit_light"
    assert summary["deleted_count"] >= 6
    assert summary["failed_delete_count"] == 0


def test_audit_light_failed_run_only_cleans_current_tracked_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path)
    config["global"]["artifact_retention"] = "audit_light"
    run_dir = Path(config["global"]["run_dir"])
    stale_file = run_dir / "stale_previous_best_model.pkl"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale", encoding="utf-8")
    tracked_file = run_dir / "fresh_best_model.pkl"

    def _boom(context: dict) -> None:
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text("new", encoding="utf-8")
        _track_heavy_artifact(context, str(tracked_file))
        _track_heavy_artifact_glob(context, str(run_dir / "*_best_model.pkl"))
        raise RuntimeError("node exploded")

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _boom)

    with pytest.raises(RuntimeError, match="node exploded"):
        run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    summary = status["artifact_retention_summary"]
    assert status["status"] == "failed"
    assert summary["retention"] == "audit_light"
    assert summary["run_status"] == "failed"
    assert summary["deleted_count"] == 1
    assert summary["skipped_preexisting_count"] >= 1
    assert summary["failed_delete_count"] == 0
    assert stale_file.exists()
    assert not tracked_file.exists()


def test_config_fingerprint_ignores_artifact_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_full = _base_config(tmp_path)
    config_full["global"]["run_dir"] = str(tmp_path / "run_full")
    config_audit = json.loads(json.dumps(config_full))
    config_audit["global"]["run_dir"] = str(tmp_path / "run_audit")
    config_audit["global"]["artifact_retention"] = "audit_light"

    def _ok(_context: dict) -> None:
        return None

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _ok)

    assert run_configured_pipeline_nodes(config_full, str(tmp_path / "full.yaml"))
    assert run_configured_pipeline_nodes(config_audit, str(tmp_path / "audit.yaml"))

    full_status = json.loads(
        (Path(config_full["global"]["run_dir"]) / "run_status.json").read_text(encoding="utf-8")
    )
    audit_status = json.loads(
        (Path(config_audit["global"]["run_dir"]) / "run_status.json").read_text(encoding="utf-8")
    )
    assert full_status["config_fingerprint"] == audit_status["config_fingerprint"]
