from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

import analysis


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_config(path: Path, *, run_dir: Path, fold_index: int) -> None:
    payload = {
        "global": {
            "pipeline_type": "local_doe_test",
            "task_type": "regression",
            "base_dir": str(path.parent / "data"),
            "run_dir": str(run_dir),
            "target_column": "target",
            "thresholds": {"active": 1, "inactive": 2},
            "runs": {"enabled": True, "id": path.stem},
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "featurize.morgan", "split", "train"],
            "feature_input": "featurize.morgan",
        },
        "split": {
            "mode": "cv",
            "strategy": "random",
            "cv": {"n_splits": 2, "repeats": 1, "fold_index": fold_index, "repeat_index": 0},
        },
        "train": {
            "model": {"type": "random_forest"},
            "tuning": {"method": "fixed"},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_completed_run(run_dir: Path, *, config_path: Path, test_r2: float) -> None:
    split_metrics_path = run_dir / "random_forest_split_metrics.json"
    _write_json(
        split_metrics_path,
        {
            "train": {"r2": 0.9, "mae": 0.1},
            "val": {"r2": test_r2 - 0.05, "mae": 0.25},
            "test": {"r2": test_r2, "mae": 0.2},
        },
    )
    _write_json(
        run_dir / "random_forest_metrics.json",
        {"r2": test_r2, "mae": 0.2, "split_metrics_path": str(split_metrics_path)},
    )
    _write_json(
        run_dir / "run_status.json",
        {
            "status": "success",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:05+00:00",
            "config_path": str(config_path),
            "run_dir": str(run_dir),
        },
    )


def _write_run_status(run_dir: Path, *, config_path: Path, status: str) -> None:
    _write_json(
        run_dir / "run_status.json",
        {
            "status": status,
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:05+00:00",
            "config_path": str(config_path),
            "run_dir": str(run_dir),
        },
    )


def _write_manifest(doe_dir: Path, records: list[dict]) -> None:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    (doe_dir / "manifest.jsonl").write_text(text, encoding="utf-8")


def _write_execution_manifest(doe_dir: Path, records: list[dict]) -> Path:
    path = doe_dir / "execution_manifest.jsonl"
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")
    return path


def test_analysis_local_backend_reads_manifest_and_run_status(
    tmp_path: Path, monkeypatch
) -> None:
    doe_dir = tmp_path / "generated"
    output_dir = tmp_path / "analysis_local"
    parent_id = "parent_0001"
    sci_id = "sci_config_1"
    records = []
    for idx, test_r2 in enumerate((0.6, 0.8), start=1):
        config_path = doe_dir / f"case_{idx:04d}.yaml"
        run_dir = tmp_path / "runs" / f"case_{idx:04d}"
        _write_config(config_path, run_dir=run_dir, fold_index=idx - 1)
        _write_completed_run(run_dir, config_path=config_path, test_r2=test_r2)
        records.append(
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": f"case_{idx:04d}",
                "parent_case_id": parent_id,
                "scientific_config_id": sci_id,
                "execution_label": f"rep0_fold{idx - 1}",
                "config_path": str(config_path),
            }
        )
    _write_manifest(doe_dir, records)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.py",
            "--backend",
            "local",
            "--doe-dir",
            str(doe_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analysis.main() == 0

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["backend"] == "local"
    assert report["valid_config_count_from_manifest"] == 2
    assert report["child_job_count_from_log"] == 2
    assert report["mapping_mismatch"] is False
    assert report["state_counts"] == {"COMPLETED": 2}

    with (output_dir / "all_runs_metrics.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["parent_case_id"] == parent_id
    assert row["slice_count"] == "2"
    assert row["completed_slices"] == "2"
    assert float(row["test_r2"]) == 0.7


def test_run_doe_local_dry_run_writes_execution_manifest(tmp_path: Path) -> None:
    doe_dir = tmp_path / "generated"
    config_path = doe_dir / "case_0001.yaml"
    _write_config(config_path, run_dir=tmp_path / "runs" / "case_0001", fold_index=0)
    _write_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": "rep0_fold0",
                "config_path": str(config_path),
            }
        ],
    )

    output_manifest = doe_dir / "execution_manifest.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_doe_local.py"),
            "--doe-dir",
            str(doe_dir),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [
        json.loads(line)
        for line in output_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["record_type"] == "execution_attempt"
    assert rows[0]["backend"] == "local"
    assert rows[0]["execution_id"] == "local_00001"
    assert rows[0]["state"] == "DRY_RUN"
    assert rows[0]["case_id"] == "case_0001"
    assert rows[0]["config_path"] == str(config_path)


def test_analysis_local_backend_uses_failed_execution_attempt_over_running_status(
    tmp_path: Path, monkeypatch
) -> None:
    doe_dir = tmp_path / "generated"
    output_dir = tmp_path / "analysis_local"
    config_path = doe_dir / "case_0001.yaml"
    run_dir = tmp_path / "runs" / "case_0001"
    _write_config(config_path, run_dir=run_dir, fold_index=0)
    _write_run_status(run_dir, config_path=config_path, status="running")
    _write_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": "rep0_fold0",
                "config_path": str(config_path),
            }
        ],
    )
    execution_manifest = _write_execution_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_attempt",
                "backend": "local",
                "execution_id": "local_00001",
                "case_id": "case_0001",
                "config_path": str(config_path),
                "state": "FAILED",
                "return_code": 1,
                "failure_reason": "nonzero_return_code",
                "log_path": str(doe_dir / "local_logs" / "case_0001.log"),
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.py",
            "--backend",
            "local",
            "--doe-dir",
            str(doe_dir),
            "--execution-manifest",
            str(execution_manifest),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analysis.main() == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["state_counts"] == {"FAILED": 1}
    assert report["failure_reason_counts"] == {"nonzero_return_code": 1}

    with (output_dir / "jobs.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["state"] == "FAILED"
    assert rows[0]["log_path"].endswith("case_0001.log")


def test_analysis_local_backend_marks_success_without_metrics_partial(
    tmp_path: Path, monkeypatch
) -> None:
    doe_dir = tmp_path / "generated"
    output_dir = tmp_path / "analysis_local"
    config_path = doe_dir / "case_0001.yaml"
    run_dir = tmp_path / "runs" / "case_0001"
    _write_config(config_path, run_dir=run_dir, fold_index=0)
    _write_run_status(run_dir, config_path=config_path, status="success")
    _write_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": "rep0_fold0",
                "config_path": str(config_path),
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.py",
            "--backend",
            "local",
            "--doe-dir",
            str(doe_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analysis.main() == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["state_counts"] == {"PARTIAL": 1}
    assert report["failure_reason_counts"] == {"missing_metrics": 1}

    with (output_dir / "all_runs_metrics.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["state"] == "PARTIAL"
    assert rows[0]["failure_reason"] == "missing_metrics=1"
    assert rows[0]["completed_slices"] == "0"


def test_analysis_local_backend_keeps_top_level_metrics_without_split_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    doe_dir = tmp_path / "generated"
    output_dir = tmp_path / "analysis_local"
    config_path = doe_dir / "case_0001.yaml"
    run_dir = tmp_path / "runs" / "case_0001"
    _write_config(config_path, run_dir=run_dir, fold_index=0)
    _write_run_status(run_dir, config_path=config_path, status="success")
    _write_json(run_dir / "random_forest_metrics.json", {"r2": 0.72, "mae": 0.2})
    _write_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": "rep0_fold0",
                "config_path": str(config_path),
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.py",
            "--backend",
            "local",
            "--doe-dir",
            str(doe_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analysis.main() == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["state_counts"] == {"COMPLETED": 1}
    assert report["failure_reason_counts"] == {}
    assert report["metric_artifacts"] == {
        "completed_jobs": 1,
        "missing_metrics": 0,
        "top_level_metric_jobs": 1,
        "split_metric_jobs": 0,
        "missing_split_metrics": 1,
        "primary_metric_complete": True,
        "split_diagnostics_complete": False,
    }
    assert report["generalization"]["records_by_execution"] == 0

    with (output_dir / "all_runs_metrics.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["state"] == "COMPLETED"
    assert rows[0]["failure_reason"] == ""
    assert rows[0]["completed_slices"] == "1"
    assert rows[0]["r2"] == "0.72"


def test_run_doe_local_refuses_parallel_shared_artifact_dirs(tmp_path: Path) -> None:
    doe_dir = tmp_path / "generated"
    shared_run_dir = tmp_path / "runs" / "shared"
    records = []
    for idx in (1, 2):
        config_path = doe_dir / f"case_{idx:04d}.yaml"
        _write_config(config_path, run_dir=shared_run_dir, fold_index=idx - 1)
        records.append(
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": f"case_{idx:04d}",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": f"rep0_fold{idx - 1}",
                "config_path": str(config_path),
            }
        )
    _write_manifest(doe_dir, records)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_doe_local.py"),
            "--doe-dir",
            str(doe_dir),
            "--max-workers",
            "2",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "refusing parallel execution" in completed.stderr
    assert "run_dir" in completed.stderr


def test_run_doe_local_refuses_parallel_stop_on_failure(tmp_path: Path) -> None:
    doe_dir = tmp_path / "generated"
    config_path = doe_dir / "case_0001.yaml"
    _write_config(config_path, run_dir=tmp_path / "runs" / "case_0001", fold_index=0)
    _write_manifest(
        doe_dir,
        [
            {
                "record_type": "execution_child",
                "status": "valid",
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "scientific_config_id": "sci_config_1",
                "execution_label": "rep0_fold0",
                "config_path": str(config_path),
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_doe_local.py"),
            "--doe-dir",
            str(doe_dir),
            "--max-workers",
            "2",
            "--stop-on-failure",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "refusing --max-workers > 1 with --stop-on-failure" in completed.stderr
    assert "serial fail-fast debugging" in completed.stderr
