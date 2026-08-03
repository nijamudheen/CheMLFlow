from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import run_doe_local


def test_local_runner_emits_running_attempt_before_terminal_result(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        f"global:\n  run_dir: {run_dir}\n  base_dir: {tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    fake_main = tmp_path / "fake_main.py"
    fake_main.write_text("raise SystemExit(0)\n", encoding="utf-8")
    updates: list[dict] = []

    result = run_doe_local._run_one(
        {
            "case_id": "case_0001",
            "parent_case_id": "parent_0001",
            "config_path": str(config_path),
            "status": "valid",
        },
        sequence=1,
        doe_dir=tmp_path,
        logs_dir=tmp_path / "logs",
        python_bin=sys.executable,
        main_path=fake_main,
        resume=False,
        dry_run=False,
        on_update=updates.append,
    )

    assert [row["state"] for row in updates] == ["RUNNING"]
    assert updates[0]["return_code"] is None
    assert updates[0]["end_time"] == ""
    assert result["state"] == "COMPLETED"
    assert result["return_code"] == 0


def test_local_runner_finalizes_attempt_when_process_launch_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        f"global:\n  run_dir: {run_dir}\n  base_dir: {tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    fake_main = tmp_path / "fake_main.py"
    fake_main.write_text("raise SystemExit(0)\n", encoding="utf-8")
    updates: list[dict] = []

    result = run_doe_local._run_one(
        {
            "case_id": "case_0001",
            "parent_case_id": "parent_0001",
            "config_path": str(config_path),
            "status": "valid",
        },
        sequence=1,
        doe_dir=tmp_path,
        logs_dir=tmp_path / "logs",
        python_bin=str(tmp_path / "missing-python"),
        main_path=fake_main,
        resume=False,
        dry_run=False,
        on_update=updates.append,
    )

    assert [row["state"] for row in updates] == ["RUNNING", "FAILED"]
    assert updates[0]["attempt_id"] == updates[1]["attempt_id"]
    assert result == updates[1]
    assert result["return_code"] is None
    assert result["end_time"]
    assert result["failure_reason"] == "launch_exception:FileNotFoundError"
    assert "FileNotFoundError" in Path(result["log_path"]).read_text(encoding="utf-8")


def test_local_runner_finalizes_attempt_before_reraising_interrupt(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        f"global:\n  run_dir: {run_dir}\n  base_dir: {tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    fake_main = tmp_path / "fake_main.py"
    fake_main.write_text("raise SystemExit(0)\n", encoding="utf-8")
    updates: list[dict] = []

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_doe_local.subprocess, "run", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_doe_local._run_one(
            {
                "case_id": "case_0001",
                "parent_case_id": "parent_0001",
                "config_path": str(config_path),
                "status": "valid",
            },
            sequence=1,
            doe_dir=tmp_path,
            logs_dir=tmp_path / "logs",
            python_bin=sys.executable,
            main_path=fake_main,
            resume=False,
            dry_run=False,
            on_update=updates.append,
        )

    assert [row["state"] for row in updates] == ["RUNNING", "CANCELLED"]
    assert updates[0]["attempt_id"] == updates[1]["attempt_id"]
    assert updates[1]["failure_reason"] == "interrupted:KeyboardInterrupt"
    assert updates[1]["end_time"]
