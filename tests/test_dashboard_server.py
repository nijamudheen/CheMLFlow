from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from chemlflow_dashboard import cli as dashboard_cli
from chemlflow_dashboard.server import DashboardHTTPServer
from chemlflow_dashboard.state import StudyStateCollector


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_server_serves_static_snapshot_detail_and_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"global:\n  task_type: classification\n  run_dir: {run_dir}\n"
        "pipeline:\n  feature_input: featurize.morgan\n"
        "train:\n  model:\n    type: random_forest\n",
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "random_forest_metrics.json").write_text(
        json.dumps({"auc": 0.81}),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text("line one\nline two\n", encoding="utf-8")

    collector = StudyStateCollector(
        mode="single", source_path=config_path, repo_root=tmp_path
    )
    server = DashboardHTTPServer(("127.0.0.1", 0), collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"{server.url}/", timeout=5) as response:
            assert b"CheMLFlow Agent Mission Control" in response.read()
        with urlopen(f"{server.url}/api/v1/snapshot", timeout=5) as response:
            snapshot = json.load(response)
        case_id = snapshot["cases"][0]["case_id"]
        with urlopen(
            f"{server.url}/api/v1/cases/{case_id}/detail", timeout=5
        ) as response:
            detail = json.load(response)
        assert detail["metric_value"] == 0.81
        with urlopen(
            f"{server.url}/api/v1/cases/{case_id}/log?tail=20", timeout=5
        ) as response:
            log = json.load(response)
        assert "line two" in log["text"]

        (run_dir / "run.log").unlink()
        with urlopen(
            f"{server.url}/api/v1/cases/{case_id}/log?tail=20", timeout=5
        ) as response:
            unavailable_log = json.load(response)
        assert unavailable_log["available"] is False
        assert "until this case starts" in unavailable_log["text"]
    finally:
        server.stop()
        thread.join(timeout=5)
        server.server_close()


def test_dashboard_static_uses_backend_population_spread() -> None:
    script = (REPO_ROOT / "chemlflow_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "function stdev(" not in script
    assert "best.metric_std" in script


def test_dashboard_static_uses_artifact_timeline_when_attached() -> None:
    script = (REPO_ROOT / "chemlflow_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "function artifactStudyClock(" in script
    assert "snapshot.timeline" in script
    assert "view.elapsedTicking = artifactClock?.ticking ?? false" in script


def test_dashboard_results_use_one_candidate_comparison_surface() -> None:
    app_script = (REPO_ROOT / "chemlflow_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    page = (REPO_ROOT / "chemlflow_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "function renderComparison(" in app_script
    assert "renderChart(snapshot)" not in app_script
    assert "renderMatrix(snapshot)" not in app_script
    assert "renderLeaderboard(snapshot)" not in app_script
    assert 'id="comparisonCard"' in page
    assert 'id="chartCard"' not in page
    assert 'id="leaderboard"' not in page
    assert 'id="matrixCard"' not in page


def test_persistent_dashboard_survives_start_command_exit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"global:\n  task_type: classification\n  run_dir: {run_dir}\n"
        "pipeline:\n  feature_input: featurize.morgan\n"
        "train:\n  model:\n    type: random_forest\n",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "dashboard.json"
    lifecycle_args = [
        "--config",
        str(config_path),
        "--metadata-file",
        str(metadata_path),
    ]
    metadata: dict[str, object] | None = None
    started = subprocess.run(
        [
            sys.executable,
            "-m",
            "chemlflow_dashboard",
            "start",
            *lifecycle_args,
            "--no-open",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    try:
        assert started.returncode == 0, started.stderr
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with urlopen(f"{metadata['url']}/api/v1/health", timeout=5) as response:
            health = json.load(response)
        assert health["status"] == "ok"
        assert health["instance_id"] == metadata["instance_id"]
        assert health["pid"] == metadata["pid"]
        assert health["source_path"] == str(config_path.resolve())
        assert list(run_dir.iterdir()) == []
        status = subprocess.run(
            [
                sys.executable,
                "-m",
                "chemlflow_dashboard",
                "status",
                *lifecycle_args,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert status.returncode == 0, status.stderr
        assert '"status": "running"' in status.stdout
        started_again = subprocess.run(
            [
                sys.executable,
                "-m",
                "chemlflow_dashboard",
                "start",
                *lifecycle_args,
                "--no-open",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert started_again.returncode == 0, started_again.stderr
        assert "already running" in started_again.stdout
        unchanged = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert unchanged["pid"] == metadata["pid"]
    finally:
        if metadata_path.exists():
            stopped = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chemlflow_dashboard",
                    "stop",
                    *lifecycle_args,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            assert stopped.returncode == 0, stopped.stderr

    assert metadata is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            urlopen(f"{metadata['url']}/api/v1/health", timeout=0.2)
        except (OSError, URLError):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("persistent dashboard was still serving after stop")


def test_persistent_stop_treats_process_exit_race_as_stopped(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("global:\n  task_type: classification\n", encoding="utf-8")
    metadata_path = tmp_path / "dashboard.json"
    metadata_path.write_text(
        json.dumps(
            {
                "instance_id": "instance-one",
                "pid": 4242,
                "url": "http://127.0.0.1:54321",
                "source_path": str(config_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config_path),
        doe_dir=None,
        metadata_file=str(metadata_path),
    )
    monkeypatch.setattr(dashboard_cli, "_metadata_is_healthy", lambda *_args: True)

    def process_exited(_pid: int, _signal_number: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(dashboard_cli.os, "kill", process_exited)

    assert dashboard_cli._persistent_stop(args) == 0
    assert not metadata_path.exists()
