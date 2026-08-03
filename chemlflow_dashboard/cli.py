"""Command-line launcher for live CheMLFlow dashboards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from .server import DashboardHTTPServer
from .state import StudyStateCollector

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_DIR = Path(tempfile.gettempdir()) / "chemlflow-dashboard"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_lifecycle_metadata(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _default_metadata_path(source_path: Path) -> Path:
    source_key = str(source_path.resolve()).encode("utf-8")
    digest = hashlib.sha256(source_key).hexdigest()[:16]
    return LIFECYCLE_DIR / f"{digest}.json"


def _metadata_path(args: argparse.Namespace, source_path: Path) -> Path:
    configured = str(getattr(args, "metadata_file", "")).strip()
    return Path(configured).resolve() if configured else _default_metadata_path(source_path)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _health_payload(url: str, *, timeout: float = 0.5) -> dict[str, Any]:
    try:
        with urlopen(f"{url}/api/v1/health", timeout=timeout) as response:
            loaded = json.load(response)
    except (OSError, URLError, json.JSONDecodeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _metadata_is_healthy(metadata: dict[str, Any], source_path: Path) -> bool:
    expected_source = str(source_path.resolve())
    if str(metadata.get("source_path", "")) != expected_source:
        return False
    instance_id = str(metadata.get("instance_id", "")).strip()
    url = str(metadata.get("url", "")).rstrip("/")
    if not instance_id or not url:
        return False
    health = _health_payload(url)
    try:
        metadata_pid = int(metadata.get("pid", 0))
        health_pid = int(health.get("pid", 0))
    except (TypeError, ValueError):
        return False
    return (
        health.get("status") == "ok"
        and str(health.get("instance_id", "")) == instance_id
        and health_pid == metadata_pid
        and str(health.get("source_path", "")) == expected_source
    )


class LauncherState:
    def __init__(self, *, mode: str) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "mode": mode,
            "phase": "ready",
            "status": "idle",
            "message": "Dashboard attached; no process launched.",
            "started_at": None,
            "updated_at": _utc_now(),
            "ended_at": None,
            "return_code": None,
            "preflight": {},
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
            self._state["updated_at"] = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML must parse to a mapping: {path}")
    return loaded


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_lifecycle_source(args: argparse.Namespace) -> tuple[str, Path]:
    raw_config = str(getattr(args, "config", "") or "").strip()
    if raw_config:
        source_path = _resolve_repo_path(raw_config)
        if not source_path.is_file():
            raise SystemExit(f"Config not found: {source_path}")
        return "single", source_path
    raw_doe_dir = str(getattr(args, "doe_dir", "") or "").strip()
    source_path = _resolve_repo_path(raw_doe_dir)
    if not source_path.is_dir():
        raise SystemExit(f"DOE directory not found: {source_path}")
    return "doe", source_path


def _remove_metadata_for_instance(path: Path, instance_id: str) -> None:
    current = _read_lifecycle_metadata(path)
    if str(current.get("instance_id", "")) != instance_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _persistent_serve(args: argparse.Namespace) -> int:
    mode, source_path = _resolve_lifecycle_source(args)
    metadata_path = _metadata_path(args, source_path)
    instance_id = str(args.instance_id)
    launcher = LauncherState(mode="start")
    collector = StudyStateCollector(
        mode=mode,
        source_path=source_path,
        repo_root=REPO_ROOT,
        stale_after_seconds=args.stale_after,
        launcher_state=launcher.snapshot,
    )
    server = DashboardHTTPServer(
        (args.host, int(args.port)),
        collector,
        instance_id=instance_id,
    )
    metadata = {
        "schema_version": 1,
        "instance_id": instance_id,
        "pid": os.getpid(),
        "url": server.url,
        "mode": mode,
        "source_path": str(source_path.resolve()),
        "metadata_path": str(metadata_path),
        "log_path": str(Path(args.log_file).resolve()),
        "started_at": _utc_now(),
        "python": sys.executable,
    }
    _atomic_write_json(metadata_path, metadata)
    print(f"[dashboard] persistent mission control: {server.url}", flush=True)
    print(f"[dashboard] source: {source_path}", flush=True)

    def stop_on_signal(_signal_number: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.stop_event.set()
        server.server_close()
        _remove_metadata_for_instance(metadata_path, instance_id)
    return 0


def _persistent_start(args: argparse.Namespace) -> int:
    mode, source_path = _resolve_lifecycle_source(args)
    metadata_path = _metadata_path(args, source_path)
    existing = _read_lifecycle_metadata(metadata_path)
    if existing and _metadata_is_healthy(existing, source_path):
        url = str(existing["url"])
        print(f"[dashboard] already running: {url}")
        print(f"[dashboard] metadata: {metadata_path}")
        if not args.no_open:
            webbrowser.open(url)
        return 0
    if existing:
        try:
            existing_pid = int(existing.get("pid", 0))
        except (TypeError, ValueError):
            existing_pid = 0
        if _pid_is_alive(existing_pid):
            print(
                "[dashboard] refusing to replace metadata for a live, unverified "
                f"process (pid={existing_pid}): {metadata_path}",
                file=sys.stderr,
            )
            return 2
        try:
            metadata_path.unlink()
        except FileNotFoundError:
            pass

    instance_id = uuid.uuid4().hex
    log_path = metadata_path.with_suffix(".log")
    command = [
        sys.executable,
        "-m",
        "chemlflow_dashboard",
        "_serve",
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--stale-after",
        str(float(args.stale_after)),
        "--no-open",
        "--metadata-file",
        str(metadata_path),
        "--instance-id",
        instance_id,
        "--log-file",
        str(log_path),
    ]
    if mode == "single":
        command.extend(["--config", str(source_path)])
    else:
        command.extend(["--doe-dir", str(source_path)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            close_fds=True,
            start_new_session=True,
        )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            try:
                detail = log_path.read_text(encoding="utf-8")[-4000:]
            except OSError:
                detail = ""
            print(
                f"[dashboard] persistent server exited with code {process.returncode}.\n{detail}",
                file=sys.stderr,
            )
            return 2
        metadata = _read_lifecycle_metadata(metadata_path)
        if (
            str(metadata.get("instance_id", "")) == instance_id
            and int(metadata.get("pid", 0) or 0) == process.pid
            and _metadata_is_healthy(metadata, source_path)
        ):
            url = str(metadata["url"])
            print(f"[dashboard] persistent mission control: {url}")
            print(f"[dashboard] source: {source_path}")
            print(f"[dashboard] metadata: {metadata_path}")
            if not args.no_open:
                webbrowser.open(url)
            return 0
        time.sleep(0.05)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    print(
        f"[dashboard] persistent server did not become healthy; see {log_path}",
        file=sys.stderr,
    )
    return 2


def _persistent_status(args: argparse.Namespace) -> int:
    _, source_path = _resolve_lifecycle_source(args)
    metadata_path = _metadata_path(args, source_path)
    metadata = _read_lifecycle_metadata(metadata_path)
    if metadata and _metadata_is_healthy(metadata, source_path):
        print(json.dumps({**metadata, "status": "running"}, indent=2, sort_keys=True))
        return 0
    print(f"[dashboard] no healthy persistent dashboard for {source_path}")
    return 1


def _persistent_stop(args: argparse.Namespace) -> int:
    _, source_path = _resolve_lifecycle_source(args)
    metadata_path = _metadata_path(args, source_path)
    metadata = _read_lifecycle_metadata(metadata_path)
    if not metadata:
        print(f"[dashboard] no persistent dashboard metadata: {metadata_path}")
        return 0
    if str(metadata.get("source_path", "")) != str(source_path.resolve()):
        print(
            f"[dashboard] metadata source does not match requested source: {metadata_path}",
            file=sys.stderr,
        )
        return 2
    try:
        pid = int(metadata.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    instance_id = str(metadata.get("instance_id", ""))
    if not _metadata_is_healthy(metadata, source_path):
        if not _pid_is_alive(pid):
            _remove_metadata_for_instance(metadata_path, instance_id)
            print(f"[dashboard] removed stale metadata: {metadata_path}")
            return 0
        print(
            f"[dashboard] refusing to stop live, unverified process pid={pid}",
            file=sys.stderr,
        )
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_metadata_for_instance(metadata_path, instance_id)
        print(f"[dashboard] persistent dashboard pid={pid} already stopped")
        return 0
    except PermissionError as exc:
        print(f"[dashboard] unable to stop process pid={pid}: {exc}", file=sys.stderr)
        return 2
    deadline = time.monotonic() + 5.0
    url = str(metadata.get("url", "")).rstrip("/")
    while time.monotonic() < deadline:
        health = _health_payload(url, timeout=0.2)
        if str(health.get("instance_id", "")) != instance_id:
            break
        time.sleep(0.05)
    else:
        print(
            f"[dashboard] process pid={pid} did not stop within 5 seconds",
            file=sys.stderr,
        )
        return 2
    _remove_metadata_for_instance(metadata_path, instance_id)
    print(f"[dashboard] stopped persistent dashboard pid={pid}")
    return 0


def _doe_output_dir(spec_path: Path) -> Path:
    spec = _read_yaml(spec_path)
    output = spec.get("output") if isinstance(spec.get("output"), dict) else {}
    raw = str(output.get("dir", "")).strip()
    if not raw:
        raise ValueError("DOE spec must define output.dir for live monitoring.")
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _models_in_doe(spec: dict[str, Any]) -> set[str]:
    models: set[str] = set()
    search_space = (
        spec.get("search_space") if isinstance(spec.get("search_space"), dict) else {}
    )
    raw_models = search_space.get("train.model.type", [])
    if not isinstance(raw_models, list):
        raw_models = [raw_models]
    models.update(
        str(value).strip().lower() for value in raw_models if str(value).strip()
    )
    defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
    default_model = str(defaults.get("train.model.type", "")).strip().lower()
    if default_model:
        models.add(default_model)
    return models


def _feature_inputs_in_doe(spec: dict[str, Any]) -> set[str]:
    feature_inputs: set[str] = set()
    search_space = (
        spec.get("search_space") if isinstance(spec.get("search_space"), dict) else {}
    )
    raw_inputs = search_space.get("pipeline.feature_input", [])
    if not isinstance(raw_inputs, list):
        raw_inputs = [raw_inputs]
    feature_inputs.update(
        str(value).strip().lower() for value in raw_inputs if str(value).strip()
    )
    defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
    default_input = str(defaults.get("pipeline.feature_input", "")).strip().lower()
    if default_input:
        feature_inputs.add(default_input)
    return feature_inputs


def _preflight_doe(spec_path: Path) -> tuple[bool, dict[str, Any]]:
    spec = _read_yaml(spec_path)
    models = _models_in_doe(spec)
    feature_inputs = _feature_inputs_in_doe(spec)
    needs_chemeleon_fp = "featurize.chemeleon_fp" in feature_inputs
    needs_tabpfn = "tabpfn" in models
    required_modules: dict[str, str] = {}
    if any(model.startswith("dl_") for model in models) or needs_chemeleon_fp or needs_tabpfn:
        required_modules["torch"] = "PyTorch models"
    if models & {"chemprop", "chemeleon"}:
        required_modules["chemprop"] = "Chemprop models"
        required_modules["lightning"] = "Chemprop Lightning trainer"
    elif needs_chemeleon_fp:
        required_modules["chemprop"] = "CheMeleon fingerprint generation"
    if needs_tabpfn:
        required_modules["tabpfn"] = "TabPFN 2.6 model"
    checks = {
        module: {
            "available": importlib.util.find_spec(module) is not None,
            "required_for": reason,
        }
        for module, reason in required_modules.items()
    }
    missing = sorted(
        module for module, result in checks.items() if not result["available"]
    )
    result: dict[str, Any] = {
        "models": sorted(models),
        "feature_inputs": sorted(feature_inputs),
        "modules": checks,
        "missing": missing,
        "runtime_probe": {},
    }
    if needs_chemeleon_fp:
        defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
        checkpoint_value = str(defaults.get("featurize.checkpoint", "")).strip()
        checkpoint_path = Path(checkpoint_value).expanduser() if checkpoint_value else None
        if checkpoint_path is not None and not checkpoint_path.is_absolute():
            checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
        checkpoint_available = bool(checkpoint_path and checkpoint_path.is_file())
        result["checkpoint"] = {
            "path": str(checkpoint_path) if checkpoint_path else "",
            "available": checkpoint_available,
            "required_for": "CheMeleon fingerprint generation",
        }
        if not checkpoint_available:
            result["missing"].append("featurize.checkpoint")
            missing = result["missing"]
    if missing:
        return False, result

    if (
        any(model.startswith("dl_") for model in models)
        or models & {"chemprop", "chemeleon"}
        or needs_chemeleon_fp
        or needs_tabpfn
    ):
        statements = [
            "import platform, torch",
            "print('python', platform.python_version())",
            "print('torch', torch.__version__)",
            "print('torch_threads', torch.get_num_threads())",
            "print('cuda', torch.cuda.is_available())",
            "print('mps', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())",
        ]
        if models & {"chemprop", "chemeleon"}:
            statements.extend(
                [
                    "import lightning, chemprop",
                    "from chemprop import data, featurizers, models, nn",
                    "print('lightning', lightning.__version__)",
                    "print('chemprop', chemprop.__version__)",
                ]
            )
        elif needs_chemeleon_fp:
            statements.extend(
                [
                    "import chemprop",
                    "from chemprop import data, featurizers, models, nn",
                    "print('chemprop', chemprop.__version__)",
                ]
            )
        if needs_tabpfn:
            statements.extend(
                [
                    "import tabpfn",
                    "from tabpfn import TabPFNClassifier",
                    "from tabpfn.constants import ModelVersion",
                    "assert ModelVersion.V2_6",
                    "assert hasattr(TabPFNClassifier, 'create_default_for_version')",
                    "print('tabpfn', getattr(tabpfn, '__version__', 'unknown'))",
                ]
            )
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "; ".join(statements)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            result["runtime_probe"] = {
                "return_code": probe.returncode,
                "stdout": probe.stdout.strip(),
                "stderr": probe.stderr.strip(),
            }
            if probe.returncode != 0:
                return False, result
        except subprocess.TimeoutExpired:
            result["runtime_probe"] = {
                "return_code": None,
                "stdout": "",
                "stderr": "Dependency import probe timed out after 90 seconds.",
            }
            return False, result
    return True, result


def _run_single_config(
    config_path: Path,
    launcher: LauncherState,
) -> None:
    launcher.update(
        phase="running",
        status="running",
        message=f"Running single config {config_path.name}.",
        started_at=_utc_now(),
    )
    env = os.environ.copy()
    env["CHEMLFLOW_CONFIG"] = str(config_path)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py")],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    status = "completed" if completed.returncode == 0 else "failed"
    launcher.update(
        phase=status,
        status=status,
        message=(
            "Single-config run completed successfully."
            if completed.returncode == 0
            else f"Single-config run exited with code {completed.returncode}."
        ),
        return_code=completed.returncode,
        ended_at=_utc_now(),
    )


def _run_doe(
    spec_path: Path,
    doe_dir: Path,
    launcher: LauncherState,
    args: argparse.Namespace,
) -> None:
    launcher.update(
        phase="preflight",
        status="running",
        message="Checking model runtime dependencies.",
        started_at=_utc_now(),
    )
    preflight_ok, preflight = _preflight_doe(spec_path)
    launcher.update(preflight=preflight)
    if not preflight_ok:
        missing = ", ".join(preflight["missing"])
        detail = (
            f"missing: {missing}"
            if missing
            else "the model runtime import probe failed"
        )
        launcher.update(
            phase="failed",
            status="failed",
            message=f"Dependency preflight failed; {detail}.",
            return_code=2,
            ended_at=_utc_now(),
        )
        return

    doe_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = doe_dir / "dashboard_launcher.log"
    launcher.update(
        phase="generating",
        message=f"Generating DOE from {spec_path.name}.",
    )
    with launcher_log.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[dashboard] generation start={_utc_now()} spec={spec_path}\n"
        )
        generated = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_doe.py"),
                "--doe",
                str(spec_path),
            ],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if generated.returncode != 0:
        launcher.update(
            phase="failed",
            status="failed",
            message=f"DOE generation failed with code {generated.returncode}; see {launcher_log}.",
            return_code=generated.returncode,
            ended_at=_utc_now(),
        )
        return

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_doe_local.py"),
        "--doe-dir",
        str(doe_dir),
        "--max-workers",
        str(max(1, int(args.max_workers))),
    ]
    if args.resume:
        command.append("--resume")
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.stop_on_failure:
        command.append("--stop-on-failure")
    if args.allow_shared_artifacts:
        command.append("--allow-shared-artifacts")
    launcher.update(
        phase="running",
        status="running",
        message=f"Running valid DOE cases with {max(1, int(args.max_workers))} local worker(s).",
    )
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    launcher.update(
        phase=status,
        status=status,
        message=(
            "DOE execution completed successfully."
            if completed.returncode == 0
            else f"DOE execution exited with code {completed.returncode}."
        ),
        return_code=completed.returncode,
        ended_at=_utc_now(),
    )


def _add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1."
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Bind port. Default: choose a free port."
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the dashboard in a browser."
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=20.0,
        help="Seconds without a run heartbeat before marking it stale. Default: 20.",
    )


def _add_lifecycle_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Single CheMLFlow runtime config YAML.")
    source.add_argument("--doe-dir", help="Generated DOE directory.")


def _add_metadata_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata-file",
        default="",
        help="Override persistent dashboard metadata path.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chemlflow_dashboard",
        description="Launch or attach to a truthful, read-only CheMLFlow live dashboard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch a run and its dashboard.")
    run_source = run_parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--config", help="Single CheMLFlow runtime config YAML.")
    run_source.add_argument("--doe", help="CheMLFlow DOE spec YAML.")
    run_parser.add_argument(
        "--max-workers", type=int, default=1, help="Local DOE workers. Default: 1."
    )
    run_parser.add_argument(
        "--resume", action="store_true", help="Reuse successful DOE cases."
    )
    run_parser.add_argument(
        "--limit", type=int, default=0, help="Run at most N valid DOE cases."
    )
    run_parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop a serial DOE after failure.",
    )
    run_parser.add_argument(
        "--allow-shared-artifacts",
        action="store_true",
        help="Allow parallel DOE configs that resolve to shared artifact directories.",
    )
    _add_server_arguments(run_parser)

    attach_parser = subparsers.add_parser(
        "attach", help="Open a dashboard without launching work."
    )
    attach_source = attach_parser.add_mutually_exclusive_group(required=True)
    attach_source.add_argument("--config", help="Single CheMLFlow runtime config YAML.")
    attach_source.add_argument("--doe-dir", help="Generated DOE directory.")
    _add_server_arguments(attach_parser)

    start_parser = subparsers.add_parser(
        "start", help="Start a persistent read-only dashboard."
    )
    _add_lifecycle_source(start_parser)
    _add_server_arguments(start_parser)
    _add_metadata_argument(start_parser)

    status_parser = subparsers.add_parser(
        "status", help="Check a persistent dashboard."
    )
    _add_lifecycle_source(status_parser)
    _add_metadata_argument(status_parser)

    stop_parser = subparsers.add_parser(
        "stop", help="Stop a verified persistent dashboard."
    )
    _add_lifecycle_source(stop_parser)
    _add_metadata_argument(stop_parser)

    serve_parser = subparsers.add_parser(
        "_serve", help="Internal process used by persistent dashboard start."
    )
    _add_lifecycle_source(serve_parser)
    _add_server_arguments(serve_parser)
    _add_metadata_argument(serve_parser)
    serve_parser.add_argument("--instance-id", required=True)
    serve_parser.add_argument("--log-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        return _persistent_start(args)
    if args.command == "status":
        return _persistent_status(args)
    if args.command == "stop":
        return _persistent_stop(args)
    if args.command == "_serve":
        return _persistent_serve(args)

    launcher = LauncherState(mode=args.command)

    worker: threading.Thread | None = None
    if args.command == "run" and args.config:
        config_path = _resolve_repo_path(args.config)
        if not config_path.is_file():
            raise SystemExit(f"Config not found: {config_path}")
        mode = "single"
        source_path = config_path
        worker = threading.Thread(
            target=_run_single_config,
            args=(config_path, launcher),
            name="chemlflow-single-runner",
            daemon=True,
        )
    elif args.command == "run":
        spec_path = _resolve_repo_path(args.doe)
        if not spec_path.is_file():
            raise SystemExit(f"DOE spec not found: {spec_path}")
        try:
            doe_dir = _doe_output_dir(spec_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SystemExit(str(exc)) from exc
        mode = "doe"
        source_path = doe_dir
        worker = threading.Thread(
            target=_run_doe,
            args=(spec_path, doe_dir, launcher, args),
            name="chemlflow-doe-runner",
            daemon=True,
        )
    elif args.config:
        source_path = _resolve_repo_path(args.config)
        if not source_path.is_file():
            raise SystemExit(f"Config not found: {source_path}")
        mode = "single"
    else:
        source_path = _resolve_repo_path(args.doe_dir)
        if not source_path.is_dir():
            raise SystemExit(f"DOE directory not found: {source_path}")
        mode = "doe"

    collector = StudyStateCollector(
        mode=mode,
        source_path=source_path,
        repo_root=REPO_ROOT,
        stale_after_seconds=args.stale_after,
        launcher_state=launcher.snapshot,
    )
    server = DashboardHTTPServer((args.host, int(args.port)), collector)
    print(f"[dashboard] read-only mission control: {server.url}")
    print(f"[dashboard] source: {source_path}")
    if worker is not None:
        worker.start()
    if not args.no_open:
        webbrowser.open(server.url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print(
            "\n[dashboard] stopping server; launched training processes are not forcibly terminated."
        )
    finally:
        server.stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
