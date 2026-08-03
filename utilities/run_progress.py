"""Atomic, best-effort progress telemetry for CheMLFlow runs.

The dashboard is deliberately a reader of run artifacts.  This reporter is the
single writer for ``run_progress.json`` and must never be able to fail a
scientific run: telemetry write failures are logged and retried on the next
update or heartbeat.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Return a strict-JSON representation without importing model libraries."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # NumPy and similar scalar types generally expose ``item``.  Keep this
    # duck-typed so importing the reporter never pulls a training framework in.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


class RunProgressReporter:
    """Write truthful run and training progress to ``run_progress.json``.

    Known totals are reported only when the training adapter provides one.
    Opaque estimators remain explicitly indeterminate while heartbeats prove
    that the process is still alive.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config_path: str | Path,
        pipeline_nodes: list[str] | tuple[str, ...],
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self.path = Path(run_dir) / "run_progress.json"
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_interval = max(0.25, float(heartbeat_interval_seconds))
        self._thread: threading.Thread | None = None
        now = _utc_now()
        nodes = [str(node) for node in pipeline_nodes]
        self._payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "start_time": now,
            "updated_at": now,
            "last_heartbeat_at": now,
            "end_time": None,
            "pid": os.getpid(),
            "config_path": str(config_path),
            "run_dir": str(Path(run_dir)),
            "pipeline": {
                "total": len(nodes),
                "completed": 0,
                "completed_nodes": [],
                "current": None,
                "current_index": None,
                "nodes": nodes,
            },
            "training": {
                "active": False,
                "scopes": {},
            },
            "error": None,
        }

    def start(self) -> "RunProgressReporter":
        """Write the initial snapshot and start the heartbeat thread."""

        self._write()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="chemlflow-progress-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe copy, primarily for tests."""

        with self._lock:
            return json.loads(json.dumps(_json_safe(self._payload), allow_nan=False))

    def heartbeat(self) -> None:
        with self._lock:
            if self._payload["state"] != "running":
                return
            now = _utc_now()
            self._payload["updated_at"] = now
            self._payload["last_heartbeat_at"] = now
            self._write_locked()

    def node_started(self, node_name: str) -> None:
        with self._lock:
            pipeline = self._payload["pipeline"]
            nodes = pipeline["nodes"]
            try:
                current_index = nodes.index(node_name)
            except ValueError:
                current_index = None
            pipeline["current"] = str(node_name)
            pipeline["current_index"] = current_index
            self._touch_and_write_locked()

    def node_completed(self, node_name: str) -> None:
        with self._lock:
            pipeline = self._payload["pipeline"]
            completed_nodes = pipeline["completed_nodes"]
            if node_name not in completed_nodes:
                completed_nodes.append(str(node_name))
            pipeline["completed"] = len(completed_nodes)
            if pipeline.get("current") == node_name:
                pipeline["current"] = None
                pipeline["current_index"] = None
            self._touch_and_write_locked()

    def training_started(
        self,
        scope: str,
        *,
        unit: str,
        total: int | None = None,
        phase: str = "training",
        message: str = "",
    ) -> None:
        with self._lock:
            now = _utc_now()
            normalized_total = self._normalize_total(total)
            training = self._payload["training"]
            training["active"] = True
            training["scopes"][str(scope)] = {
                "status": "running",
                "unit": str(unit),
                "current": 0 if normalized_total is not None else None,
                "total": normalized_total,
                "indeterminate": normalized_total is None,
                "phase": str(phase),
                "message": str(message),
                "metrics": {},
                "start_time": now,
                "updated_at": now,
                "end_time": None,
            }
            self._touch_and_write_locked(now=now)

    def training_indeterminate(
        self,
        scope: str = "fit",
        *,
        unit: str = "fit",
        phase: str = "training",
        message: str = "Estimator does not expose step progress.",
    ) -> None:
        self.training_started(
            scope,
            unit=unit,
            total=None,
            phase=phase,
            message=message,
        )

    def training_update(
        self,
        scope: str,
        *,
        current: int | None = None,
        total: int | None = None,
        phase: str | None = None,
        message: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            now = _utc_now()
            training = self._payload["training"]
            scopes = training["scopes"]
            if scope not in scopes:
                scopes[scope] = {
                    "status": "running",
                    "unit": "step",
                    "current": None,
                    "total": None,
                    "indeterminate": True,
                    "phase": "training",
                    "message": "",
                    "metrics": {},
                    "start_time": now,
                    "updated_at": now,
                    "end_time": None,
                }
            item = scopes[scope]
            item["status"] = "running"
            if current is not None:
                item["current"] = max(0, int(current))
            if total is not None:
                item["total"] = self._normalize_total(total)
            item["indeterminate"] = item.get("total") is None
            if phase is not None:
                item["phase"] = str(phase)
            if message is not None:
                item["message"] = str(message)
            if metrics is not None:
                item["metrics"] = _json_safe(dict(metrics))
            item["updated_at"] = now
            training["active"] = True
            self._touch_and_write_locked(now=now)

    def training_scope_finished(
        self,
        scope: str,
        *,
        status: str = "completed",
        message: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            now = _utc_now()
            training = self._payload["training"]
            item = training["scopes"].get(scope)
            if item is None:
                return
            item["status"] = str(status)
            item["end_time"] = now
            item["updated_at"] = now
            if message is not None:
                item["message"] = str(message)
            if metrics is not None:
                item["metrics"] = _json_safe(dict(metrics))
            training["active"] = any(
                value.get("status") == "running"
                for value in training["scopes"].values()
            )
            self._touch_and_write_locked(now=now)

    def success(self) -> None:
        self._finish("success")

    def failed(
        self, *, exception_type: str, message: str, failed_node: str | None
    ) -> None:
        self._finish(
            "failed",
            error={
                "exception_type": str(exception_type),
                "message": str(message),
                "failed_node": failed_node,
            },
        )

    def close(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))

    def _finish(self, state: str, error: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            now = _utc_now()
            self._payload["state"] = state
            self._payload["updated_at"] = now
            self._payload["last_heartbeat_at"] = now
            self._payload["end_time"] = now
            self._payload["training"]["active"] = False
            self._payload["error"] = _json_safe(error) if error else None
            self._write_locked()
        self.close()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            self.heartbeat()

    @staticmethod
    def _normalize_total(total: int | None) -> int | None:
        if total is None:
            return None
        value = int(total)
        return value if value > 0 else None

    def _touch_and_write_locked(self, *, now: str | None = None) -> None:
        timestamp = now or _utc_now()
        self._payload["updated_at"] = timestamp
        self._payload["last_heartbeat_at"] = timestamp
        self._write_locked()

    def _write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            text = json.dumps(
                _json_safe(self._payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except OSError as exc:
            logging.warning(
                "Unable to write run progress telemetry %s: %s", self.path, exc
            )
