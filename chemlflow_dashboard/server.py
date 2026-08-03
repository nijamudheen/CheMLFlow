"""Dependency-free HTTP/SSE server for the local CheMLFlow dashboard."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .state import StudyStateCollector

STATIC_DIR = Path(__file__).with_name("static")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_safe(payload),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _tail_text(path: Path, line_count: int) -> str:
    """Read a bounded tail without loading an arbitrarily large log."""

    line_count = min(5000, max(20, int(line_count)))
    block_size = 8192
    data = b""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and data.count(b"\n") <= line_count:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size) + data
    return b"\n".join(data.splitlines()[-line_count:]).decode("utf-8", errors="replace")


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        collector: StudyStateCollector,
        *,
        instance_id: str = "",
    ) -> None:
        self.collector = collector
        self.instance_id = instance_id
        self.stop_event = threading.Event()
        super().__init__(server_address, DashboardRequestHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        display_host = "127.0.0.1" if host in {"", "0.0.0.0"} else str(host)
        return f"http://{display_host}:{port}"

    def stop(self) -> None:
        self.stop_event.set()
        self.shutdown()

    def handle_error(self, request: Any, client_address: Any) -> None:
        _ = (request, client_address)
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/health":
            self._send_json(
                {
                    "status": "ok",
                    "read_only": True,
                    "instance_id": self.server.instance_id,
                    "pid": os.getpid(),
                    "source_path": str(self.server.collector.source_path.resolve()),
                }
            )
            return
        if parsed.path == "/api/v1/snapshot":
            self._send_json(self.server.collector.collect())
            return
        if parsed.path == "/api/v1/events":
            self._serve_events()
            return
        if parsed.path.startswith("/api/v1/cases/"):
            self._serve_case_route(parsed.path, parse_qs(parsed.query))
            return
        self._serve_static(parsed.path)

    def _serve_case_route(self, path: str, query: dict[str, list[str]]) -> None:
        parts = path.split("/")
        if len(parts) != 6:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        case_id = unquote(parts[4])
        action = parts[5]
        if action == "detail":
            detail = self.server.collector.case_detail(case_id)
            if detail is None:
                self._send_json({"error": "unknown case"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if action == "log":
            log_path = self.server.collector.log_path(case_id)
            if log_path is None:
                if self.server.collector.case_detail(case_id) is None:
                    self._send_json(
                        {"error": "unknown case"}, status=HTTPStatus.NOT_FOUND
                    )
                    return
                self._send_json(
                    {
                        "case_id": case_id,
                        "path": "",
                        "tail_lines": 0,
                        "available": False,
                        "text": "Log unavailable until this case starts.",
                    }
                )
                return
            raw_tail = (query.get("tail") or ["400"])[0]
            try:
                tail = int(raw_tail)
            except ValueError:
                tail = 400
            self._send_json(
                {
                    "case_id": case_id,
                    "path": str(log_path),
                    "tail_lines": min(5000, max(20, tail)),
                    "available": True,
                    "text": _tail_text(log_path, tail),
                }
            )
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._security_headers()
        self.end_headers()
        previous_digest = ""
        last_emit = 0.0
        try:
            while not self.server.stop_event.wait(1.0):
                snapshot = self.server.collector.collect()
                payload = _json_bytes(snapshot)
                digest_source = dict(snapshot)
                digest_source.pop("generated_at", None)
                digest = hashlib.sha256(_json_bytes(digest_source)).hexdigest()
                now = time.monotonic()
                if digest != previous_digest or now - last_emit >= 15.0:
                    event = b"event: snapshot\ndata: " + payload + b"\n\n"
                    self.wfile.write(event)
                    self.wfile.flush()
                    previous_digest = digest
                    last_emit = now
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _serve_static(self, request_path: str) -> None:
        relative = (
            "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        )
        candidate = (STATIC_DIR / relative).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = (
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        try:
            data = candidate.read_bytes()
        except OSError:
            self._send_json({"error": "asset unavailable"}, status=HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'",
        )
