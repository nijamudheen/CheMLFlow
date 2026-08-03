"""Normalize single-run and DOE artifacts into one dashboard snapshot."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

SNAPSHOT_SCHEMA_VERSION = 1
TERMINAL_STATES = {"completed", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_path(
    raw: Any, *, repo_root: Path, relative_to: Path | None = None
) -> Path:
    path = Path(str(raw or ""))
    if path.is_absolute():
        return path
    candidates = [repo_root / path]
    if relative_to is not None:
        candidates.append(relative_to / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _config_run_dir(config: dict[str, Any], *, repo_root: Path) -> Path:
    global_cfg = config.get("global") if isinstance(config.get("global"), dict) else {}
    run_dir = str(global_cfg.get("run_dir", "")).strip()
    if run_dir:
        return _resolve_path(run_dir, repo_root=repo_root)
    runs = global_cfg.get("runs") if isinstance(global_cfg.get("runs"), dict) else {}
    if runs.get("enabled") and runs.get("id"):
        return repo_root / "runs" / str(runs["id"])
    return repo_root / "results"


def _single_run_dir(
    config: dict[str, Any], *, repo_root: Path, config_path: Path
) -> Path:
    """Resolve timestamped single runs by matching their status artifact."""

    global_cfg = config.get("global") if isinstance(config.get("global"), dict) else {}
    if str(global_cfg.get("run_dir", "")).strip():
        return _config_run_dir(config, repo_root=repo_root)
    runs = global_cfg.get("runs") if isinstance(global_cfg.get("runs"), dict) else {}
    if not runs.get("enabled") or runs.get("id"):
        return _config_run_dir(config, repo_root=repo_root)
    runs_root = repo_root / "runs"
    matches: list[tuple[float, Path]] = []
    try:
        candidates = list(runs_root.iterdir())
    except OSError:
        candidates = []
    for candidate in candidates:
        status_path = candidate / "run_status.json"
        status = _read_json(status_path)
        raw_status_config = str(status.get("config_path", "")).strip()
        if not raw_status_config:
            continue
        resolved_status_config = _resolve_path(raw_status_config, repo_root=repo_root)
        try:
            same_config = resolved_status_config.resolve() == config_path.resolve()
        except OSError:
            same_config = str(resolved_status_config) == str(config_path)
        if not same_config:
            continue
        try:
            modified = status_path.stat().st_mtime
        except OSError:
            modified = 0.0
        matches.append((modified, candidate))
    return max(matches, default=(0.0, repo_root / "results"), key=lambda item: item[0])[
        1
    ]


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: Any) -> float | None:
    timestamp = _parse_time(value)
    if timestamp is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


def _belongs_to_attempt(artifact_start: Any, attempt_start: Any) -> bool:
    attempt = _parse_time(attempt_start)
    artifact = _parse_time(artifact_start)
    if attempt is None:
        return True
    if artifact is None:
        return False
    return artifact >= attempt - timedelta(seconds=2)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _study_timeline(cases: list[dict[str, Any]]) -> dict[str, str | None]:
    """Return the artifact-backed wall-clock bounds for a study."""

    starts = [
        parsed
        for case in cases
        if (parsed := _parse_time(case.get("start_time"))) is not None
    ]
    ends = [
        parsed
        for case in cases
        if (parsed := _parse_time(case.get("end_time"))) is not None
    ]
    started_at = min(starts).isoformat() if starts else None
    study_is_terminal = bool(cases) and all(
        case.get("status") in TERMINAL_STATES for case in cases
    )
    ended_at = (
        max(ends).isoformat()
        if study_is_terminal and len(ends) == len(cases)
        else None
    )
    return {"started_at": started_at, "ended_at": ended_at}


def _metric_aliases(metric: str) -> tuple[str, ...]:
    normalized = metric.strip().lower()
    aliases = {
        "auroc": ("auroc", "auc"),
        "auc": ("auc", "auroc"),
        "roc_auc": ("roc_auc", "auc", "auroc"),
        "rmse": ("rmse",),
        "r2": ("r2",),
        "mae": ("mae",),
        "auprc": ("auprc",),
        "accuracy": ("accuracy",),
        "f1": ("f1",),
    }
    return aliases.get(normalized, (normalized,))


def _extract_metric(metrics: dict[str, Any], metric: str) -> float | None:
    for key in _metric_aliases(metric):
        value = _finite_number(metrics.get(key))
        if value is not None:
            return value
    return None


def _latest_execution_attempts(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("record_type", "")).lower() != "execution_attempt":
            continue
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            attempts[case_id] = row
    return attempts


def _model_type(config: dict[str, Any], factors: dict[str, Any]) -> str:
    factor_value = factors.get("train.model.type")
    if factor_value:
        return str(factor_value)
    train_cfg = config.get("train") if isinstance(config.get("train"), dict) else {}
    model_cfg = (
        train_cfg.get("model") if isinstance(train_cfg.get("model"), dict) else {}
    )
    return str(model_cfg.get("type", "unknown"))


def _feature_input(config: dict[str, Any], factors: dict[str, Any]) -> str:
    value = factors.get("pipeline.feature_input")
    if value:
        return str(value)
    pipeline = (
        config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
    )
    return str(pipeline.get("feature_input", ""))


def _find_metrics(run_dir: Path, model_type: str) -> tuple[dict[str, Any], str]:
    names = [f"{model_type}_metrics.json"]
    if model_type in {"chemprop", "chemeleon"}:
        names.insert(0, "chemprop_metrics.json")
    for name in names:
        path = run_dir / name
        if path.is_file():
            return _read_json(path), str(path)
    try:
        candidates = sorted(
            path
            for path in run_dir.glob("*_metrics.json")
            if "split_metrics" not in path.name
        )
    except OSError:
        candidates = []
    if candidates:
        return _read_json(candidates[0]), str(candidates[0])
    return {}, ""


def _split_metrics(
    run_dir: Path, metrics: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    configured = str(metrics.get("split_metrics_path", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    try:
        candidates.extend(sorted(run_dir.glob("*_split_metrics.json")))
    except OSError:
        pass
    for path in candidates:
        if not path.is_absolute():
            path = run_dir / path
        if path.is_file():
            return _read_json(path), str(path)
    return {}, ""


def _status_for_case(
    *,
    manifest_status: str,
    execution: dict[str, Any],
    run_status: dict[str, Any],
    progress: dict[str, Any],
    stale_after_seconds: float,
) -> tuple[str, str, float | None]:
    if manifest_status != "valid":
        return "skipped_validation", "DOE compatibility or validation skip", None

    execution_state = str(execution.get("state", "")).strip().upper()
    artifact_state = str(run_status.get("status", "")).strip().lower()
    progress_state = str(progress.get("state", "")).strip().lower()
    attempt_start = execution.get("start_time")
    progress_is_current = _belongs_to_attempt(progress.get("start_time"), attempt_start)
    status_is_current = _belongs_to_attempt(run_status.get("start_time"), attempt_start)
    effective_progress_state = progress_state if progress_is_current else ""
    effective_artifact_state = artifact_state if status_is_current else ""
    freshness = (
        _age_seconds(progress.get("last_heartbeat_at") or progress.get("updated_at"))
        if progress_is_current
        else None
    )

    if execution_state == "FAILED":
        reason = str(
            execution.get("failure_reason")
            or (run_status.get("message") if status_is_current else "")
            or "run failed"
        )
        return "failed", reason, freshness
    if execution_state == "RUNNING":
        if effective_artifact_state == "failed" or effective_progress_state == "failed":
            reason = str(
                (run_status.get("message") if status_is_current else "") or "run failed"
            )
            return "failed", reason, freshness
        if (
            effective_artifact_state == "success"
            or effective_progress_state == "success"
        ):
            return "completed", "success artifact observed", freshness
        if freshness is not None and freshness > stale_after_seconds:
            return (
                "stale",
                f"no telemetry heartbeat for {int(freshness)} seconds",
                freshness,
            )
        launch_age = _age_seconds(attempt_start)
        if (
            freshness is None
            and launch_age is not None
            and launch_age > max(60.0, stale_after_seconds * 3)
        ):
            return (
                "stale",
                "runner says active but no current telemetry artifact appeared",
                launch_age,
            )
        return "running", "active local execution attempt observed", freshness

    if (
        execution_state == "SKIPPED"
        and str(execution.get("failure_reason", "")) == "already_successful"
    ):
        return "completed", "resume reused an existing successful run", freshness
    if effective_artifact_state == "failed" or effective_progress_state == "failed":
        reason = str(
            (run_status.get("message") if status_is_current else "") or "run failed"
        )
        return "failed", reason, freshness
    if effective_artifact_state == "success" or effective_progress_state == "success":
        return "completed", "success artifact observed", freshness
    if execution_state == "COMPLETED":
        return (
            "failed",
            "local process completed without a success run_status artifact",
            freshness,
        )
    if execution_state == "DRY_RUN":
        return "queued", "dry-run only; no training process launched", freshness

    is_running = (
        effective_artifact_state == "running" or effective_progress_state == "running"
    )
    if is_running:
        if freshness is not None and freshness > stale_after_seconds:
            return (
                "stale",
                f"no telemetry heartbeat for {int(freshness)} seconds",
                freshness,
            )
        if freshness is None:
            launch_age = _age_seconds(execution.get("start_time"))
            if launch_age is not None and launch_age > max(
                60.0, stale_after_seconds * 3
            ):
                return (
                    "stale",
                    "runner says active but no telemetry artifact appeared",
                    launch_age,
                )
        return "running", "active process or run artifact observed", freshness
    return "queued", "valid case has not started", freshness


class StudyStateCollector:
    """Collect dashboard state from files without mutating a study."""

    def __init__(
        self,
        *,
        mode: str,
        source_path: str | Path,
        repo_root: str | Path,
        stale_after_seconds: float = 20.0,
        launcher_state: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if mode not in {"single", "doe"}:
            raise ValueError("mode must be 'single' or 'doe'")
        self.mode = mode
        self.repo_root = Path(repo_root).resolve()
        self.source_path = Path(source_path).resolve()
        self.stale_after_seconds = max(5.0, float(stale_after_seconds))
        self._launcher_state = launcher_state

    def collect(self) -> dict[str, Any]:
        if self.mode == "single":
            snapshot = self._collect_single()
        else:
            snapshot = self._collect_doe()
        snapshot["launcher"] = self._launcher_state() if self._launcher_state else {}
        snapshot["generated_at"] = _utc_now()
        return snapshot

    def case_detail(self, case_id: str) -> dict[str, Any] | None:
        snapshot = self.collect()
        case = next(
            (item for item in snapshot["cases"] if item["case_id"] == case_id), None
        )
        if case is None:
            return None
        run_dir = Path(case["run_dir"])
        detail = dict(case)
        detail["artifacts"] = self._artifact_inventory(run_dir)
        detail["split_metrics"] = _split_metrics(run_dir, case.get("metrics", {}))[0]
        detail["issues"] = case.get("issues", [])
        return detail

    def log_path(self, case_id: str) -> Path | None:
        snapshot = self.collect()
        case = next(
            (item for item in snapshot["cases"] if item["case_id"] == case_id), None
        )
        if case is None:
            return None
        candidates = [case.get("log_path"), str(Path(case["run_dir"]) / "run.log")]
        for raw in candidates:
            if raw and Path(str(raw)).is_file():
                return Path(str(raw))
        return None

    def _collect_single(self) -> dict[str, Any]:
        config = _read_yaml(self.source_path)
        run_dir = _single_run_dir(
            config,
            repo_root=self.repo_root,
            config_path=self.source_path,
        )
        global_cfg = (
            config.get("global") if isinstance(config.get("global"), dict) else {}
        )
        task_type = str(global_cfg.get("task_type", "unknown"))
        primary_metric = "auc" if task_type == "classification" else "r2"
        case = self._build_case(
            record={
                "case_id": self.source_path.stem,
                "parent_case_id": self.source_path.stem,
                "status": "valid",
                "factors": {},
                "execution_label": "single",
                "task_type": task_type,
                "config_path": str(self.source_path),
            },
            config_path=self.source_path,
            run_dir=run_dir,
            execution={},
            primary_metric=primary_metric,
        )
        return self._assemble_snapshot(
            name=str(global_cfg.get("run_name") or self.source_path.stem),
            task_type=task_type,
            primary_metric=primary_metric,
            optimize="max",
            cases=[case],
            skipped_records=[],
            parent_records=[],
        )

    def _collect_doe(self) -> dict[str, Any]:
        doe_dir = self.source_path
        summary = _read_json(doe_dir / "summary.json")
        manifest_rows = _read_jsonl(doe_dir / "manifest.jsonl")
        parent_rows = _read_jsonl(doe_dir / "parent_manifest.jsonl")
        attempts = _latest_execution_attempts(
            _read_jsonl(doe_dir / "execution_manifest.jsonl")
        )
        selection = (
            summary.get("selection")
            if isinstance(summary.get("selection"), dict)
            else {}
        )
        primary_metric = str(selection.get("primary_metric", "auc"))
        optimize = str(selection.get("optimize", "max")).lower()
        cases: list[dict[str, Any]] = []
        skipped_records: list[dict[str, Any]] = []
        for record in manifest_rows:
            manifest_status = str(record.get("status", "")).lower()
            if manifest_status != "valid":
                skipped_records.append(record)
                continue
            config_path = _resolve_path(
                record.get("config_path"),
                repo_root=self.repo_root,
                relative_to=doe_dir,
            )
            config = _read_yaml(config_path)
            run_dir = _config_run_dir(config, repo_root=self.repo_root)
            cases.append(
                self._build_case(
                    record=record,
                    config_path=config_path,
                    run_dir=run_dir,
                    execution=attempts.get(str(record.get("case_id", "")), {}),
                    primary_metric=primary_metric,
                    config=config,
                )
            )
        return self._assemble_snapshot(
            name=str(summary.get("dataset_probe", {}).get("name") or doe_dir.name),
            task_type=str(summary.get("task_type", "unknown")),
            primary_metric=primary_metric,
            optimize=optimize,
            cases=cases,
            skipped_records=skipped_records,
            parent_records=parent_rows,
            source_summary=summary,
        )

    def _build_case(
        self,
        *,
        record: dict[str, Any],
        config_path: Path,
        run_dir: Path,
        execution: dict[str, Any],
        primary_metric: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = config if config is not None else _read_yaml(config_path)
        factors = (
            record.get("factors") if isinstance(record.get("factors"), dict) else {}
        )
        progress = _read_json(run_dir / "run_progress.json")
        run_status = _read_json(run_dir / "run_status.json")
        attempt_start = execution.get("start_time")
        progress_is_current = _belongs_to_attempt(
            progress.get("start_time"), attempt_start
        )
        status_is_current = _belongs_to_attempt(
            run_status.get("start_time"), attempt_start
        )
        status, status_reason, freshness = _status_for_case(
            manifest_status=str(record.get("status", "valid")).lower(),
            execution=execution,
            run_status=run_status,
            progress=progress,
            stale_after_seconds=self.stale_after_seconds,
        )
        model_type = _model_type(config, factors)
        metrics, metrics_path = _find_metrics(run_dir, model_type)
        if (
            str(execution.get("state", "")).strip().upper()
            in {"RUNNING", "COMPLETED", "FAILED", "DRY_RUN"}
            and metrics_path
        ):
            attempt_started = _parse_time(attempt_start)
            try:
                metrics_modified = datetime.fromtimestamp(
                    Path(metrics_path).stat().st_mtime,
                    tz=timezone.utc,
                )
            except OSError:
                metrics_modified = None
            if (
                attempt_started is not None
                and metrics_modified is not None
                and metrics_modified < attempt_started - timedelta(seconds=2)
            ):
                metrics = {}
                metrics_path = ""
        metric_value = _extract_metric(metrics, primary_metric)
        current_progress = progress if progress_is_current else {}
        current_run_status = run_status if status_is_current else {}
        start_time = (
            current_progress.get("start_time")
            or current_run_status.get("start_time")
            or attempt_start
        )
        end_time = (
            current_progress.get("end_time")
            or current_run_status.get("end_time")
            or execution.get("end_time")
        )
        elapsed = _finite_number(execution.get("elapsed_seconds"))
        if elapsed is None:
            started = _parse_time(start_time)
            ended = _parse_time(end_time) or (
                datetime.now(timezone.utc)
                if started and status in {"running", "stale"}
                else None
            )
            elapsed = (
                max(0.0, (ended - started).total_seconds())
                if started and ended
                else None
            )
        split_meta = _read_json(run_dir / "split_meta.json")
        return {
            "case_id": str(record.get("case_id") or config_path.stem),
            "parent_case_id": str(
                record.get("parent_case_id")
                or record.get("case_id")
                or config_path.stem
            ),
            "execution_label": str(record.get("execution_label", "single")),
            "status": status,
            "status_reason": status_reason,
            "model_type": model_type,
            "feature_input": _feature_input(config, factors),
            "split_strategy": str(
                factors.get("split.strategy")
                or (config.get("split", {}) or {}).get("strategy", "")
            ),
            "fold_index": (record.get("execution_factors") or {}).get(
                "split.cv.fold_index"
            ),
            "repeat_index": (record.get("execution_factors") or {}).get(
                "split.cv.repeat_index"
            ),
            "task_type": str(record.get("task_type", "")),
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "log_path": str(execution.get("log_path", "")),
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_seconds": elapsed,
            "freshness_seconds": freshness,
            "progress": current_progress,
            "run_status": current_run_status,
            "metric_name": primary_metric,
            "metric_value": metric_value,
            "metrics": metrics,
            "metrics_path": metrics_path,
            "split_meta": split_meta,
            "factors": factors,
            "issues": record.get("issues", []),
        }

    def _assemble_snapshot(
        self,
        *,
        name: str,
        task_type: str,
        primary_metric: str,
        optimize: str,
        cases: list[dict[str, Any]],
        skipped_records: list[dict[str, Any]],
        parent_records: list[dict[str, Any]],
        source_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        counts = {
            key: 0 for key in ("queued", "running", "completed", "failed", "stale")
        }
        for case in cases:
            counts[case["status"]] = counts.get(case["status"], 0) + 1
        valid = len(cases)
        settled = counts["completed"] + counts["failed"]
        active_elapsed = [
            case["elapsed_seconds"]
            for case in cases
            if case.get("elapsed_seconds") is not None
        ]
        parents = self._build_parents(parent_records, cases, primary_metric)
        leaderboard = [
            parent
            for parent in parents
            if parent["status"] == "completed" and parent.get("metric_mean") is not None
        ]
        reverse = optimize not in {"min", "minimize"}
        leaderboard.sort(key=lambda item: item["metric_mean"], reverse=reverse)
        for index, item in enumerate(leaderboard, start=1):
            item["rank"] = index
            item["provisional"] = settled < valid
        skip_reasons: dict[str, int] = {}
        for record in skipped_records:
            issues = (
                record.get("issues") if isinstance(record.get("issues"), list) else []
            )
            if not issues:
                skip_reasons["UNKNOWN"] = skip_reasons.get("UNKNOWN", 0) + 1
            for issue in issues:
                code = (
                    str(issue.get("code", "UNKNOWN"))
                    if isinstance(issue, dict)
                    else "UNKNOWN"
                )
                skip_reasons[code] = skip_reasons.get(code, 0) + 1
        source_summary = source_summary or {}
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "mode": self.mode,
            "study": {
                "name": name,
                "source_path": str(self.source_path),
                "task_type": task_type,
                "primary_metric": primary_metric,
                "optimize": optimize,
                "backend": "local",
                "read_only": True,
            },
            "timeline": _study_timeline(cases),
            "summary": {
                "valid_cases": valid,
                "skipped_cases": len(skipped_records),
                "total_cases": valid + len(skipped_records),
                "settled_cases": settled,
                "progress_fraction": (settled / valid) if valid else 0.0,
                "active_workers": counts["running"] + counts["stale"],
                "observed_wall_seconds": (
                    max(active_elapsed) if active_elapsed else None
                ),
                **counts,
            },
            "cases": sorted(cases, key=lambda item: item["case_id"]),
            "parents": parents,
            "leaderboard": leaderboard,
            "skips": {
                "count": len(skipped_records),
                "by_reason": dict(sorted(skip_reasons.items())),
                "records": skipped_records[:200],
            },
            "source_summary": source_summary,
        }

    @staticmethod
    def _build_parents(
        parent_records: list[dict[str, Any]],
        cases: list[dict[str, Any]],
        primary_metric: str,
    ) -> list[dict[str, Any]]:
        cases_by_parent: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            cases_by_parent.setdefault(case["parent_case_id"], []).append(case)
        if not parent_records:
            parent_records = [
                {
                    "case_id": parent_id,
                    "parent_case_id": parent_id,
                    "model_type": children[0]["model_type"] if children else "unknown",
                    "factors": children[0].get("factors", {}) if children else {},
                }
                for parent_id, children in cases_by_parent.items()
            ]
        parents: list[dict[str, Any]] = []
        for record in parent_records:
            parent_id = str(record.get("parent_case_id") or record.get("case_id", ""))
            children = cases_by_parent.get(parent_id, [])
            if not children:
                continue
            statuses = [child["status"] for child in children]
            if all(status == "completed" for status in statuses):
                status = "completed"
            elif any(status == "failed" for status in statuses):
                status = "failed"
            elif any(status == "stale" for status in statuses):
                status = "stale"
            elif any(status == "running" for status in statuses) or any(
                status == "completed" for status in statuses
            ):
                status = "running"
            else:
                status = "queued"
            values = [
                child["metric_value"]
                for child in children
                if child.get("metric_value") is not None
            ]
            metric_mean = (
                sum(values) / len(values)
                if status == "completed" and len(values) == len(children)
                else None
            )
            metric_std = (
                _population_std(values)
                if status == "completed" and len(values) == len(children)
                else None
            )
            factors = (
                record.get("factors") if isinstance(record.get("factors"), dict) else {}
            )
            parents.append(
                {
                    "parent_case_id": parent_id,
                    "status": status,
                    "model_type": str(
                        record.get("model_type") or children[0]["model_type"]
                    ),
                    "feature_input": str(
                        factors.get("pipeline.feature_input")
                        or children[0]["feature_input"]
                    ),
                    "metric_name": primary_metric,
                    "metric_mean": metric_mean,
                    "metric_std": metric_std,
                    "completed_folds": statuses.count("completed"),
                    "total_folds": len(children),
                    "child_case_ids": [child["case_id"] for child in children],
                    "folds": [
                        {
                            "case_id": child["case_id"],
                            "fold_index": child["fold_index"],
                            "status": child["status"],
                            "metric_value": child["metric_value"],
                        }
                        for child in children
                    ],
                    "factors": factors,
                }
            )
        return sorted(parents, key=lambda item: item["parent_case_id"])

    @staticmethod
    def _artifact_inventory(run_dir: Path, limit: int = 200) -> list[dict[str, Any]]:
        if not run_dir.is_dir():
            return []
        artifacts: list[dict[str, Any]] = []
        try:
            paths = sorted(path for path in run_dir.rglob("*") if path.is_file())
        except OSError:
            return []
        for path in paths[:limit]:
            try:
                stat = path.stat()
            except OSError:
                continue
            artifacts.append(
                {
                    "name": str(path.relative_to(run_dir)),
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return artifacts
