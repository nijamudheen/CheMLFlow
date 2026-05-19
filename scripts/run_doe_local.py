#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency is required for runtime use
    raise SystemExit("PyYAML is required to run local DOE configs.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run valid CheMLFlow DOE execution-child configs locally from a generated "
            "DOE directory. This is the local backend counterpart to Slurm submission."
        )
    )
    parser.add_argument(
        "--doe-dir",
        required=True,
        help="Generated DOE directory containing manifest.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest path. Default: <doe-dir>/manifest.jsonl.",
    )
    parser.add_argument(
        "--output-manifest",
        default="",
        help="Execution attempt manifest path. Default: <doe-dir>/execution_manifest.jsonl.",
    )
    parser.add_argument(
        "--logs-dir",
        default="",
        help="Directory for per-case stdout/stderr logs. Default: <doe-dir>/local_logs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run main.py. Default: current interpreter.",
    )
    parser.add_argument(
        "--main",
        default=str(REPO_ROOT / "main.py"),
        help="Path to CheMLFlow main.py. Default: repo main.py.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of local cases to run concurrently. Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs whose run_status.json already reports status=success.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many valid configs after filtering. Default: no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned local execution rows without running main.py.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop scheduling additional work after the first failed local run.",
    )
    parser.add_argument(
        "--allow-shared-artifacts",
        action="store_true",
        help=(
            "Allow parallel runs even when generated configs share global.run_dir or "
            "global.base_dir. By default this is refused for --max-workers > 1."
        ),
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_valid_manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        if str(record.get("status", "")).lower() != "valid":
            continue
        record_type = str(record.get("record_type", "execution_child")).strip().lower()
        if record_type == "parent":
            continue
        if record.get("config_path"):
            records.append(record)
    return records


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must parse to a mapping/object: {path}")
    return loaded


def _resolve_config_path(raw_path: str, *, doe_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path, doe_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _run_dir_for_config(config_path: Path) -> Path | None:
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
    run_dir = str(global_cfg.get("run_dir", "")).strip()
    return Path(run_dir) if run_dir else None


def _artifact_dirs_for_config(config_path: Path) -> tuple[Path | None, Path | None]:
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
    run_dir = str(global_cfg.get("run_dir", "")).strip()
    base_dir = str(global_cfg.get("base_dir", "")).strip()
    return (Path(run_dir) if run_dir else None, Path(base_dir) if base_dir else None)


def _load_run_status(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / "run_status.json"
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _case_name(record: dict[str, Any], config_path: Path) -> str:
    value = str(record.get("case_id", "")).strip()
    return value or config_path.stem


def _log_path_for(record: dict[str, Any], config_path: Path, logs_dir: Path) -> Path:
    return logs_dir / f"{_case_name(record, config_path)}.log"


def _base_output_record(
    record: dict[str, Any],
    *,
    config_path: Path,
    run_dir: Path | None,
    log_path: Path,
    sequence: int,
    state: str,
    return_code: int | None,
    start_time: str,
    end_time: str,
    elapsed_seconds: float,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    run_status = _load_run_status(run_dir)
    status_value = str((run_status or {}).get("status", "")).strip()
    return {
        "record_type": "execution_attempt",
        "backend": "local",
        "execution_id": f"local_{sequence:05d}",
        "local_job_id": f"local_{sequence:05d}",
        "scheduler_job_id": "",
        "case_id": record.get("case_id", ""),
        "parent_case_id": record.get("parent_case_id", ""),
        "scientific_config_id": record.get("scientific_config_id", ""),
        "execution_label": record.get("execution_label", ""),
        "profile": record.get("profile", ""),
        "task_type": record.get("task_type", ""),
        "factors": record.get("factors", {}),
        "execution_factors": record.get("execution_factors", {}),
        "config_hash": record.get("config_hash", ""),
        "config_fingerprint": record.get("config_fingerprint", ""),
        "config_path": str(config_path),
        "run_dir": str(run_dir) if run_dir else "",
        "run_status_path": str(run_dir / "run_status.json") if run_dir else "",
        "run_status": status_value,
        "state": state,
        "return_code": return_code,
        "exit_code": "" if return_code is None else str(return_code),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "log_path": str(log_path),
        "failure_reason": failure_reason or "",
        "cwd": str(REPO_ROOT),
    }


def _run_one(
    record: dict[str, Any],
    *,
    sequence: int,
    doe_dir: Path,
    logs_dir: Path,
    python_bin: str,
    main_path: Path,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = _resolve_config_path(str(record.get("config_path", "")), doe_dir=doe_dir)
    run_dir = _run_dir_for_config(config_path)
    log_path = _log_path_for(record, config_path, logs_dir)
    existing_status = _load_run_status(run_dir)

    start_time = _utc_now()
    monotonic_start = time.monotonic()
    if dry_run:
        return _base_output_record(
            record,
            config_path=config_path,
            run_dir=run_dir,
            log_path=log_path,
            sequence=sequence,
            state="DRY_RUN",
            return_code=None,
            start_time=start_time,
            end_time=_utc_now(),
            elapsed_seconds=time.monotonic() - monotonic_start,
        )

    if resume and str((existing_status or {}).get("status", "")).strip().lower() == "success":
        return _base_output_record(
            record,
            config_path=config_path,
            run_dir=run_dir,
            log_path=log_path,
            sequence=sequence,
            state="SKIPPED",
            return_code=0,
            start_time=start_time,
            end_time=_utc_now(),
            elapsed_seconds=time.monotonic() - monotonic_start,
            failure_reason="already_successful",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CHEMLFLOW_CONFIG"] = str(config_path)
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"[local-doe] start={start_time}\n")
        log_fh.write(f"[local-doe] config={config_path}\n")
        log_fh.write(f"[local-doe] run_dir={run_dir or ''}\n")
        log_fh.flush()
        completed = subprocess.run(
            [python_bin, str(main_path)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        end_time = _utc_now()
        log_fh.write(f"\n[local-doe] end={end_time}\n")
        log_fh.write(f"[local-doe] return_code={completed.returncode}\n")

    state = "COMPLETED" if completed.returncode == 0 else "FAILED"
    failure_reason = None if completed.returncode == 0 else "nonzero_return_code"
    return _base_output_record(
        record,
        config_path=config_path,
        run_dir=run_dir,
        log_path=log_path,
        sequence=sequence,
        state=state,
        return_code=completed.returncode,
        start_time=start_time,
        end_time=end_time,
        elapsed_seconds=time.monotonic() - monotonic_start,
        failure_reason=failure_reason,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _duplicate_artifact_dirs(records: list[dict[str, Any]], doe_dir: Path) -> dict[str, dict[str, list[str]]]:
    seen: dict[tuple[str, str], list[str]] = {}
    for record in records:
        config_path = _resolve_config_path(str(record.get("config_path", "")), doe_dir=doe_dir)
        run_dir, base_dir = _artifact_dirs_for_config(config_path)
        case_id = str(record.get("case_id", "")).strip() or config_path.stem
        for kind, path in (("run_dir", run_dir), ("base_dir", base_dir)):
            if path is None:
                continue
            key = (kind, str(path.resolve() if path.exists() else path.absolute()))
            seen.setdefault(key, []).append(case_id)

    duplicates: dict[str, dict[str, list[str]]] = {"run_dir": {}, "base_dir": {}}
    for (kind, path), case_ids in seen.items():
        if len(case_ids) > 1:
            duplicates[kind][path] = case_ids
    return {kind: values for kind, values in duplicates.items() if values}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    doe_dir = Path(args.doe_dir)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else doe_dir / "manifest.jsonl"
    output_manifest = (
        Path(args.output_manifest)
        if str(args.output_manifest).strip()
        else doe_dir / "execution_manifest.jsonl"
    )
    logs_dir = Path(args.logs_dir) if str(args.logs_dir).strip() else doe_dir / "local_logs"
    main_path = Path(args.main)

    if not manifest_path.exists():
        print(f"[local-doe] missing manifest: {manifest_path}", file=sys.stderr)
        return 2
    if not main_path.exists():
        print(f"[local-doe] missing main.py: {main_path}", file=sys.stderr)
        return 2

    records = _load_valid_manifest_records(manifest_path)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]
    if not records:
        print("[local-doe] no valid execution-child records found.", file=sys.stderr)
        return 3

    logs_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, int(args.max_workers))
    if max_workers > 1 and args.stop_on_failure:
        print(
            "[local-doe] refusing --max-workers > 1 with --stop-on-failure because "
            "--stop-on-failure currently uses the serial execution path. Remove "
            "--stop-on-failure for parallel execution, or set --max-workers 1 for "
            "serial fail-fast debugging.",
            file=sys.stderr,
        )
        return 2
    if max_workers > 1 and not args.allow_shared_artifacts:
        duplicates = _duplicate_artifact_dirs(records, doe_dir=doe_dir)
        if duplicates:
            print(
                "[local-doe] refusing parallel execution because generated configs "
                "share artifact directories. Use --allow-shared-artifacts to override.",
                file=sys.stderr,
            )
            print(json.dumps(duplicates, indent=2, sort_keys=True), file=sys.stderr)
            return 2

    results: list[dict[str, Any]] = []
    failed = False

    if max_workers == 1 or args.stop_on_failure:
        for sequence, record in enumerate(records, start=1):
            result = _run_one(
                record,
                sequence=sequence,
                doe_dir=doe_dir,
                logs_dir=logs_dir,
                python_bin=str(args.python),
                main_path=main_path,
                resume=bool(args.resume),
                dry_run=bool(args.dry_run),
            )
            results.append(result)
            _write_jsonl(output_manifest, results)
            print(
                "[local-doe] {case} {state} rc={rc} log={log}".format(
                    case=result.get("case_id") or Path(str(result["config_path"])).stem,
                    state=result["state"],
                    rc=result["return_code"],
                    log=result["log_path"],
                )
            )
            if args.stop_on_failure and result["state"] == "FAILED":
                failed = True
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    record,
                    sequence=sequence,
                    doe_dir=doe_dir,
                    logs_dir=logs_dir,
                    python_bin=str(args.python),
                    main_path=main_path,
                    resume=bool(args.resume),
                    dry_run=bool(args.dry_run),
                ): sequence
                for sequence, record in enumerate(records, start=1)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                results.sort(key=lambda row: str(row.get("execution_id", "")))
                _write_jsonl(output_manifest, results)
                print(
                    "[local-doe] {case} {state} rc={rc} log={log}".format(
                        case=result.get("case_id") or Path(str(result["config_path"])).stem,
                        state=result["state"],
                        rc=result["return_code"],
                        log=result["log_path"],
                    )
                )

    results.sort(key=lambda row: str(row.get("execution_id", "")))
    _write_jsonl(output_manifest, results)

    counts: dict[str, int] = {}
    for row in results:
        counts[str(row.get("state", "UNKNOWN"))] = counts.get(str(row.get("state", "UNKNOWN")), 0) + 1
    print(f"[local-doe] manifest: {output_manifest}")
    print(f"[local-doe] state_counts: {json.dumps(counts, sort_keys=True)}")

    failed = failed or any(row.get("state") == "FAILED" for row in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
