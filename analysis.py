#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional for CSV-only mode
    yaml = None


CHILD_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


@dataclass(frozen=True)
class ChildJob:
    job_id: str
    case_name: str | None
    parent_case_id: str | None
    config_path: str | None
    profile: str | None
    model_type: str | None
    scaler: str | None
    split_mode: str | None
    split_strategy: str | None
    scientific_config_id: str | None
    execution_label: str | None
    state: str
    exit_code: str
    elapsed: str
    failure_reason: str | None


@dataclass(frozen=True)
class GeneralizationRecord:
    case_name: str
    model_type: str | None
    split_mode: str | None
    split_strategy: str | None
    feature_input: str | None
    metric_name: str
    train_value: float
    test_value: float
    val_value: float | None
    gap_train_minus_test: float
    gap_train_minus_test_std: float | None
    overfit_flag: bool
    underfit_flag: bool
    config_path: str | None
    run_dir: str | None
    scaler: str | None = None
    parent_case_id: str | None = None
    scientific_config_id: str | None = None
    execution_label: str | None = None
    state: str | None = None
    failure_reason: str | None = None
    slice_count: int = 1
    metric_slices_used: int = 1
    completed_slices: int = 1
    failed_slices: int = 0
    config_paths: tuple[str, ...] = ()
    run_dirs: tuple[str, ...] = ()


METRIC_FIELDS = [
    "r2",
    "mae",
    "rmse",
    "mse",
    "auc",
    "auprc",
    "accuracy",
    "f1",
    "train_r2",
    "test_r2",
    "val_r2",
    "train_mae",
    "test_mae",
    "val_mae",
    "train_rmse",
    "test_rmse",
    "val_rmse",
    "train_mse",
    "test_mse",
    "val_mse",
    "train_auc",
    "test_auc",
    "val_auc",
    "train_auprc",
    "test_auprc",
    "val_auprc",
    "train_accuracy",
    "test_accuracy",
    "val_accuracy",
    "train_f1",
    "test_f1",
    "val_f1",
]


RUN_METRIC_CSV_FIELDS = [
    "case_name",
    "group_index",
    "parent_case_id",
    "scientific_config_id",
    "job_id",
    "state",
    "failure_reason",
    "profile",
    "model_type",
    "scaler",
    "feature_input",
    "split_mode",
    "split_strategy",
    "execution_label",
    "slice_count",
    "completed_slices",
    "failed_slices",
    "elapsed",
    "config_path",
    "config_paths",
    "run_dir",
    "run_dirs",
    "metrics_path",
    "metrics_paths",
] + METRIC_FIELDS + [f"{field}_std" for field in METRIC_FIELDS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze CheMLFlow DOE orchestration runs from Slurm job logs + sacct and "
            "write summary reports (JSON/CSV)."
        )
    )
    parser.add_argument(
        "--orchestrator-job-id",
        required=True,
        help="Slurm orchestrator job ID (for example: 2307731).",
    )
    parser.add_argument(
        "--logs-dir",
        default="/mnt/home/f0113398/logs",
        help="Directory containing <orchestrator-log-prefix>-<jobid>.out/err logs.",
    )
    parser.add_argument(
        "--orchestrator-log-prefix",
        default="ysi_doe",
        help=(
            "Prefix used for orchestrator log filenames in --logs-dir "
            "(default: ysi_doe; example for PAH: pah_doe)."
        ),
    )
    parser.add_argument(
        "--doe-dir",
        default="/mnt/home/f0113398/ysi_doe",
        help="DOE output directory containing manifest.jsonl and generated case YAMLs.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory for reports. "
            "Default: <doe-dir>/analysis_<orchestrator-job-id>."
        ),
    )
    parser.add_argument(
        "--sacct-bin",
        default="sacct",
        help="Path/name of sacct executable.",
    )
    parser.add_argument(
        "--all-runs-csv",
        default="",
        help=(
            "Optional shortcut mode: path to an existing all_runs_metrics.csv. "
            "When set, analysis runs in CSV-only plotting mode."
        ),
    )
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=0.15,
        help="Threshold on train-test metric gap to flag overfitting (default: 0.15).",
    )
    parser.add_argument(
        "--underfit-threshold-r2",
        type=float,
        default=0.20,
        help="Regression underfit threshold on train r2 (default: 0.20).",
    )
    parser.add_argument(
        "--underfit-threshold-auc",
        type=float,
        default=0.65,
        help="Classification underfit threshold on train auc (default: 0.65).",
    )
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_child_job_ids(orchestrator_out: Path) -> list[str]:
    text = _read_text(orchestrator_out)
    ids = [m.group(1) for m in CHILD_ID_PATTERN.finditer(text)]
    # Keep original order, drop duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for job_id in ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        deduped.append(job_id)
    return deduped


def _load_valid_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for line in _read_text(manifest_path).splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if str(rec.get("status", "")).lower() != "valid":
            continue
        record_type = str(rec.get("record_type", "execution_child")).strip().lower()
        if record_type == "parent":
            continue
        cfg = rec.get("config_path")
        if cfg:
            valid.append(rec)
    return valid


def _run_sacct(sacct_bin: str, child_ids: list[str]) -> list[dict[str, str]]:
    if not child_ids:
        return []
    job_arg = ",".join(child_ids)
    cmd = [
        sacct_bin,
        "-n",
        "-P",
        "-j",
        job_arg,
        "--format",
        "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows: list[dict[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 6:
            continue
        rows.append(
            {
                "job_id_raw": parts[0],
                "job_id": parts[1],
                "job_name": parts[2],
                "state": parts[3],
                "exit_code": parts[4],
                "elapsed": parts[5],
            }
        )
    return rows


def _parse_case_name(case_config_path: str | None) -> str | None:
    if not case_config_path:
        return None
    return Path(case_config_path).stem


def _extract_case_fields(case_name: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    if not case_name:
        return None, None, None, None
    parts = case_name.split("__")
    if len(parts) < 5:
        return None, None, None, None
    return parts[1], parts[2], parts[3], parts[4]


def _derive_failure_reason(job_state: str, step_states: list[str]) -> str | None:
    state = (job_state or "").upper()
    step_upper = [s.upper() for s in step_states]
    if state == "COMPLETED":
        return None
    if "OUT_OF_MEMORY" in state or any("OUT_OF_MEMORY" in s for s in step_upper):
        return "out_of_memory"
    if "TIMEOUT" in state:
        return "timeout"
    if "CANCELLED" in state:
        return "cancelled"
    if "FAILED" in state:
        return "failed_other"
    if "PENDING" in state:
        return "pending"
    if "RUNNING" in state:
        return "running"
    return "other"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_case_index(case_name: str) -> int | None:
    m = re.match(r"^case_(\d+)", case_name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _load_all_runs_metrics_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _stable_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _get_dotted(container: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = container
    for part in [p for p in dotted.split(".") if p]:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _pop_dotted(container: dict[str, Any], dotted: str) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    parents: list[tuple[dict[str, Any], str]] = []
    current: Any = container
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        parents.append((current, part))
        current = current[part]
    if not isinstance(current, dict):
        return
    current.pop(parts[-1], None)
    while parents:
        parent, key = parents.pop()
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break


def _infer_feature_input(config: dict[str, Any]) -> str | None:
    nodes = ((config.get("pipeline") or {}).get("nodes") if isinstance(config.get("pipeline"), dict) else None) or []
    if "featurize.morgan" in nodes:
        return "featurize.morgan"
    if "featurize.rdkit_labeled" in nodes:
        return "featurize.rdkit_labeled"
    if "featurize.rdkit" in nodes:
        return "featurize.rdkit"
    if "featurize.none" in nodes or "use.curated_features" in nodes:
        return "featurize.none"
    pipeline_cfg = config.get("pipeline")
    if isinstance(pipeline_cfg, dict):
        configured = str(pipeline_cfg.get("feature_input", "")).strip()
        if configured:
            if configured == "use.curated_features":
                return "featurize.none"
            return configured
    model_type = str((((config.get("train") or {}).get("model") or {}).get("type", ""))).strip().lower()
    if _is_chemprop_like_model_type(model_type):
        return "smiles_native"
    return "none" if "train" in nodes else None


def _infer_scaler(config: dict[str, Any]) -> str | None:
    scaler = str(_get_dotted(config, "preprocess.scaler", "")).strip()
    return scaler or None


def _infer_scaler_from_config_path(config_path: str | None) -> str | None:
    if not config_path or yaml is None:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    return _infer_scaler(cfg)


def _is_chemprop_like_model_type(model_type: str) -> bool:
    return str(model_type).strip().lower() in {"chemprop", "chemeleon"}


def _scientific_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    global_cfg = payload.get("global")
    if isinstance(global_cfg, dict):
        global_cfg.pop("base_dir", None)
        global_cfg.pop("run_dir", None)
        global_cfg.pop("artifact_retention", None)
        runs_cfg = global_cfg.get("runs")
        if isinstance(runs_cfg, dict):
            runs_cfg.pop("id", None)
    split_mode = str(_get_dotted(payload, "split.mode", "")).strip().lower()
    if split_mode == "cv":
        _pop_dotted(payload, "split.cv.fold_index")
        _pop_dotted(payload, "split.cv.repeat_index")
    elif split_mode == "nested_holdout_cv":
        _pop_dotted(payload, "split.inner.fold_index")
        _pop_dotted(payload, "split.inner.repeat_index")
    return payload


def _scientific_config_id(config: dict[str, Any]) -> str:
    return _stable_hash(_scientific_config_payload(config))


def _execution_label(config: dict[str, Any]) -> str:
    split_mode = str(_get_dotted(config, "split.mode", "holdout")).strip().lower() or "holdout"
    if split_mode == "cv":
        fold_index = int(_get_dotted(config, "split.cv.fold_index", 0))
        repeat_index = int(_get_dotted(config, "split.cv.repeat_index", 0))
        return f"rep{repeat_index}_fold{fold_index}"
    if split_mode == "nested_holdout_cv":
        stage = str(_get_dotted(config, "split.stage", "inner")).strip().lower() or "inner"
        if stage == "inner":
            fold_index = int(_get_dotted(config, "split.inner.fold_index", 0))
            repeat_index = int(_get_dotted(config, "split.inner.repeat_index", 0))
            return f"{stage}_rep{repeat_index}_fold{fold_index}"
        return stage
    return split_mode


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _join_unique(values: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return ";".join(out)


def _failure_summary(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(r.get("failure_reason", "")).strip() for r in rows if str(r.get("failure_reason", "")).strip())
    if not counts:
        return ""
    return ",".join(f"{reason}={count}" for reason, count in sorted(counts.items()))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _resolve_metrics_path(run_dir: Path, model_type: str) -> Path:
    if _is_chemprop_like_model_type(model_type):
        return run_dir / "chemprop_metrics.json"
    return run_dir / f"{model_type}_metrics.json"


def _resolve_split_metrics(
    model_type: str,
    run_dir: Path,
    metrics_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metrics_payload:
        split_metrics_path = metrics_payload.get("split_metrics_path")
        if split_metrics_path:
            path = Path(str(split_metrics_path))
            payload = _load_json(path)
            if payload is not None:
                return payload
    if _is_chemprop_like_model_type(model_type):
        fallback = run_dir / "chemprop_split_metrics.json"
    else:
        fallback = run_dir / f"{model_type}_split_metrics.json"
    return _load_json(fallback)


def _extract_primary_metric(split_metrics: dict[str, Any]) -> str | None:
    train = split_metrics.get("train")
    test = split_metrics.get("test")
    if not isinstance(train, dict) or not isinstance(test, dict):
        return None
    # Prefer "higher-is-better" metrics for a clean generalization gap signal.
    for candidate in ("r2", "auc", "auprc", "accuracy", "f1"):
        if candidate in train and candidate in test:
            if _safe_float(train.get(candidate)) is not None and _safe_float(test.get(candidate)) is not None:
                return candidate
    return None


def _build_generalization_records(
    jobs: list[ChildJob],
    overfit_threshold: float,
    underfit_threshold_r2: float,
    underfit_threshold_auc: float,
) -> list[GeneralizationRecord]:
    records: list[GeneralizationRecord] = []
    for job in jobs:
        if job.state != "COMPLETED":
            continue
        if not job.config_path:
            continue
        cfg_path = Path(job.config_path)
        if not cfg_path.exists():
            continue
        if yaml is None:
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        run_dir_raw = ((cfg.get("global") or {}).get("run_dir") if isinstance(cfg.get("global"), dict) else None)
        model_cfg = (cfg.get("train") or {}).get("model") if isinstance(cfg.get("train"), dict) else {}
        model_type = ""
        if isinstance(model_cfg, dict):
            model_type = str(model_cfg.get("type", "")).strip()
        if not run_dir_raw or not model_type:
            continue
        run_dir = Path(str(run_dir_raw))
        metrics_path = _resolve_metrics_path(run_dir, model_type)
        metrics_payload = _load_json(metrics_path)
        split_metrics = _resolve_split_metrics(model_type=model_type, run_dir=run_dir, metrics_payload=metrics_payload)
        if not isinstance(split_metrics, dict):
            continue
        metric_name = _extract_primary_metric(split_metrics)
        if not metric_name:
            continue
        split_mode = str(_get_dotted(cfg, "split.mode", job.split_mode or "")).strip() or None
        split_strategy = str(_get_dotted(cfg, "split.strategy", job.split_strategy or "")).strip() or None
        feature_input = _infer_feature_input(cfg)
        scaler = _infer_scaler(cfg)
        scientific_config_id = job.scientific_config_id or _scientific_config_id(cfg)
        execution_label = job.execution_label or _execution_label(cfg)
        train = split_metrics.get("train") or {}
        test = split_metrics.get("test") or {}
        val = split_metrics.get("val") or {}
        train_v = _safe_float(train.get(metric_name))
        test_v = _safe_float(test.get(metric_name))
        val_v = _safe_float(val.get(metric_name)) if isinstance(val, dict) else None
        if train_v is None or test_v is None:
            continue
        gap = train_v - test_v
        overfit_flag = gap >= overfit_threshold
        underfit_flag = False
        if metric_name == "r2":
            underfit_flag = train_v < underfit_threshold_r2 and test_v < underfit_threshold_r2
        elif metric_name in {"auc", "auprc", "accuracy", "f1"}:
            underfit_flag = train_v < underfit_threshold_auc and test_v < underfit_threshold_auc
        records.append(
            GeneralizationRecord(
                case_name=job.case_name or cfg_path.stem,
                model_type=job.model_type or (model_type or None),
                scaler=job.scaler or scaler,
                split_mode=split_mode,
                split_strategy=split_strategy,
                feature_input=feature_input,
                metric_name=metric_name,
                train_value=train_v,
                test_value=test_v,
                val_value=val_v,
                gap_train_minus_test=gap,
                gap_train_minus_test_std=None,
                overfit_flag=overfit_flag,
                underfit_flag=underfit_flag,
                config_path=str(cfg_path),
                run_dir=str(run_dir),
                parent_case_id=job.parent_case_id,
                scientific_config_id=scientific_config_id,
                execution_label=execution_label,
                state="COMPLETED",
                failure_reason=None,
                slice_count=1,
                metric_slices_used=1,
                completed_slices=1,
                failed_slices=0,
                config_paths=(str(cfg_path),),
                run_dirs=(str(run_dir),),
            )
        )
    return records


def _aggregate_generalization_records(
    records: list[GeneralizationRecord],
    overfit_threshold: float,
    underfit_threshold_r2: float,
    underfit_threshold_auc: float,
    config_summary: dict[str, dict[str, Any]] | None = None,
) -> list[GeneralizationRecord]:
    grouped: dict[tuple[str, str], list[GeneralizationRecord]] = defaultdict(list)
    for record in records:
        group_key = (record.parent_case_id or record.scientific_config_id or record.case_name, record.metric_name)
        grouped[group_key].append(record)

    def _sort_key(item: tuple[tuple[str, str], list[GeneralizationRecord]]) -> tuple[int, str]:
        recs = item[1]
        indices = [_parse_case_index(r.case_name) for r in recs]
        numeric = [idx for idx in indices if idx is not None]
        return (min(numeric) if numeric else 10**9, item[0][0])

    aggregated: list[GeneralizationRecord] = []
    for (aggregate_key, metric_name), recs in sorted(grouped.items(), key=_sort_key):
        representative = sorted(recs, key=lambda r: (_parse_case_index(r.case_name) or 10**9, r.case_name))[0]
        train_values = [r.train_value for r in recs]
        test_values = [r.test_value for r in recs]
        val_values = [r.val_value for r in recs if r.val_value is not None]
        gaps = [r.gap_train_minus_test for r in recs]
        train_mean = _mean(train_values)
        test_mean = _mean(test_values)
        val_mean = _mean(val_values) if val_values else None
        gap_mean = _mean(gaps)
        gap_std = _std(gaps)
        if metric_name == "r2":
            underfit_flag = train_mean < underfit_threshold_r2 and test_mean < underfit_threshold_r2
        else:
            underfit_flag = train_mean < underfit_threshold_auc and test_mean < underfit_threshold_auc
        summary = (config_summary or {}).get(aggregate_key, {})
        slice_count = max(int(summary.get("slice_count", len(recs))), len(recs))
        completed_job_slices = max(int(summary.get("completed_slices", len(recs))), len(recs))
        metric_slices_used = len(recs)
        completed_slices = metric_slices_used
        failed_slices = max(slice_count - metric_slices_used, 0)
        state = str(summary.get("state", "COMPLETED" if completed_slices == slice_count else "PARTIAL"))
        failure_reason = str(summary.get("failure_reason", "")).strip() or None
        missing_metric_slices = max(completed_job_slices - metric_slices_used, 0)
        if missing_metric_slices > 0:
            state = "PARTIAL"
            missing_reason = f"missing_generalization_metrics={missing_metric_slices}"
            failure_reason = (
                f"{failure_reason}; {missing_reason}"
                if failure_reason and missing_reason not in failure_reason
                else failure_reason or missing_reason
            )
        elif completed_slices < slice_count and state == "COMPLETED":
            state = "PARTIAL"
        aggregated.append(
            GeneralizationRecord(
                case_name=representative.case_name,
                model_type=representative.model_type,
                scaler=representative.scaler,
                split_mode=representative.split_mode,
                split_strategy=representative.split_strategy,
                feature_input=representative.feature_input,
                metric_name=metric_name,
                train_value=train_mean,
                test_value=test_mean,
                val_value=val_mean,
                gap_train_minus_test=gap_mean,
                gap_train_minus_test_std=gap_std,
                overfit_flag=gap_mean >= overfit_threshold,
                underfit_flag=underfit_flag,
                config_path=representative.config_path,
                run_dir=representative.run_dir,
                parent_case_id=representative.parent_case_id,
                scientific_config_id=representative.scientific_config_id,
                execution_label="aggregated",
                state=state,
                failure_reason=failure_reason,
                slice_count=slice_count,
                metric_slices_used=metric_slices_used,
                completed_slices=completed_slices,
                failed_slices=failed_slices,
                config_paths=tuple(sorted({p for r in recs for p in r.config_paths if p})),
                run_dirs=tuple(sorted({p for r in recs for p in r.run_dirs if p})),
            )
        )
    return aggregated


def _write_generalization_csv(path: Path, records: list[GeneralizationRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_name",
                "parent_case_id",
                "scientific_config_id",
                "model_type",
                "scaler",
                "feature_input",
                "split_mode",
                "split_strategy",
                "execution_label",
                "state",
                "failure_reason",
                "slice_count",
                "metric_slices_used",
                "completed_slices",
                "failed_slices",
                "metric_name",
                "train_value",
                "test_value",
                "val_value",
                "gap_train_minus_test",
                "gap_train_minus_test_std",
                "overfit_flag",
                "underfit_flag",
                "config_path",
                "config_paths",
                "run_dir",
                "run_dirs",
            ],
        )
        writer.writeheader()
        for r in sorted(records, key=lambda x: x.gap_train_minus_test, reverse=True):
            writer.writerow(
                {
                    "case_name": r.case_name,
                    "parent_case_id": r.parent_case_id or "",
                    "scientific_config_id": r.scientific_config_id or "",
                    "model_type": r.model_type or "",
                    "scaler": r.scaler or "",
                    "feature_input": r.feature_input or "",
                    "split_mode": r.split_mode or "",
                    "split_strategy": r.split_strategy or "",
                    "execution_label": r.execution_label or "",
                    "state": r.state or "",
                    "failure_reason": r.failure_reason or "",
                    "slice_count": r.slice_count,
                    "metric_slices_used": r.metric_slices_used,
                    "completed_slices": r.completed_slices,
                    "failed_slices": r.failed_slices,
                    "metric_name": r.metric_name,
                    "train_value": r.train_value,
                    "test_value": r.test_value,
                    "val_value": "" if r.val_value is None else r.val_value,
                    "gap_train_minus_test": r.gap_train_minus_test,
                    "gap_train_minus_test_std": "" if r.gap_train_minus_test_std is None else r.gap_train_minus_test_std,
                    "overfit_flag": int(r.overfit_flag),
                    "underfit_flag": int(r.underfit_flag),
                    "config_path": r.config_path or "",
                    "config_paths": ";".join(r.config_paths),
                    "run_dir": r.run_dir or "",
                    "run_dirs": ";".join(r.run_dirs),
                }
            )


def _write_generalization_plot(path: Path, records: list[GeneralizationRecord], threshold: float) -> bool:
    if not records:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ranked = sorted(records, key=lambda x: x.gap_train_minus_test, reverse=True)
    x = list(range(1, len(ranked) + 1))
    y = [r.gap_train_minus_test for r in ranked]
    colors = ["#d62728" if r.overfit_flag else "#1f77b4" for r in ranked]
    aggregated = any(r.slice_count > 1 for r in ranked)
    plt.figure(figsize=(13, 5.5))
    plt.bar(x, y, color=colors, alpha=0.9)
    plt.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.4, label=f"Overfit threshold ({threshold:.2f})")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.xlabel("Scientific Config Rank" if aggregated else "Case Rank (sorted by train-test gap)")
    plt.ylabel("Generalization gap (train - test)")
    plt.title(
        "Per-configuration Generalization Gap (higher => more overfitting risk)"
        if aggregated
        else "Per-case Generalization Gap (higher => more overfitting risk)"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def _write_model_gap_summary(path: Path, records: list[GeneralizationRecord]) -> None:
    by_model: dict[tuple[str, str], list[GeneralizationRecord]] = defaultdict(list)
    for r in records:
        by_model[(r.model_type or "unknown", r.scaler or "")].append(r)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_type",
                "scaler",
                "n_cases",
                "mean_gap",
                "median_gap",
                "max_gap",
                "overfit_count",
                "underfit_count",
            ],
        )
        writer.writeheader()
        for model, scaler in sorted(by_model):
            records_for_group = by_model[(model, scaler)]
            vals = sorted(r.gap_train_minus_test for r in records_for_group)
            n = len(vals)
            mean_gap = sum(vals) / n if n else 0.0
            median_gap = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
            writer.writerow(
                {
                    "model_type": model,
                    "scaler": scaler,
                    "n_cases": n,
                    "mean_gap": mean_gap,
                    "median_gap": median_gap,
                    "max_gap": max(vals) if vals else 0.0,
                    "overfit_count": sum(1 for r in records_for_group if r.overfit_flag),
                    "underfit_count": sum(1 for r in records_for_group if r.underfit_flag),
                }
            )


def _build_all_runs_metric_rows(jobs: list[ChildJob]) -> list[dict[str, Any]]:
    def _pick_metric(
        metric: str,
        top_metrics: dict[str, Any],
        test_split: dict[str, Any],
        val_split: dict[str, Any],
        train_split: dict[str, Any],
    ) -> Any:
        # Prefer top-level metrics; otherwise fall back to test/val/train split values.
        for source in (top_metrics, test_split, val_split, train_split):
            if metric in source and source.get(metric, "") != "":
                return source.get(metric)
        return ""

    rows: list[dict[str, Any]] = []
    for job in jobs:
        cfg: dict[str, Any] = {}
        run_dir: str | None = None
        model_type = job.model_type or ""
        profile = job.profile or ""
        split_mode = job.split_mode or ""
        split_strategy = job.split_strategy or ""
        scaler = job.scaler or ""
        feature_input = ""
        parent_case_id = job.parent_case_id or ""
        scientific_config_id = job.scientific_config_id or ""
        execution_label = job.execution_label or ""
        metrics_payload: dict[str, Any] | None = None
        metrics_path: str | None = None
        split_metrics: dict[str, Any] | None = None

        if job.config_path:
            cfg_path = Path(job.config_path)
            if cfg_path.exists():
                if yaml is not None:
                    try:
                        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                        if isinstance(loaded, dict):
                            cfg = loaded
                    except Exception:
                        cfg = {}

        if cfg:
            split_mode = str(_get_dotted(cfg, "split.mode", split_mode)).strip() or split_mode
            split_strategy = str(_get_dotted(cfg, "split.strategy", split_strategy)).strip() or split_strategy
            scaler = _infer_scaler(cfg) or ""
            feature_input = _infer_feature_input(cfg) or ""
            if not scientific_config_id:
                scientific_config_id = _scientific_config_id(cfg)
            if not execution_label:
                execution_label = _execution_label(cfg)

        global_cfg = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
        train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
        train_model_cfg = train_cfg.get("model") if isinstance(train_cfg.get("model"), dict) else {}
        if not model_type:
            model_type = str(train_model_cfg.get("type", "")).strip()
        run_dir_raw = global_cfg.get("run_dir")
        if run_dir_raw:
            run_dir = str(run_dir_raw)

        if run_dir and model_type:
            run_dir_path = Path(run_dir)
            mp = _resolve_metrics_path(run_dir_path, model_type)
            metrics_path = str(mp)
            metrics_payload = _load_json(mp)
            split_metrics = _resolve_split_metrics(model_type=model_type, run_dir=run_dir_path, metrics_payload=metrics_payload)

        top_metrics = metrics_payload if isinstance(metrics_payload, dict) else {}
        train_split = split_metrics.get("train") if isinstance(split_metrics, dict) and isinstance(split_metrics.get("train"), dict) else {}
        test_split = split_metrics.get("test") if isinstance(split_metrics, dict) and isinstance(split_metrics.get("test"), dict) else {}
        val_split = split_metrics.get("val") if isinstance(split_metrics, dict) and isinstance(split_metrics.get("val"), dict) else {}

        rows.append(
            {
                "case_name": job.case_name or "",
                "group_index": "",
                "parent_case_id": parent_case_id,
                "scientific_config_id": scientific_config_id,
                "job_id": job.job_id,
                "state": job.state,
                "failure_reason": job.failure_reason or "",
                "profile": profile,
                "model_type": model_type,
                "scaler": scaler,
                "feature_input": feature_input,
                "split_mode": split_mode,
                "split_strategy": split_strategy,
                "execution_label": execution_label,
                "slice_count": 1,
                "completed_slices": 1 if str(job.state).upper() == "COMPLETED" else 0,
                "failed_slices": 0 if str(job.state).upper() == "COMPLETED" else 1,
                "elapsed": job.elapsed,
                "config_path": job.config_path or "",
                "config_paths": job.config_path or "",
                "run_dir": run_dir or "",
                "run_dirs": run_dir or "",
                "metrics_path": metrics_path or "",
                "metrics_paths": metrics_path or "",
                "r2": _pick_metric("r2", top_metrics, test_split, val_split, train_split),
                "mae": _pick_metric("mae", top_metrics, test_split, val_split, train_split),
                "rmse": _pick_metric("rmse", top_metrics, test_split, val_split, train_split),
                "mse": _pick_metric("mse", top_metrics, test_split, val_split, train_split),
                "auc": _pick_metric("auc", top_metrics, test_split, val_split, train_split),
                "auprc": _pick_metric("auprc", top_metrics, test_split, val_split, train_split),
                "accuracy": _pick_metric("accuracy", top_metrics, test_split, val_split, train_split),
                "f1": _pick_metric("f1", top_metrics, test_split, val_split, train_split),
                "train_r2": train_split.get("r2", ""),
                "test_r2": test_split.get("r2", ""),
                "val_r2": val_split.get("r2", ""),
                "train_mae": train_split.get("mae", ""),
                "test_mae": test_split.get("mae", ""),
                "val_mae": val_split.get("mae", ""),
                "train_rmse": train_split.get("rmse", ""),
                "test_rmse": test_split.get("rmse", ""),
                "val_rmse": val_split.get("rmse", ""),
                "train_mse": train_split.get("mse", ""),
                "test_mse": test_split.get("mse", ""),
                "val_mse": val_split.get("mse", ""),
                "train_auc": train_split.get("auc", ""),
                "test_auc": test_split.get("auc", ""),
                "val_auc": val_split.get("auc", ""),
                "train_auprc": train_split.get("auprc", ""),
                "test_auprc": test_split.get("auprc", ""),
                "val_auprc": val_split.get("auprc", ""),
                "train_accuracy": train_split.get("accuracy", ""),
                "test_accuracy": test_split.get("accuracy", ""),
                "val_accuracy": val_split.get("accuracy", ""),
                "train_f1": train_split.get("f1", ""),
                "test_f1": test_split.get("f1", ""),
                "val_f1": val_split.get("f1", ""),
            }
        )
        for metric_field in METRIC_FIELDS:
            rows[-1][f"{metric_field}_std"] = ""

    return rows


def _aggregate_all_runs_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent_case_id = str(row.get("parent_case_id", "")).strip()
        scientific_id = str(row.get("scientific_config_id", "")).strip()
        key = (
            parent_case_id
            or scientific_id
            or str(row.get("config_path", "")).strip()
            or str(row.get("case_name", "")).strip()
        )
        grouped[key].append(row)

    def _sort_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, str]:
        recs = item[1]
        indices = [_parse_case_index(str(r.get("case_name", ""))) for r in recs]
        numeric = [idx for idx in indices if idx is not None]
        return (min(numeric) if numeric else 10**9, item[0])

    aggregated_rows: list[dict[str, Any]] = []
    for group_index, (_, recs) in enumerate(sorted(grouped.items(), key=_sort_key), start=1):
        representative = sorted(
            recs,
            key=lambda r: (_parse_case_index(str(r.get("case_name", ""))) or 10**9, str(r.get("case_name", ""))),
        )[0]
        completed = [r for r in recs if str(r.get("state", "")).upper() == "COMPLETED"]
        completed_count = len(completed)
        slice_count = len(recs)
        if completed_count == slice_count and slice_count > 0:
            state = "COMPLETED"
            failure_reason = ""
        elif completed_count > 0:
            state = "PARTIAL"
            failure_reason = _failure_summary([r for r in recs if str(r.get("state", "")).upper() != "COMPLETED"])
        else:
            state_counts = Counter(str(r.get("state", "")).strip() for r in recs if str(r.get("state", "")).strip())
            state = state_counts.most_common(1)[0][0] if state_counts else "UNKNOWN"
            failure_reason = _failure_summary(recs)

        aggregate: dict[str, Any] = {
            "case_name": representative.get("case_name", ""),
            "group_index": group_index,
            "parent_case_id": representative.get("parent_case_id", ""),
            "scientific_config_id": representative.get("scientific_config_id", ""),
            "job_id": _join_unique([str(r.get("job_id", "")) for r in recs]),
            "state": state,
            "failure_reason": failure_reason,
            "profile": representative.get("profile", ""),
            "model_type": representative.get("model_type", ""),
            "scaler": representative.get("scaler", ""),
            "feature_input": representative.get("feature_input", ""),
            "split_mode": representative.get("split_mode", ""),
            "split_strategy": representative.get("split_strategy", ""),
            "execution_label": "aggregated",
            "slice_count": slice_count,
            "completed_slices": completed_count,
            "failed_slices": slice_count - completed_count,
            "elapsed": "",
            "config_path": representative.get("config_path", ""),
            "config_paths": _join_unique([str(r.get("config_path", "")) for r in recs]),
            "run_dir": representative.get("run_dir", ""),
            "run_dirs": _join_unique([str(r.get("run_dir", "")) for r in recs]),
            "metrics_path": representative.get("metrics_path", ""),
            "metrics_paths": _join_unique([str(r.get("metrics_path", "")) for r in recs]),
        }
        for metric_field in METRIC_FIELDS:
            values = [
                v
                for v in (_safe_float(r.get(metric_field)) for r in completed)
                if v is not None
            ]
            aggregate[metric_field] = _mean(values) if values else ""
            aggregate[f"{metric_field}_std"] = _std(values) if values else ""
        aggregated_rows.append(aggregate)
    return aggregated_rows


def _summarize_scientific_configs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        parent_case_id = str(row.get("parent_case_id", "")).strip()
        scientific_id = str(row.get("scientific_config_id", "")).strip()
        key = parent_case_id or scientific_id
        if not key:
            continue
        summary[key] = {
            "state": str(row.get("state", "")).strip() or "UNKNOWN",
            "failure_reason": str(row.get("failure_reason", "")).strip(),
            "slice_count": int(_safe_float(row.get("slice_count")) or 0),
            "completed_slices": int(_safe_float(row.get("completed_slices")) or 0),
            "failed_slices": int(_safe_float(row.get("failed_slices")) or 0),
        }
    return summary


def _write_all_runs_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_METRIC_CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r.get("group_index", "")), str(r.get("case_name", "")), str(r.get("job_id", "")))):
            writer.writerow(row)


def _write_all_runs_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    completed = [r for r in rows if str(r.get("state", "")).upper() == "COMPLETED"]
    if not completed:
        return []

    has_group_index = any(_safe_float(r.get("group_index")) is not None for r in completed)
    for r in completed:
        if has_group_index:
            r["_plot_index"] = int(_safe_float(r.get("group_index")) or 0)
        else:
            case_name = str(r.get("case_name", ""))
            r["_plot_index"] = _parse_case_index(case_name)

    plotted_paths: list[str] = []

    # Plot 1: test metrics by case index.
    metric_specs = [
        ("test_r2", "Test R2"),
        ("test_mae", "Test MAE"),
        ("test_rmse", "Test RMSE"),
        ("test_mse", "Test MSE"),
    ]
    available = []
    for key, label in metric_specs:
        vals = [_safe_float(r.get(key)) for r in completed]
        if any(v is not None for v in vals):
            available.append((key, label))
    if available:
        n = len(available)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        split_colors = {"random": "#1f77b4", "scaffold": "#ff7f0e"}
        for ax, (key, label) in zip(axes, available):
            points = []
            for r in completed:
                x = r.get("_plot_index")
                y = _safe_float(r.get(key))
                if x is None or y is None:
                    continue
                points.append((int(x), float(y), str(r.get("split_strategy", "")).lower()))
            points.sort(key=lambda t: t[0])
            for x, y, split in points:
                ax.scatter(x, y, s=28, color=split_colors.get(split, "#7f7f7f"), alpha=0.85)
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color="#b0b0b0", linewidth=0.8, alpha=0.5)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Scientific Config Index" if has_group_index else "Case Number")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=7, label="random"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=7, label="scaffold"),
        ]
        axes[0].legend(handles=handles, title="split_strategy", loc="best")
        plt.suptitle("Test Metrics by Scientific Config" if has_group_index else "Test Metrics by Case Number", y=0.995)
        plt.tight_layout()
        out = output_dir / "test_metrics_by_case.png"
        plt.savefig(out, dpi=160)
        plt.close(fig)
        plotted_paths.append(str(out))

    # Plot 2/3: per-model split-strategy comparison for selected metrics.
    model_metrics = [
        ("test_r2", "Test R2"),
        ("test_mae", "Test MAE"),
    ]
    for metric_key, metric_label in model_metrics:
        recs: list[tuple[str, str, float]] = []
        for r in completed:
            model = str(r.get("model_type", "")).strip()
            split = str(r.get("split_strategy", "")).strip()
            val = _safe_float(r.get(metric_key))
            if not model or not split or val is None:
                continue
            recs.append((model, split, float(val)))
        if not recs:
            continue
        models = sorted({m for m, _, _ in recs})
        n_models = len(models)
        ncols = 4
        nrows = math.ceil(n_models / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
        split_order = ["random", "scaffold"]
        split_colors = {"random": "#1f77b4", "scaffold": "#ff7f0e"}
        for i, model in enumerate(models):
            ax = axes[i // ncols][i % ncols]
            model_points = [(s.lower(), v) for m, s, v in recs if m == model]
            by_split: dict[str, list[float]] = {"random": [], "scaffold": []}
            for split, val in model_points:
                by_split.setdefault(split, [])
                by_split[split].append(val)
            for x_pos, split in enumerate(split_order):
                vals = by_split.get(split, [])
                if not vals:
                    continue
                jitter = [x_pos + (j - (len(vals) - 1) / 2.0) * 0.06 for j in range(len(vals))]
                ax.scatter(jitter, vals, s=28, alpha=0.85, color=split_colors.get(split, "#7f7f7f"))
                mean_v = sum(vals) / len(vals)
                ax.hlines(mean_v, x_pos - 0.22, x_pos + 0.22, colors="black", linewidth=1.3)
            ax.set_xticks([0, 1], ["random", "scaffold"])
            ax.set_title(model, fontsize=9)
            ax.grid(alpha=0.25)
        for j in range(n_models, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])
        fig.suptitle(f"{metric_label} by Split Strategy per Model", y=0.995)
        fig.tight_layout()
        out = output_dir / f"{metric_key}_by_model_split.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        plotted_paths.append(str(out))

    return plotted_paths


def _write_csv(path: Path, jobs: list[ChildJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "job_id",
                "case_name",
                "parent_case_id",
                "profile",
                "model_type",
                "scaler",
                "split_mode",
                "split_strategy",
                "scientific_config_id",
                "execution_label",
                "state",
                "exit_code",
                "elapsed",
                "failure_reason",
                "config_path",
            ],
        )
        writer.writeheader()
        for j in jobs:
            writer.writerow(
                {
                    "job_id": j.job_id,
                    "case_name": j.case_name or "",
                    "parent_case_id": j.parent_case_id or "",
                    "profile": j.profile or "",
                    "model_type": j.model_type or "",
                    "scaler": j.scaler or "",
                    "split_mode": j.split_mode or "",
                    "split_strategy": j.split_strategy or "",
                    "scientific_config_id": j.scientific_config_id or "",
                    "execution_label": j.execution_label or "",
                    "state": j.state,
                    "exit_code": j.exit_code,
                    "elapsed": j.elapsed,
                    "failure_reason": j.failure_reason or "",
                    "config_path": j.config_path or "",
                }
            )


def _print_summary(
    orchestrator_job_id: str,
    jobs: list[ChildJob],
    mapping_mismatch: bool,
    output_dir: Path,
    overfit_records: list[GeneralizationRecord],
) -> None:
    state_counts = Counter(j.state for j in jobs)
    fail_reason_counts = Counter(j.failure_reason for j in jobs if j.failure_reason)
    print(f"Orchestrator job: {orchestrator_job_id}")
    print(f"Child jobs analyzed: {len(jobs)}")
    print("State counts:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count}")
    if fail_reason_counts:
        print("Failure reasons:")
        for reason, count in sorted(fail_reason_counts.items()):
            print(f"  {reason}: {count}")
    failed = [j for j in jobs if j.state not in {"COMPLETED"}]
    print(f"Failed/unfinished cases: {len(failed)}")
    if mapping_mismatch:
        print("WARNING: child job count and valid config count differ; case mapping may be partial.")
    if overfit_records:
        overfit_count = sum(1 for r in overfit_records if r.overfit_flag)
        underfit_count = sum(1 for r in overfit_records if r.underfit_flag)
        print(f"Generalization records: {len(overfit_records)}")
        print(f"Overfit-flagged cases: {overfit_count}")
        print(f"Underfit-flagged cases: {underfit_count}")
    print(f"Report directory: {output_dir}")


def main() -> int:
    args = _parse_args()
    if str(args.all_runs_csv).strip():
        csv_path = Path(str(args.all_runs_csv).strip())
        if not csv_path.exists():
            print(f"Missing all-runs CSV: {csv_path}", file=sys.stderr)
            return 2
        output_dir = Path(args.output_dir) if str(args.output_dir).strip() else csv_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = _load_all_runs_metrics_csv(csv_path)
        if not rows:
            print(f"No rows found in CSV: {csv_path}", file=sys.stderr)
            return 3
        plot_paths = _write_all_runs_plots(output_dir=output_dir, rows=rows)
        if not plot_paths:
            print("No plots generated (no completed rows or matplotlib unavailable).", file=sys.stderr)
            return 4
        print(f"Rows loaded: {len(rows)}")
        print(f"Plots generated: {len(plot_paths)}")
        for p in plot_paths:
            print(p)
        return 0

    orch_id = str(args.orchestrator_job_id).strip()
    logs_dir = Path(args.logs_dir)
    doe_dir = Path(args.doe_dir)
    output_dir = (
        Path(args.output_dir)
        if str(args.output_dir).strip()
        else doe_dir / f"analysis_{orch_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator_prefix = str(args.orchestrator_log_prefix).strip() or "ysi_doe"
    orchestrator_out = logs_dir / f"{orchestrator_prefix}-{orch_id}.out"
    if not orchestrator_out.exists():
        print(f"Missing orchestrator out log: {orchestrator_out}", file=sys.stderr)
        return 2

    manifest_path = doe_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 2

    child_ids = _parse_child_job_ids(orchestrator_out)
    if not child_ids:
        print("No child job IDs found in orchestrator out log.", file=sys.stderr)
        return 3

    valid_manifest_records = _load_valid_manifest_records(manifest_path)
    mapping_mismatch = len(valid_manifest_records) != len(child_ids)

    manifest_by_job_id: dict[str, dict[str, Any]] = {}
    for idx, job_id in enumerate(child_ids):
        if idx < len(valid_manifest_records):
            manifest_by_job_id[job_id] = valid_manifest_records[idx]

    sacct_rows = _run_sacct(args.sacct_bin, child_ids)
    # Capture step states so we can detect OOM even when the parent state is only FAILED.
    step_states_by_root: dict[str, list[str]] = defaultdict(list)
    root_rows: dict[str, dict[str, str]] = {}
    for row in sacct_rows:
        raw = row["job_id_raw"]
        if "." in raw:
            root_id = raw.split(".", 1)[0]
            step_states_by_root[root_id].append(row["state"])
            continue
        if raw.endswith(".batch") or raw.endswith(".extern"):
            continue
        if raw.isdigit():
            root_rows[raw] = row

    jobs: list[ChildJob] = []
    for job_id in child_ids:
        row = root_rows.get(job_id)
        if row is None:
            jobs.append(
                ChildJob(
                    job_id=job_id,
                    case_name=_parse_case_name((manifest_by_job_id.get(job_id) or {}).get("config_path")),
                    parent_case_id=str((manifest_by_job_id.get(job_id) or {}).get("parent_case_id", "")).strip() or None,
                    config_path=(manifest_by_job_id.get(job_id) or {}).get("config_path"),
                    profile=None,
                    model_type=None,
                    scaler=_infer_scaler_from_config_path((manifest_by_job_id.get(job_id) or {}).get("config_path")),
                    split_mode=None,
                    split_strategy=None,
                    scientific_config_id=str((manifest_by_job_id.get(job_id) or {}).get("scientific_config_id", "")).strip() or None,
                    execution_label=str((manifest_by_job_id.get(job_id) or {}).get("execution_label", "")).strip() or None,
                    state="UNKNOWN",
                    exit_code="",
                    elapsed="",
                    failure_reason="missing_sacct_row",
                )
            )
            continue
        manifest_record = manifest_by_job_id.get(job_id) or {}
        cfg = manifest_record.get("config_path")
        case_name = _parse_case_name(cfg)
        profile, model_type, split_mode, split_strategy = _extract_case_fields(case_name)
        scaler = _infer_scaler_from_config_path(cfg)
        state = row["state"]
        failure_reason = _derive_failure_reason(state, step_states_by_root.get(job_id, []))
        jobs.append(
            ChildJob(
                job_id=job_id,
                case_name=case_name,
                parent_case_id=str(manifest_record.get("parent_case_id", "")).strip() or None,
                config_path=cfg,
                profile=profile,
                model_type=model_type,
                scaler=scaler,
                split_mode=split_mode,
                split_strategy=split_strategy,
                scientific_config_id=str(manifest_record.get("scientific_config_id", "")).strip() or None,
                execution_label=str(manifest_record.get("execution_label", "")).strip() or None,
                state=state,
                exit_code=row["exit_code"],
                elapsed=row["elapsed"],
                failure_reason=failure_reason,
            )
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "orchestrator_job_id": orch_id,
        "logs_dir": str(logs_dir),
        "doe_dir": str(doe_dir),
        "manifest_path": str(manifest_path),
        "parent_manifest_path": str(doe_dir / "parent_manifest.jsonl"),
        "orchestrator_out_path": str(orchestrator_out),
        "child_job_count_from_log": len(child_ids),
        "valid_config_count_from_manifest": len(valid_manifest_records),
        "mapping_mismatch": mapping_mismatch,
        "state_counts": dict(Counter(j.state for j in jobs)),
        "failure_reason_counts": dict(Counter(j.failure_reason for j in jobs if j.failure_reason)),
        "jobs": [j.__dict__ for j in jobs],
    }

    json_path = output_dir / "report.json"
    csv_path = output_dir / "jobs.csv"
    failed_cfg_path = output_dir / "failed_case_configs.txt"
    failed_job_path = output_dir / "failed_job_ids.txt"
    raw_gap_csv_path = output_dir / "generalization_gaps_by_execution.csv"
    gap_csv_path = output_dir / "generalization_gaps.csv"
    model_gap_csv_path = output_dir / "generalization_gap_by_model.csv"
    overfit_plot_path = output_dir / "generalization_gap_plot.png"
    overfit_cfg_path = output_dir / "overfit_case_configs.txt"
    underfit_cfg_path = output_dir / "underfit_case_configs.txt"
    raw_all_runs_metrics_path = output_dir / "all_runs_metrics_by_execution.csv"
    all_runs_metrics_path = output_dir / "all_runs_metrics.csv"

    raw_all_runs_metric_rows = _build_all_runs_metric_rows(jobs)
    all_runs_metric_rows = _aggregate_all_runs_metric_rows(raw_all_runs_metric_rows)
    config_summary = _summarize_scientific_configs(all_runs_metric_rows)
    raw_overfit_records = _build_generalization_records(
        jobs=jobs,
        overfit_threshold=float(args.overfit_threshold),
        underfit_threshold_r2=float(args.underfit_threshold_r2),
        underfit_threshold_auc=float(args.underfit_threshold_auc),
    )
    overfit_records = _aggregate_generalization_records(
        records=raw_overfit_records,
        overfit_threshold=float(args.overfit_threshold),
        underfit_threshold_r2=float(args.underfit_threshold_r2),
        underfit_threshold_auc=float(args.underfit_threshold_auc),
        config_summary=config_summary,
    )
    complete_overfit_records = [r for r in overfit_records if r.slice_count > 0 and r.completed_slices == r.slice_count]
    _write_generalization_csv(raw_gap_csv_path, raw_overfit_records)
    _write_generalization_csv(gap_csv_path, overfit_records)
    _write_model_gap_summary(model_gap_csv_path, complete_overfit_records)
    _write_generalization_plot(overfit_plot_path, complete_overfit_records, float(args.overfit_threshold))
    _write_all_runs_metrics_csv(raw_all_runs_metrics_path, raw_all_runs_metric_rows)
    _write_all_runs_metrics_csv(all_runs_metrics_path, all_runs_metric_rows)
    all_runs_plot_paths = _write_all_runs_plots(output_dir=output_dir, rows=all_runs_metric_rows)

    report["generalization"] = {
        "overfit_threshold": float(args.overfit_threshold),
        "underfit_threshold_r2": float(args.underfit_threshold_r2),
        "underfit_threshold_auc": float(args.underfit_threshold_auc),
        "records_by_execution": len(raw_overfit_records),
        "records_analyzed": len(overfit_records),
        "records_complete": len(complete_overfit_records),
        "records_partial_or_incomplete": len(overfit_records) - len(complete_overfit_records),
        "overfit_count": sum(1 for r in complete_overfit_records if r.overfit_flag),
        "underfit_count": sum(1 for r in complete_overfit_records if r.underfit_flag),
        "generalization_gaps_by_execution_csv": str(raw_gap_csv_path),
        "generalization_gaps_csv": str(gap_csv_path),
        "generalization_gap_by_model_csv": str(model_gap_csv_path),
        "generalization_gap_plot": str(overfit_plot_path),
    }
    report["all_runs_metrics_by_execution_csv"] = str(raw_all_runs_metrics_path)
    report["all_runs_metrics_csv"] = str(all_runs_metrics_path)
    report["all_runs_plot_paths"] = all_runs_plot_paths

    overfit_cfg_lines = sorted({path for r in complete_overfit_records if r.overfit_flag for path in (r.config_paths or ((r.config_path,) if r.config_path else ())) if path})
    underfit_cfg_lines = sorted({path for r in complete_overfit_records if r.underfit_flag for path in (r.config_paths or ((r.config_path,) if r.config_path else ())) if path})
    overfit_cfg_path.write_text("\n".join(overfit_cfg_lines) + ("\n" if overfit_cfg_lines else ""), encoding="utf-8")
    underfit_cfg_path.write_text("\n".join(underfit_cfg_lines) + ("\n" if underfit_cfg_lines else ""), encoding="utf-8")

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(csv_path, jobs)

    failed = [j for j in jobs if j.state != "COMPLETED"]
    failed_cfg_lines = [j.config_path for j in failed if j.config_path]
    failed_job_lines = [j.job_id for j in failed]
    failed_cfg_path.write_text("\n".join(failed_cfg_lines) + ("\n" if failed_cfg_lines else ""), encoding="utf-8")
    failed_job_path.write_text("\n".join(failed_job_lines) + ("\n" if failed_job_lines else ""), encoding="utf-8")

    _print_summary(orch_id, jobs, mapping_mismatch, output_dir, complete_overfit_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
