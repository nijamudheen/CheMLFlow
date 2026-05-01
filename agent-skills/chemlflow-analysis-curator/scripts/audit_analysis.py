#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


def _counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") for row in rows))


def _tuple_counts(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[str, int]:
    counts = Counter("|".join(row.get(key, "") for key in keys) for row in rows)
    return dict(counts)


def audit(analysis_dir: Path) -> dict[str, Any]:
    report_path = analysis_dir / "report.json"
    agg_path = analysis_dir / "all_runs_metrics.csv"
    raw_path = analysis_dir / "all_runs_metrics_by_execution.csv"
    failed_configs = analysis_dir / "failed_case_configs.txt"
    failed_jobs = analysis_dir / "failed_job_ids.txt"

    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    agg_fields, agg_rows = _read_csv(agg_path)
    raw_fields, raw_rows = _read_csv(raw_path)
    issues: list[str] = []

    if report.get("mapping_mismatch"):
        issues.append("report.mapping_mismatch is true")

    child_count = report.get("child_job_count_from_log")
    valid_count = report.get("valid_config_count_from_manifest")
    if child_count is not None and valid_count is not None and child_count != valid_count:
        issues.append("child_job_count_from_log differs from valid_config_count_from_manifest")

    state_counts = report.get("state_counts")
    if isinstance(state_counts, dict):
        bad_states = {k: v for k, v in state_counts.items() if k != "COMPLETED" and v}
        if bad_states:
            issues.append(f"non-completed states present: {bad_states}")

    if "scaler" not in agg_fields:
        issues.append("all_runs_metrics.csv has no scaler column")
    if "scaler" not in raw_fields:
        issues.append("all_runs_metrics_by_execution.csv has no scaler column")

    bad_agg_slices = [
        row
        for row in agg_rows
        if row.get("slice_count") not in {"", row.get("completed_slices", "")}
        or row.get("failed_slices", "0") not in {"", "0"}
    ]
    if bad_agg_slices:
        issues.append(f"{len(bad_agg_slices)} aggregated rows have incomplete or failed slices")

    failed_config_lines = []
    if failed_configs.exists():
        failed_config_lines = [line for line in failed_configs.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed_job_lines = []
    if failed_jobs.exists():
        failed_job_lines = [line for line in failed_jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if failed_config_lines:
        issues.append(f"failed_case_configs.txt has {len(failed_config_lines)} entries")
    if failed_job_lines:
        issues.append(f"failed_job_ids.txt has {len(failed_job_lines)} entries")

    return {
        "analysis_dir": str(analysis_dir),
        "report_exists": report_path.exists(),
        "raw_rows": len(raw_rows),
        "aggregated_rows": len(agg_rows),
        "child_job_count_from_log": child_count,
        "valid_config_count_from_manifest": valid_count,
        "mapping_mismatch": report.get("mapping_mismatch"),
        "state_counts": state_counts,
        "failure_reason_counts": report.get("failure_reason_counts"),
        "raw_states": _counts(raw_rows, "state"),
        "aggregated_states": _counts(agg_rows, "state"),
        "raw_features": _counts(raw_rows, "feature_input"),
        "aggregated_features": _counts(agg_rows, "feature_input"),
        "raw_scalers": _counts(raw_rows, "scaler"),
        "aggregated_scalers": _counts(agg_rows, "scaler"),
        "raw_models": _counts(raw_rows, "model_type"),
        "aggregated_models": _counts(agg_rows, "model_type"),
        "aggregated_slice_counts": _tuple_counts(agg_rows, ("slice_count", "completed_slices", "failed_slices")),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CheMLFlow analysis output integrity.")
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.analysis_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
