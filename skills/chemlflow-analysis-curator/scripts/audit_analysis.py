#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_finite_number(value: str | None) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_primary_metric(fields: list[str], rows: list[dict[str, str]]) -> str | None:
    for metric in ("auc", "r2", "mae", "rmse", "accuracy", "f1", "auprc"):
        if metric in fields and rows and any(str(row.get(metric, "")).strip() for row in rows):
            return metric
    return None


def audit(
    analysis_dir: Path,
    *,
    expected_valid_children: int | None = None,
    expected_valid_parents: int | None = None,
    expected_folds: int | None = None,
    primary_metric: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    report_path = analysis_dir / "report.json"
    agg_path = analysis_dir / "all_runs_metrics.csv"
    raw_path = analysis_dir / "all_runs_metrics_by_execution.csv"
    failed_configs = analysis_dir / "failed_case_configs.txt"
    failed_jobs = analysis_dir / "failed_job_ids.txt"

    issues: list[str] = []
    observations: list[str] = []
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        issues.append("report.json is missing")

    agg_fields, agg_rows = _read_csv(agg_path)
    raw_fields, raw_rows = _read_csv(raw_path)

    if not agg_path.exists():
        issues.append("all_runs_metrics.csv is missing")
    if not raw_path.exists():
        issues.append("all_runs_metrics_by_execution.csv is missing")

    mapping_mismatch = report.get("mapping_mismatch")
    if mapping_mismatch is None:
        issues.append("report.mapping_mismatch is missing")
    elif mapping_mismatch:
        issues.append("report.mapping_mismatch is true")

    child_count = report.get("child_job_count_from_log")
    valid_count = report.get("valid_config_count_from_manifest")
    child_count_int = _as_int(child_count)
    valid_count_int = _as_int(valid_count)
    if child_count is None:
        issues.append("report.child_job_count_from_log is missing")
    if valid_count is None:
        issues.append("report.valid_config_count_from_manifest is missing")
    if child_count_int is not None and valid_count_int is not None and child_count_int != valid_count_int:
        issues.append("child_job_count_from_log differs from valid_config_count_from_manifest")

    if expected_valid_children is None:
        expected_valid_children = valid_count_int
    if expected_valid_parents is None and expected_valid_children is not None and expected_folds:
        if expected_valid_children % expected_folds == 0:
            expected_valid_parents = expected_valid_children // expected_folds
        else:
            issues.append("expected_valid_children is not divisible by expected_folds")

    if expected_valid_children is not None:
        if valid_count_int is not None and valid_count_int != expected_valid_children:
            issues.append(
                "valid_config_count_from_manifest="
                f"{valid_count_int} differs from expected_valid_children={expected_valid_children}"
            )
        if child_count_int is not None and child_count_int != expected_valid_children:
            issues.append(
                "child_job_count_from_log="
                f"{child_count_int} differs from expected_valid_children={expected_valid_children}"
            )
        if len(raw_rows) != expected_valid_children:
            issues.append(
                f"raw execution rows={len(raw_rows)} differs from expected_valid_children={expected_valid_children}"
            )

    if expected_valid_parents is not None and len(agg_rows) != expected_valid_parents:
        issues.append(
            f"aggregated rows={len(agg_rows)} differs from expected_valid_parents={expected_valid_parents}"
        )

    state_counts = report.get("state_counts")
    if state_counts is None:
        issues.append("report.state_counts is missing")
    elif isinstance(state_counts, dict):
        bad_states = {k: v for k, v in state_counts.items() if k != "COMPLETED" and v}
        if bad_states:
            issues.append(f"non-completed states present: {bad_states}")
    else:
        issues.append("report.state_counts is not an object")

    failure_reason_counts = report.get("failure_reason_counts")
    if failure_reason_counts is None:
        issues.append("report.failure_reason_counts is missing")
    elif isinstance(failure_reason_counts, dict):
        failures = {k: v for k, v in failure_reason_counts.items() if v}
        if failures:
            issues.append(f"failure_reason_counts is not empty: {failures}")
    else:
        issues.append("report.failure_reason_counts is not an object")

    raw_bad_states = {
        k: v for k, v in _counts(raw_rows, "state").items() if k != "COMPLETED" and v
    }
    if raw_bad_states:
        issues.append(f"raw execution rows contain non-completed states: {raw_bad_states}")

    agg_bad_states = {
        k: v for k, v in _counts(agg_rows, "state").items() if k != "COMPLETED" and v
    }
    if agg_bad_states:
        issues.append(f"aggregated rows contain non-completed states: {agg_bad_states}")

    metric_artifacts = report.get("metric_artifacts")
    primary_metric_complete: bool | None = None
    split_diagnostics_complete: bool | None = None
    if isinstance(metric_artifacts, dict):
        primary_metric_complete = metric_artifacts.get("primary_metric_complete")
        split_diagnostics_complete = metric_artifacts.get("split_diagnostics_complete")
        if primary_metric_complete is False:
            issues.append("report.metric_artifacts.primary_metric_complete is false")
        if split_diagnostics_complete is False:
            missing_split = metric_artifacts.get("missing_split_metrics", "")
            observations.append(
                "split diagnostics are incomplete"
                + (f" (missing_split_metrics={missing_split})" if missing_split != "" else "")
            )

    if "scaler" not in agg_fields:
        issues.append("all_runs_metrics.csv has no scaler column")
    if "scaler" not in raw_fields:
        issues.append("all_runs_metrics_by_execution.csv has no scaler column")

    if expected_folds is None:
        fold_values = [_as_int(row.get("slice_count")) for row in agg_rows]
        nonempty_fold_values = [value for value in fold_values if value is not None]
        if nonempty_fold_values:
            expected_folds = Counter(nonempty_fold_values).most_common(1)[0][0]

    bad_agg_slices = [
        row
        for row in agg_rows
        if row.get("slice_count") not in {"", row.get("completed_slices", "")}
        or row.get("failed_slices", "0") not in {"", "0"}
        or (expected_folds is not None and _as_int(row.get("slice_count")) != expected_folds)
        or (expected_folds is not None and _as_int(row.get("completed_slices")) != expected_folds)
    ]
    if bad_agg_slices:
        issues.append(f"{len(bad_agg_slices)} aggregated rows have incomplete or failed slices")

    generalization = report.get("generalization")
    if isinstance(generalization, dict):
        partial = _as_int(generalization.get("records_partial_or_incomplete"))
        if partial:
            issues.append(f"generalization has {partial} partial or incomplete records")

    primary_metric = primary_metric or _infer_primary_metric(agg_fields, agg_rows)
    nonfinite_metric_rows: list[str] = []
    if primary_metric:
        if primary_metric not in agg_fields:
            issues.append(f"primary metric {primary_metric!r} is not present in all_runs_metrics.csv")
        else:
            for row in agg_rows:
                if not _is_finite_number(row.get(primary_metric)):
                    nonfinite_metric_rows.append(
                        row.get("case_name") or row.get("parent_case_id") or "<unknown>"
                    )
            if nonfinite_metric_rows:
                issues.append(
                    f"{len(nonfinite_metric_rows)} aggregated rows have non-finite "
                    f"{primary_metric} values"
                )

    failed_config_lines = _read_nonempty_lines(failed_configs)
    failed_job_lines = _read_nonempty_lines(failed_jobs)
    if failed_config_lines:
        issues.append(f"failed_case_configs.txt has {len(failed_config_lines)} entries")
    if failed_job_lines:
        issues.append(f"failed_job_ids.txt has {len(failed_job_lines)} entries")

    final_issues = issues
    reported_issues = [] if allow_partial else final_issues

    ranking_ready = not final_issues
    final_claim_ready = ranking_ready and split_diagnostics_complete is not False

    return {
        "analysis_dir": str(analysis_dir),
        "report_exists": report_path.exists(),
        "raw_rows": len(raw_rows),
        "aggregated_rows": len(agg_rows),
        "expected_valid_children": expected_valid_children,
        "expected_valid_parents": expected_valid_parents,
        "expected_folds": expected_folds,
        "primary_metric": primary_metric,
        "child_job_count_from_log": child_count,
        "valid_config_count_from_manifest": valid_count,
        "mapping_mismatch": mapping_mismatch,
        "state_counts": state_counts,
        "failure_reason_counts": failure_reason_counts,
        "metric_artifacts": metric_artifacts if isinstance(metric_artifacts, dict) else {},
        "primary_metric_complete": primary_metric_complete,
        "split_diagnostics_complete": split_diagnostics_complete,
        "raw_states": _counts(raw_rows, "state"),
        "aggregated_states": _counts(agg_rows, "state"),
        "raw_features": _counts(raw_rows, "feature_input"),
        "aggregated_features": _counts(agg_rows, "feature_input"),
        "raw_scalers": _counts(raw_rows, "scaler"),
        "aggregated_scalers": _counts(agg_rows, "scaler"),
        "raw_models": _counts(raw_rows, "model_type"),
        "aggregated_models": _counts(agg_rows, "model_type"),
        "aggregated_slice_counts": _tuple_counts(
            agg_rows, ("slice_count", "completed_slices", "failed_slices")
        ),
        "nonfinite_metric_rows": nonfinite_metric_rows[:25],
        "ranking_ready": ranking_ready,
        "final_claim_ready": final_claim_ready,
        "issues": reported_issues,
        "observations": final_issues + observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CheMLFlow analysis output integrity.")
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--expected-valid-children", type=int)
    parser.add_argument("--expected-valid-parents", type=int)
    parser.add_argument("--expected-folds", type=int)
    parser.add_argument(
        "--primary-metric", choices=("auc", "r2", "mae", "rmse", "accuracy", "f1", "auprc")
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Inspect a partial result without treating final-claim failures as unexpected.",
    )
    args = parser.parse_args()
    result = audit(
        args.analysis_dir,
        expected_valid_children=args.expected_valid_children,
        expected_valid_parents=args.expected_valid_parents,
        expected_folds=args.expected_folds,
        primary_metric=args.primary_metric,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
