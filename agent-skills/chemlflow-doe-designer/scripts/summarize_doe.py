#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _factor(rec: dict[str, Any], key: str) -> str:
    factors = rec.get("factors") if isinstance(rec.get("factors"), dict) else {}
    return str(factors.get(key, ""))


def summarize(doe_dir: Path) -> dict[str, Any]:
    summary_path = doe_dir / "summary.json"
    manifest_path = doe_dir / "manifest.jsonl"
    parent_manifest_path = doe_dir / "parent_manifest.jsonl"

    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest = _load_jsonl(manifest_path)
    parents = _load_jsonl(parent_manifest_path)
    valid = [r for r in manifest if str(r.get("status", "")).lower() == "valid"]
    skipped = [r for r in manifest if str(r.get("status", "")).lower() == "skipped"]

    issue_counts: Counter[str] = Counter()
    for rec in skipped:
        for issue in rec.get("issues") or []:
            if isinstance(issue, dict):
                issue_counts[str(issue.get("code", ""))] += 1
            else:
                issue_counts[str(issue)] += 1

    return {
        "doe_dir": str(doe_dir),
        "summary_exists": summary_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "parent_manifest_exists": parent_manifest_path.exists(),
        "profile": summary.get("profile"),
        "task_type": summary.get("task_type"),
        "doe_spec_hash": summary.get("doe_spec_hash"),
        "total_cases": summary.get("total_cases", len(manifest)),
        "valid_cases": summary.get("valid_cases", len(valid)),
        "skipped_cases": summary.get("skipped_cases", len(skipped)),
        "total_parent_cases": summary.get("total_parent_cases", len(parents)),
        "valid_parent_cases": summary.get(
            "valid_parent_cases",
            sum(1 for r in parents if str(r.get("status", "")).lower() == "valid"),
        ),
        "status_counts": dict(Counter(str(r.get("status", "")) for r in manifest)),
        "issue_counts": dict(issue_counts),
        "model_counts_valid": dict(Counter(_factor(r, "train.model.type") for r in valid)),
        "feature_counts_valid": dict(Counter(_factor(r, "pipeline.feature_input") for r in valid)),
        "scaler_counts_valid": dict(Counter(_factor(r, "preprocess.scaler") for r in valid)),
        "split_counts_valid": dict(Counter(_factor(r, "split.strategy") for r in valid)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize generated CheMLFlow DOE artifacts.")
    parser.add_argument("doe_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.doe_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
