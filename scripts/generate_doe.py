#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilities.doe import DOEGenerationError, generate_doe, load_doe_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CheMLFlow runtime config files from a single DOE definition file."
        )
    )
    parser.add_argument(
        "--doe",
        required=True,
        help="Path to DOE YAML file (version: 1).",
    )
    parser.add_argument(
        "--print-valid",
        action="store_true",
        help="Print generated config paths for valid cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        spec = load_doe_spec(args.doe)
        result = generate_doe(spec, doe_path=args.doe)
    except DOEGenerationError as exc:
        print(f"[doe] error: {exc}", file=sys.stderr)
        return 2

    summary = result["summary"]
    print(
        "[doe] profile={profile} task={task} total={total} valid={valid} skipped={skipped}".format(
            profile=summary["profile"],
            task=summary["task_type"],
            total=summary["total_cases"],
            valid=summary["valid_cases"],
            skipped=summary["skipped_cases"],
        )
    )
    print(f"[doe] manifest: {result['manifest_path']}")
    print(f"[doe] summary : {result['summary_path']}")
    if args.print_valid:
        for record in result["valid_cases"]:
            print(record["config_path"])

    # Emit machine-readable summary to stdout for wrappers.
    print(json.dumps({"summary_path": result["summary_path"], "manifest_path": result["manifest_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
