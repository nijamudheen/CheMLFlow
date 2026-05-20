---
name: chemlflow-analysis-curator
description: Audit and curate CheMLFlow analysis outputs. Use when Codex is asked to validate `analysis.py` results, inspect `report.json`, compare `all_runs_metrics.csv` with `all_runs_metrics_by_execution.csv`, verify scaler/feature/model/split balance, find failed or incomplete folds, or decide whether a CheMLFlow result bundle is complete and trustworthy.
---

# CheMLFlow Analysis Curator

## Overview

Use this skill to prove a CheMLFlow analysis result is complete, balanced, and interpretable. The core habit is to compare manifest counts, job state counts, raw execution rows, aggregated parent rows, and feature/scaler/model/split distributions before trusting metrics.

## Workflow

1. Locate the analysis directory containing `report.json`, `all_runs_metrics.csv`, and `all_runs_metrics_by_execution.csv`.
2. Read `report.json` first for provenance, child count, valid manifest count, mapping mismatch, state counts, and failure reasons.
3. Compare raw execution rows to expected valid child count.
4. Compare aggregated rows to expected valid parent count when parent data is available.
5. Check every aggregated CV row has complete slices, usually `slice_count=5`, `completed_slices=5`, `failed_slices=0`.
6. Verify `scaler`, `feature_input`, `model_type`, and `split_strategy` distributions are balanced for the DOE design.
7. Separate expected special cases from problems. For example, `chemprop` and `chemeleon` usually appear under `smiles_native`, not Morgan/RDKit.
8. Report failures, missing statuses, incomplete folds, row-count mismatches, and suspicious metric path gaps before discussing best models.

## Standard Checks

- `child_job_count_from_log` equals `valid_config_count_from_manifest`.
- `mapping_mismatch` is false.
- `state_counts` contains only `COMPLETED` unless the user explicitly asks to inspect a partial run.
- `failure_reason_counts` is empty for final results.
- Raw metrics rows equal completed execution children with metrics.
- Aggregated metrics rows equal completed scientific parents with complete slices.
- `scaler` is present and populated when the DOE varied `preprocess.scaler`.
- Morgan and RDKit row counts match for non-native models when both feature branches are in the DOE.
- Split strategy counts match for random/scaffold comparisons.
- `failed_case_configs.txt` and `failed_job_ids.txt` are empty for final complete runs.

## Useful Commands

Audit an analysis directory:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

Run analysis after jobs finish, adapting paths to the user's environment:

```bash
python analysis.py --orchestrator-job-id <job-id> --orchestrator-log-prefix <prefix> --logs-dir <logs-dir> --doe-dir <doe-dir> --output-dir <analysis-dir>
```

## References

- For detailed audit patterns and known CheMLFlow result shapes, read `references/analysis-audit.md`.
- For code behavior, inspect `analysis.py` and generated `report.json` rather than inferring from filenames alone.
