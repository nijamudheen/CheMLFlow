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
8. For source-paper or named benchmark datasets, check whether expected literature baselines are present. For SMILES benchmarks this often means `chemprop`; for foundation comparisons this may mean `chemeleon`.
9. If Chemprop/CheMeleon baselines are missing, distinguish dependency/checkpoint unavailability from scientific exclusion. Dependency preflight success is not a metric result; completed SMILES-native execution children are required before ranking those baselines.
10. Report failures, missing statuses, incomplete folds, row-count mismatches, suspicious metric path gaps, non-finite ranking metrics, and missing expected baselines before discussing best models.
11. Distinguish primary-metric completeness from split-diagnostic completeness. Missing split metrics limits overfit/underfit conclusions, but should not block ranking if every completed aggregate row has finite top-level primary metrics and the audit reports `ranking_ready: true`.
12. Treat final benchmark/generalization claims as blocked unless the audit reports `final_claim_ready: true`.
13. Identify the execution backend. Local DOE analysis should come from `analysis.py --backend local`, not fabricated Slurm logs or fake `sacct` rows.

## Standard Checks

- `report.json` exists and contains required provenance fields.
- `child_job_count_from_log` equals `valid_config_count_from_manifest`.
- `mapping_mismatch` is false.
- `state_counts` contains only `COMPLETED` unless the user explicitly asks to inspect a partial run.
- `failure_reason_counts` is empty for final results.
- Raw metrics rows equal completed execution children with metrics.
- Aggregated metrics rows equal completed scientific parents with complete slices.
- `scaler` is present and populated when the DOE varied `preprocess.scaler`.
- Morgan and RDKit row counts match for non-native models when both feature branches are in the DOE.
- Chemprop/CheMeleon rows appear under `feature_input=smiles_native` when those baselines are in scope.
- Chemprop/CheMeleon dependency preflight notes, checkpoint availability, and skipped-baseline rationale are consistent with the generated manifest and completed execution rows.
- If a final answer compares against a source paper, the analysis either includes a comparable model family or explicitly says the study is only a classical/tabular baseline.
- Split strategy counts match for random/scaffold comparisons.
- `failed_case_configs.txt` and `failed_job_ids.txt` are empty for final complete runs.
- Ranking metrics are finite in every complete aggregate row used for a final claim.
- `report.metric_artifacts.split_diagnostics_complete` is true when making overfit/underfit or train/test generalization-gap claims. If it is false but primary metrics are complete, report ranking as primary-metric-only.

The rule is audit first, rank second. If the audit script exits nonzero, do not summarize "best models" as final results.

For local runs, `child_job_count_from_log` is retained as a compatibility count in `report.json`; confirm `backend: local`, `execution_manifest_path`, `execution_count`, `local_attempt_count`, and `valid_config_count_from_manifest` are consistent.

## Useful Commands

Audit an analysis directory:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

For a common 5-fold DOE with 680 valid child executions and 136 scientific parent rows:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir> \
  --expected-valid-children 680 \
  --expected-valid-parents 136 \
  --expected-folds 5 \
  --primary-metric r2
```

Run analysis after jobs finish, adapting paths to the user's environment:

```bash
python analysis.py --orchestrator-job-id <job-id> --orchestrator-log-prefix <prefix> --logs-dir <logs-dir> --doe-dir <doe-dir> --output-dir <analysis-dir>
```

Run local analysis after `scripts/run_doe_local.py` or manually executed generated configs:

```bash
python analysis.py --backend local --doe-dir <doe-dir> --output-dir <analysis-dir>
```

## References

- For detailed audit patterns and known CheMLFlow result shapes, read `references/analysis-audit.md`.
- For code behavior, inspect `analysis.py` and generated `report.json` rather than inferring from filenames alone.
