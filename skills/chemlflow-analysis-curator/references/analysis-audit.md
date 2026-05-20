# Analysis Audit Reference

## Core Files

- `report.json`: provenance, job counts, state counts, failure reasons, output paths.
- `jobs.csv`: one row per child job with state and config mapping.
- `all_runs_metrics_by_execution.csv`: one row per execution child with metrics.
- `all_runs_metrics.csv`: aggregated scientific parent rows.
- `generalization_gaps_by_execution.csv` and `generalization_gaps.csv`: train/test gap views.
- `failed_case_configs.txt` and `failed_job_ids.txt`: expected empty for complete final analyses.

## Expected Complete DOE Shape

For common CheMLFlow 5-fold DOE runs with 680 valid execution children:

- raw execution rows: `680`
- aggregated parent rows: `136`
- each aggregated row: `slice_count=5`, `completed_slices=5`, `failed_slices=0`
- non-native feature branches: Morgan and RDKit counts should match when both are in the DOE.
- SMILES-native models appear under `smiles_native` and should not be forced into Morgan/RDKit balance.

## Known Good Fixture Patterns

- PAH final run: 680 completed children, 136 aggregated rows, Morgan/RDKit balanced, scaler populated.
- PGP recovered run: 680 completed children after case recovery, 136 aggregated rows, Morgan/RDKit balanced, scaler populated.
- YSI final local fixture can contain one failed child in older artifacts; treat it as useful for failure-path testing, not a clean final template.

## Report Style

Lead with data integrity:

1. child count vs manifest count
2. completed/failed/running state counts
3. raw and aggregated row counts
4. slice completeness
5. scaler/feature/model/split balance
6. failures or missing files

Only after those checks pass should you summarize model performance.
