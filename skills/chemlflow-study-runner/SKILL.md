---
name: chemlflow-study-runner
description: Coordinate end-to-end CheMLFlow studies across dataset profiling, runtime config design, DOE generation, local or Slurm execution, analysis, and audit. Use when a user asks an agent to run or improve a CheMLFlow experiment workflow rather than only build one config, review one DOE, or audit one analysis bundle.
---

# CheMLFlow Study Runner

## Purpose

Use this as the master CheMLFlow operating skill. It routes agents to the focused skills while keeping the workflow inside CheMLFlow's config, DOE, execution, and analysis system.

## Operating Principles

- Use CheMLFlow runtime configs, DOE generation, DOE execution, and `analysis.py` as the source of truth for scientific runs.
- Do not bypass CheMLFlow with ad hoc sklearn/PyTorch scripts unless the user explicitly asks for an external sanity check.
- Audit before ranking. Do not report "best model" claims until the analysis curator gate passes.
- Separate scientific parent configs from execution children. A runtime CV config is one fold/repeat slice; DOE fanout is the normal path for full K-fold results.
- Preserve generated configs, manifests, run statuses, logs, metrics, and analysis outputs.
- Ask or state assumptions for molecular science choices that change interpretation: Morgan vs RDKit, random vs scaffold, holdout vs CV vs nested CV, and whether SMILES-native models are in scope.
- If the dataset has a known source paper, benchmark, or public name, identify the original task, model family, split strategy, and reported metrics before DOE design. Include a feasible comparable baseline or explicitly label the study as incomplete relative to the literature.
- For SMILES molecular datasets, explicitly decide whether `chemprop` and `chemeleon` are in scope. Do not treat Morgan/RDKit tabular models as the complete search space unless the user asked to exclude SMILES-native models or the runtime cannot support them.
- Respect filesystem scope. Do not search the user's broader computer for checkpoints or datasets. Check only the active repo/workspace, paths already present in config files, and paths the user explicitly provides. Do not scan home, `/Users`, project collections, external drives, or USB folders unless the user names that location.

## Routing

1. For one runtime config, use `skills/chemlflow-config-builder`.
2. For a comparison, benchmark, or K-fold result, use `skills/chemlflow-doe-designer`.
3. For local execution, use `scripts/run_doe_local.py`.
4. For Slurm execution, use the repo's Slurm submit/orchestration workflow when present.
5. For local analysis, use `analysis.py --backend local`.
6. For Slurm analysis, use `analysis.py --backend slurm` with the orchestrator job/log inputs.
7. For final result validation, use `skills/chemlflow-analysis-curator`.

## Default Study Flow

1. Profile the dataset: rows, columns, target, SMILES column, missing values, invalid SMILES, duplicates/conflicts, class balance or target distribution.
2. Check source context when the dataset name or columns suggest a known benchmark. Look for local README/citations first; browse only when current source context or citations are needed.
3. Discover local CheMLFlow capability before fixing the DOE space: inspect docs/config examples for supported `train.model.type`, `pipeline.feature_input`, `chemprop`, `chemeleon`, and configured foundation checkpoint paths inside the active repo/workspace only.
4. Make the scientific defaults explicit:
   - Quick molecular baseline: Morgan + random split.
   - Chemistry generalization: scaffold CV.
   - Representation comparison: Morgan and RDKit, with balanced row coverage.
   - Paper-relevant SMILES baseline: Chemprop from scratch when the original work used graph/message-passing models or when the user asks for best structure-based prediction.
   - Foundation-model baseline: CheMeleon only when a checkpoint is available in an allowed path or the user approves downloading it.
   - Final claims: CV, nested CV, or an untouched final holdout depending on the claim.
5. For any DOE that will be analyzed with `analysis.py`, configure split diagnostics:
   - Set `train.reporting.plot_split_performance: true` for benchmark/model-selection runs.
   - If the DOE intentionally omits split diagnostics, report the ranking as primary-metric-only and do not make overfit/underfit claims.
6. Generate a small pilot DOE or one execution child first.
7. Run locally with:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --limit 1 --stop-on-failure
```

8. Analyze and audit the pilot before launching the full DOE:

```bash
python analysis.py --backend local --doe-dir <generated-doe-dir> --output-dir <analysis-dir>
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

9. Run the full local DOE when the pilot executes, analyzes, and audits cleanly:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --max-workers 1 --resume
```

For parallel local execution, use `--max-workers N --resume` without
`--stop-on-failure`. `--stop-on-failure` is for serial fail-fast debugging; do
not combine it with `--max-workers > 1`.

10. Analyze local results:

```bash
python analysis.py --backend local --doe-dir <generated-doe-dir> --output-dir <analysis-dir>
```

11. Audit the analysis before summarizing metrics:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

## Red Flags

- The agent trains directly with sklearn/PyTorch and presents those metrics as CheMLFlow results.
- A user asks for "5-fold CV" and the agent creates or runs only one runtime config without explaining it is one fold slice.
- A local run is launched with `--max-workers > 1 --stop-on-failure`; this is an invalid flag combination because fail-fast is serial.
- An agent kills or restarts a local DOE without first reporting active PIDs, completed cases, stale `running` statuses, and the proposed recovery command.
- Local runs require fake Slurm logs or fake `sacct` output.
- Random split results are described as chemistry-generalization results.
- A DOE compares Morgan/RDKit/scaffold branches with unbalanced or missing parent rows.
- A named molecular benchmark or source-paper dataset has SMILES but the DOE omits the source-paper model family, `chemprop`, or `chemeleon` without an explicit reason.
- Ranking happens before `report.json`, raw metrics rows, aggregate rows, and failed-case files are audited.
