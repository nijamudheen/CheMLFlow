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
- Before including `chemprop` or `chemeleon` in a local execution plan, run a fast dependency preflight in the active environment. Verify imports for `rdkit`, `torch`, `lightning`, `chemprop`, and Chemprop submodules (`data`, `featurizers`, `models`, `nn`), and record Python, OS, Torch, device availability, and Torch thread count.
- Treat dependency preflight as necessary but not sufficient. A SMILES-native branch is execution-proven only after one generated Chemprop/CheMeleon child completes through `scripts/run_doe_local.py` and the pilot analysis can read its metrics.
- Treat local native thread caps such as `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` as diagnostic or backend-specific runtime workarounds, not scientific defaults. Do not make them mandatory defaults in the DOE or CheMLFlow config unless a local smoke test shows they are needed or the user explicitly asks.
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
4. If `chemprop` or `chemeleon` are in scope, run the SMILES-native dependency preflight before generating or launching the full DOE:

```bash
python -c "import rdkit, torch, lightning, chemprop; from chemprop import data, featurizers, models, nn; print('chemprop stack ok', chemprop.__version__, torch.__version__, lightning.__version__)"
```

Also record the runtime context before launching local SMILES-native jobs:

```bash
python -c "import platform, torch; print(platform.platform(), platform.python_version(), 'torch_threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'mps', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())"
```

5. Make the scientific defaults explicit:
   - Quick molecular baseline: Morgan + random split.
   - Chemistry generalization: scaffold CV.
   - Representation comparison: Morgan and RDKit, with balanced row coverage.
   - Paper-relevant SMILES baseline: Chemprop from scratch when the original work used graph/message-passing models or when the user asks for best structure-based prediction.
   - Foundation-model baseline: CheMeleon only when a checkpoint is available in an allowed path or the user approves downloading it.
   - Final claims: CV, nested CV, or an untouched final holdout depending on the claim.
6. For broad model-selection on one dataset and one evaluation protocol, create one mixed DOE that varies model family, feature input, scaler, and split strategy together. Follow local broad DOE examples such as `doe_ysi.yaml` or `doe_pgp.yaml`; let `generate_doe.py` skip known-invalid combinations and preserve those skips in the manifest. Do not create separate tabular-vs-SMILES DOE specs unless the split mode/evaluation protocol differs, the user asks for staged runs, or a required resource such as a foundation checkpoint is intentionally excluded.
7. For any DOE that will be analyzed with `analysis.py`, configure split diagnostics:
   - Set `train.reporting.plot_split_performance: true` for benchmark/model-selection runs.
   - If the DOE intentionally omits split diagnostics, report the ranking as primary-metric-only and do not make overfit/underfit claims.
8. Generate a small pilot DOE or one execution child first.
9. Run locally with:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --limit 1 --stop-on-failure
```

For local DOE runs that include `chemprop` or `chemeleon`, make sure the pilot includes at least one valid SMILES-native execution child. If the generic `--limit 1` would only hit a tabular case, run or filter a generated SMILES-native child instead. The goal is to prove the native runtime, not just the tabular CheMLFlow path.

10. Analyze and audit the pilot before launching the full DOE:

```bash
python analysis.py --backend local --doe-dir <generated-doe-dir> --output-dir <analysis-dir>
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

If a local SMILES-native pilot fails with `SIGSEGV`, return code `-11`, a stale `run_status.json` stuck at `running`, or faulthandler output inside Chemprop/Torch native operations, retry the same generated child once with thread caps:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1 python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --max-workers 1 --resume
```

If that succeeds, report it as a local runtime workaround and run the SMILES-native branch serially with those env vars. Do not assume the same workaround is required on HPCC or any Slurm backend.

11. Run the full local DOE when the pilot executes, analyzes, and audits cleanly:

```bash
python scripts/run_doe_local.py --doe-dir <generated-doe-dir> --max-workers 1 --resume
```

For parallel local execution, use `--max-workers N --resume` without
`--stop-on-failure`. `--stop-on-failure` is for serial fail-fast debugging; do
not combine it with `--max-workers > 1`.

12. Analyze local results:

```bash
python analysis.py --backend local --doe-dir <generated-doe-dir> --output-dir <analysis-dir>
```

13. Audit the analysis before summarizing metrics:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py <analysis-dir>
```

## Slurm and HPCC Notes

- Run SMILES-native smoke tests in the actual compute environment, not only on the login node and not by assuming local Mac behavior transfers.
- Let the scheduler, module stack, Torch build, CPU allocation, and GPU availability determine whether thread caps are needed.
- If a Slurm/HPCC Chemprop smoke fails with the same native-thread signature seen locally, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in the Chemprop job environment only, then rerun the failed children and re-analyze.
- Keep local runtime workarounds out of scientific DOE factors unless the study is explicitly comparing runtime environments.

## Red Flags

- The agent trains directly with sklearn/PyTorch and presents those metrics as CheMLFlow results.
- A user asks for "5-fold CV" and the agent creates or runs only one runtime config without explaining it is one fold slice.
- A local run is launched with `--max-workers > 1 --stop-on-failure`; this is an invalid flag combination because fail-fast is serial.
- An agent kills or restarts a local DOE without first reporting active PIDs, completed cases, stale `running` statuses, and the proposed recovery command.
- A local Chemprop/CheMeleon crash is written off as bad data before checking return codes, logs, `run_status.json`, and whether a one-child native-runtime smoke can complete.
- Local runs require fake Slurm logs or fake `sacct` output.
- Local macOS thread-cap workarounds are copied into HPCC/Slurm jobs without rerunning a compute-node smoke test.
- Random split results are described as chemistry-generalization results.
- A single model-selection study is split into separate tabular and SMILES-native DOE specs even though the dataset, split mode, and selection metric are the same and `generate_doe.py` can record invalid combinations as skipped.
- Chemprop or CheMeleon is included in a local DOE without first recording package/import readiness and, for CheMeleon, checkpoint availability.
- Import success is described as proof that Chemprop training works. It is only a dependency preflight until one generated child completes and analyzes cleanly.
- A DOE compares Morgan/RDKit/scaffold branches with unbalanced or missing parent rows.
- A named molecular benchmark or source-paper dataset has SMILES but the DOE omits the source-paper model family, `chemprop`, or `chemeleon` without an explicit reason.
- Ranking happens before `report.json`, raw metrics rows, aggregate rows, and failed-case files are audited.
