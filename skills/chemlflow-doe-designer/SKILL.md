---
name: chemlflow-doe-designer
description: Design and review CheMLFlow DOE YAMLs and generated DOE artifacts. Use when Codex is asked to create, modify, or audit CheMLFlow DOE specs, search spaces, model/feature/scaler/split compatibility, manifest skip reasons, parent/child CV shape, or expected valid/skipped case counts.
---

# CheMLFlow DOE Designer

## Overview

Use this skill as a small operating manual for CheMLFlow DOE work. Keep the focus on experiment validity: compatible axes, predictable manifest shape, auditable configs, and scientifically meaningful split/evaluation design.

## Workflow

1. Locate the DOE spec, usually `config/doe_*.yaml`, `doe/doe_*.yaml`, or a user-provided YAML.
2. Read `docs/doe.md` only if the repo behavior is unfamiliar or the DOE uses a less common profile.
3. Inspect `dataset`, `defaults`, `search_space`, `constraints`, `selection`, and `output`.
4. Check model/feature/scaler/split compatibility before recommending a run.
5. If generated artifacts exist, inspect `summary.json`, `manifest.jsonl`, and `parent_manifest.jsonl`.
6. Report expected run shape: total attempted children, valid children, skipped children, valid scientific parents, and major skip reasons.
7. Choose or state the execution backend: local DOE runner for workstation runs, Slurm submission for HPCC runs.
8. Call out scientific risks separately from syntax risks.
9. For known benchmark/source-paper datasets, check that the DOE includes the source-paper model family when feasible. If not, mark the DOE as a narrower baseline study.

## Checks

- Keep fixed choices in `defaults`; keep only true experiment axes in `search_space`.
- Treat DOE as parent/child shaped: one scientific parent can expand to many execution children, usually CV folds.
- For CV runs, expect all folds/repeats to be generated unless fold/repeat indices are intentionally fixed for debugging.
- Treat `split.cv.fold_index`, `split.cv.repeat_index`, `split.inner.fold_index`, and `split.inner.repeat_index` as execution coordinates. Keep them out of `search_space`; omit them for full CV fanout or put them in `defaults` only for targeted debug/retry slices.
- Prefer separate DOE specs for holdout, CV, and nested holdout CV.
- Treat `smiles_native` as reserved for SMILES-native models such as `chemprop` and `chemeleon`.
- Expect tabular models to use `featurize.rdkit`, `featurize.morgan`, or curated numeric features, not raw SMILES.
- Expect `chemprop` and `chemeleon` to reject ordinary preprocessing/scaler branches except meaningful no-op branches.
- Keep compatibility groups explicit when a DOE mixes model families:
  - `pipeline.feature_input: smiles_native` with `train.model.type: chemprop` or `chemeleon`, usually `preprocess.scaler: none`.
  - `pipeline.feature_input: featurize.rdkit` or `featurize.morgan` with tabular models such as random forest, SVM, XGBoost, ensemble, and CatBoost.
  - Do not rely on a flat Cartesian product unless constraints are known to prune invalid feature/model pairs cleanly.
- Include Chemprop from scratch for SMILES datasets when the original benchmark used graph/message-passing models. Include CheMeleon when a valid checkpoint is available and the user has not excluded foundation baselines.
- If `chemeleon` is in the DOE, set `train.model.foundation_checkpoint` to an existing allowed path and ensure generated CheMeleon configs carry `foundation: chemeleon` when that is the repo's convention.
- Do not search the user's broader computer for `chemeleon_mp.pt`. Check only the active repo/workspace, paths already in the DOE/config, and user-provided paths. If no checkpoint is found, ask whether to download it from `https://zenodo.org/records/15460715/files/chemeleon_mp.pt`; if not, skip the CheMeleon branch cleanly.
- For comparison studies, check that Morgan/RDKit/scaler/split rows are balanced across non-native models.
- For final claims, prefer CV or nested holdout CV over selecting many configs on one fixed test split.
- For benchmark/model-selection DOEs that will be consumed by `analysis.py`, set
  `train.reporting.plot_split_performance: true`. Without split metrics, top-level
  primary metrics can still rank models, but overfit/underfit diagnostics and final
  generalization claims are intentionally limited.
- Pilot the full analysis path before the full DOE: generate the DOE, run one valid
  execution child, run `analysis.py`, then run the analysis curator audit. A training-only
  smoke test is not enough.
- For local execution, use `scripts/run_doe_local.py`; do not invent fake Slurm logs or fake `sacct` output.
- Use `--stop-on-failure` only for serial debug runs. Do not combine it with
  `--max-workers > 1`; full parallel local DOE runs should use
  `--max-workers N --resume` and then inspect `execution_manifest.jsonl`.
- For local analysis after local execution, use `analysis.py --backend local`.

## Useful Commands

Summarize generated DOE artifacts:

```bash
python skills/chemlflow-doe-designer/scripts/summarize_doe.py <generated-doe-dir>
```

Generate DOE configs from a spec only when the user has asked for execution or validation that requires it. This expands a DOE spec into concrete runtime config files and manifests; it does not train models:

```bash
python scripts/generate_doe.py --doe config/doe.example.yaml
```

Run generated valid execution children locally:

```bash
python scripts/run_doe_local.py --doe-dir config/generated/example_doe --max-workers 1 --resume
```

Run generated children locally in parallel:

```bash
python scripts/run_doe_local.py --doe-dir config/generated/example_doe --max-workers 4 --resume
```

Run a serial fail-fast debug slice:

```bash
python scripts/run_doe_local.py --doe-dir config/generated/example_doe --limit 1 --stop-on-failure
```

Analyze local DOE outputs without Slurm:

```bash
python analysis.py --backend local --doe-dir config/generated/example_doe --output-dir config/generated/example_doe/analysis_local
```

## References

- For detailed review prompts and expected red flags, read `references/doe-review.md`.
- For canonical repo docs, prefer `docs/doe.md`, `docs/doe_quickstart.md`, and `docs/dataset_profile_support_matrix.md`.
