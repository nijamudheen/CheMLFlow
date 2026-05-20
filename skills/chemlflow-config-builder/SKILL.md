---
name: chemlflow-config-builder
description: Create or review a single CheMLFlow runtime config YAML. Use when Codex is asked to build one config for a dataset, choose regression vs classification settings, handle SMILES or non-SMILES CSV inputs, choose curation/drop-row rules, set split/CV/random seed/scaler/model options, or explain when a single config should become DOE fanout.
---

# CheMLFlow Config Builder

## Scope

Use this skill to create or audit one runnable CheMLFlow runtime config for `main.py`. Keep it separate from DOE design: a runtime config is one execution slice; DOE is for generating comparable sets of configs.

## Workflow

1. Find nearby examples before inventing a pattern. Prefer `config/config*.yaml`, `tutorials/**/configs/*.yaml`, `config/doe_*.yaml`, `docs/config-options.md`, and existing `run_config.yaml` artifacts. For SMILES-native configs, search existing files for `smiles_native`, `chemprop`, `chemeleon`, `foundation`, and `foundation_checkpoint`.
   - Keep checkpoint discovery scoped to the active repo/workspace and user-provided paths. Do not search the user's broader computer, home directory, project collections, external drives, or USB folders unless the user explicitly names that location.
2. Inspect the dataset when available: columns, row count, missing target values, candidate SMILES columns, duplicate SMILES, label values/class balance, and numeric feature columns.
3. Classify the dataset shape:
   - **SMILES molecular CSV**: use `curate`, molecular featurization, and SMILES-aware splits/models.
   - **Precomputed tabular features**: consider `featurize.none`, `curate.keep_all_columns: true`, and `train.features.exclude_columns`; do not use RDKit, Morgan, scaffold split, Chemprop, or CheMeleon.
   - **Generic/time-series/single-column data**: do not force a molecular workflow. Ask for the target, feature columns, and intended prediction task before creating a config.
4. Choose `global.task_type`:
   - `regression` for continuous targets.
   - `classification` for categorical or binary targets. Add `label.normalize` when labels need mapping to 0/1.
5. Set curation explicitly. For molecular configs, default to:
   - `drop_missing_smiles: true`
   - `drop_invalid_smiles: true`
   - `drop_missing_target: true`
   - an intentional `dedupe_strategy`, commonly `drop_conflicts` for classification.
6. Choose a feature/model pair that is valid:
   - Tabular ML/DL: `featurize.rdkit`, `featurize.morgan`, or `featurize.none`.
   - SMILES-native models: `pipeline.feature_input: smiles_native` with `chemprop` or `chemeleon`.
   - Chemprop from scratch: use `train.model.type: chemprop`, `pipeline.feature_input: smiles_native`, and no descriptor-generation branch.
   - CheMeleon: use `train.model.type: chemeleon` or Chemprop with `train.model.foundation: chemeleon`; require a real local `foundation_checkpoint` path before promising the run.
7. If the config uses `chemprop` or `chemeleon` and the user expects local execution, run a dependency preflight in the active environment before saying it is runnable:

```bash
python -c "import rdkit, torch, lightning, chemprop; from chemprop import data, featurizers, models, nn; print('chemprop stack ok', chemprop.__version__, torch.__version__, lightning.__version__)"
```

   - Record whether the run is CPU-only or has CUDA/MPS available if runtime matters.
   - For CheMeleon, also verify `train.model.foundation_checkpoint` points to an existing `.pt` file inside an allowed path or a user-provided path.
   - Import success is a dependency preflight, not a completed training proof. Use one generated runtime child or a short smoke config to prove execution.
8. Configure splitting for the scientific question:
   - For a quick single run, prefer a holdout split unless the user asked for CV or benchmark-style comparison. Explain that holdout trains on one training split and evaluates on one held-out test split; it is not cross-validation.
   - Random holdout split is fine for tutorials, smoke tests, and simple baselines.
   - Scaffold split is preferred when chemistry generalization matters.
   - Classification usually needs `stratify: true` and `stratify_column`.
   - Use a fixed `global.random_state` and `split.random_state`, usually `42`.
9. Configure preprocessing only when the selected feature path needs it:
   - `standard` is a conservative scaler for SVM and many tabular DL runs.
   - `none` is reasonable for tree models when scaling is not part of the experiment.
   - `minmax` should be deliberate, not accidental.
10. Validate the config before execution. Check node/block consistency, model/feature compatibility, target column existence, split settings, output paths, and whether analysis artifacts will include the fields the user needs.

## Scientific Defaults Checkpoint

Before finalizing a molecular config, ask a concise clarification or state the assumptions when these choices are ambiguous:

- **Morgan vs RDKit**: Morgan fingerprints are a compact, common baseline for tree models; RDKit descriptors are a physicochemical descriptor baseline. Use both in DOE when representation sensitivity matters.
- **Tabular vs SMILES-native**: if a molecular CSV has a SMILES column, explicitly include or exclude Chemprop. Include Chemprop when the source literature used graph/message-passing models or when the goal is "best model from structure." Consider CheMeleon when a checkpoint is available.
- **Random vs scaffold split**: random splits are useful for quick baselines and interpolation-style prediction; scaffold splits are preferred for chemistry generalization to new molecular families.
- **Single config vs DOE**: one runtime CV config is one fold slice. Use DOE fanout for a full K-fold estimate.
- **CheMLFlow operating system**: train through CheMLFlow configs, DOE, and analysis unless the user explicitly asks for an external sanity check. Do not silently bypass CheMLFlow with ad hoc sklearn scripts for scientific results.

Useful default language: "I will use Morgan + random split as a quick baseline unless you want RDKit descriptors, Chemprop/CheMeleon, scaffold CV, or all of those represented in a DOE."

## Cross-Validation Rule

Do not assume the user knows to ask for cross-validation. If they ask for one quick runnable config, say plainly: "This is a holdout run: it trains once and evaluates once on a held-out test split. It is useful for a quick check, but it is not cross-validation. For a stronger full CV estimate, we should generate sibling fold configs with DOE fanout."

A CheMLFlow runtime config runs one CV fold slice. The `n_splits` setting says how the shared fold plan is constructed; `fold_index` says which one fold this run uses as the test fold:

```yaml
split:
  mode: cv
  strategy: random
  cv:
    n_splits: 5
    repeats: 1
    fold_index: 0
    repeat_index: 0
```

Explain this plainly to the user: "This config runs fold 0. It does not run all five folds. For a full 5-fold result, we should use DOE fanout so folds 0-4 are generated as sibling execution configs from the same scientific parent."

Prefer DOE fanout for full K-fold results. Only create five standalone config files when the user explicitly wants independent files and understands they are execution slices of the same design.

When using examples, prefer `rep0_fold0` examples for model/task config shape. Use sibling `fold1`-`fold4` configs only to explain CV fanout or repair a missing execution slice; do not treat different folds as different scientific configurations.

## Fair Comparison Checks

- When comparing RDKit, Morgan, and SMILES-native branches, confirm that row coverage is comparable. Representation-specific row loss can invalidate model/feature comparisons.
- For DOE-style benchmarking, define the common curated molecule universe first, then vary representations, scalers, models, and splits.
- Do not hide dropped-row behavior. Report how missing SMILES, invalid SMILES, missing targets, unmapped labels, duplicate rows, and feature cleaning affect the train/val/test row universe.
- For Chemprop/CheMeleon, verify the config has an explicit validation split such as `split.val_from_train.val_size: 0.1`; CheMLFlow's Chemprop path needs train/val/test partitions.
- For Chemprop/CheMeleon local runs, package preflight must include `rdkit`, `torch`, `lightning`, `chemprop`, and Chemprop submodule imports. Do not rely on package installation alone when making a final execution claim.
- For CheMeleon, verify the checkpoint exists before execution using only allowed paths. If it is absent, ask the user before downloading from Zenodo:

```bash
mkdir -p models
curl -L https://zenodo.org/records/15460715/files/chemeleon_mp.pt -o models/chemeleon_mp.pt
```

After download, set `train.model.foundation_checkpoint: models/chemeleon_mp.pt`. If the user declines download, mark the CheMeleon branch skipped, not failed.

## Useful References

- `docs/config-options.md` for runtime config schema and node rules.
- `references/config-examples.md` for canonical submitted single-config examples covering regression/classification by Chemprop, CheMeleon, random forest, and ensemble.
- `tutorials/01_single_config_colab/configs/pgp_svm_cv_fold0.yaml` for a tracked minimal single-config tutorial example.
