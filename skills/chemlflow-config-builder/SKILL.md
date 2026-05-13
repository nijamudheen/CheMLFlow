---
name: chemlflow-config-builder
description: Create or review a single CheMLFlow runtime config YAML. Use when Codex is asked to build one config for a dataset, choose regression vs classification settings, handle SMILES or non-SMILES CSV inputs, choose curation/drop-row rules, set split/CV/random seed/scaler/model options, or explain when a single config should become DOE fanout.
---

# CheMLFlow Config Builder

## Scope

Use this skill to create or audit one runnable CheMLFlow runtime config for `main.py`. Keep it separate from DOE design: a runtime config is one execution slice; DOE is for generating comparable sets of configs.

## Workflow

1. Find nearby examples before inventing a pattern. Prefer `config/config*.yaml`, `tutorials/**/configs/*.yaml`, `config/doe_*.yaml`, `docs/config-options.md`, and existing `run_config.yaml` artifacts.
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
7. Configure splitting for the scientific question:
   - Random split is fine for tutorials, smoke tests, and simple baselines.
   - Scaffold split is preferred when chemistry generalization matters.
   - Classification usually needs `stratify: true` and `stratify_column`.
   - Use a fixed `global.random_state` and `split.random_state`, usually `42`.
8. Configure preprocessing only when the selected feature path needs it:
   - `standard` is a conservative scaler for SVM and many tabular DL runs.
   - `none` is reasonable for tree models when scaling is not part of the experiment.
   - `minmax` should be deliberate, not accidental.
9. Validate the config before execution. Check node/block consistency, model/feature compatibility, target column existence, split settings, output paths, and whether analysis artifacts will include the fields the user needs.

## Cross-Validation Rule

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

## Fair Comparison Checks

- When comparing RDKit, Morgan, and SMILES-native branches, confirm that row coverage is comparable. Representation-specific row loss can invalidate model/feature comparisons.
- For DOE-style benchmarking, define the common curated molecule universe first, then vary representations, scalers, models, and splits.
- Do not hide dropped-row behavior. Report how missing SMILES, invalid SMILES, missing targets, unmapped labels, duplicate rows, and feature cleaning affect the train/val/test row universe.

## Useful References

- `docs/config-options.md` for runtime config schema and node rules.
- `references/config-examples.md` for known single-config and DOE-derived examples to imitate.
- `tutorials/01_single_config_colab/configs/pgp_svm_cv_fold0.yaml` for a concise single-config example.
- `config/config.pgp_chemprop.yaml` for binary classification with SMILES-native training.
- `config/config.chembl_cv.yaml` for ChEMBL IC50 to pIC50 regression.
- `config/doe_qm9.yaml` for DOE defaults that can inform a single config.
