# DOE Review Reference

## Files To Inspect

- DOE spec: `config/doe_*.yaml`, `doe/doe_*.yaml`, or user-provided YAML.
- Generated directory: `summary.json`, `manifest.jsonl`, `parent_manifest.jsonl`, generated `case_*.yaml`.
- Repo docs: `docs/doe.md`, `docs/doe_quickstart.md`, `docs/dataset_profile_support_matrix.md`.

## Compatibility Rules

- `smiles_native` is for SMILES-native models, usually `chemprop` and `chemeleon`.
- Tabular models need numeric feature inputs, usually `featurize.rdkit`, `featurize.morgan`, or curated features.
- If preprocessing is enabled in a mixed grid, expect SMILES-native models to skip scaler branches that are not meaningful.
- Keep `preprocess.scaler` as a DOE axis only when tabular features are part of the comparison.
- For local CSV classification, verify target labels are binary or mapped.
- For IC50-style regression, verify raw source columns needed for curation are preserved.

## Manifest Interpretation

- `summary.json` is the quickest source of total, valid, skipped, parent, profile, task, and DOE hash counts.
- `manifest.jsonl` is one row per attempted execution child.
- `parent_manifest.jsonl` is one row per scientific parent config.
- A valid 5-fold CV parent should usually map to 5 valid execution children.
- Skipped cases are expected in mixed grids. Summarize skip reasons before calling them errors.

## Red Flags

- Missing or ambiguous `target_column`, `smiles_column`, or dataset source path.
- Mixing holdout and CV axes in one DOE spec without a clear reason.
- Fixed `split.cv.fold_index` in a full-run DOE when the goal is all folds.
- Comparing Morgan and RDKit while one branch has fewer valid parents.
- Selecting a final model from many configs using one fixed test split.
- `constraints.isolate_case_artifacts: false` in a large DOE without an explicit reason.
