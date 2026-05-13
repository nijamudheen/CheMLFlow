# Config Examples

Use these repo examples as shape references before creating a new runtime config.

## Single Runtime Configs

- `tutorials/01_single_config_colab/configs/pgp_svm_cv_fold0.yaml`
  - PGP binary classification.
  - Local CSV, Morgan fingerprints, `preprocess.scaler: standard`.
  - `split.mode: cv`, `n_splits: 5`, `fold_index: 0`.
  - Good minimal example for one runnable fold slice.

- `config/config.pgp_chemprop.yaml`
  - PGP binary classification.
  - Local CSV, `pipeline.feature_input: smiles_native`, `train.model.type: chemprop`.
  - Good example for SMILES-native training without tabular featurization.

- `config/config.chembl_cv.yaml`
  - ChEMBL urease-style regression.
  - Fetches ChEMBL IC50 data, filters assay rows, converts IC50 to pIC50 with `label.ic50`.
  - Good example for `required_non_null_columns`, `row_filters`, scaffold CV, and pIC50 targets.

## DOE Specs That Inform Single Configs

- `config/doe_qm9.yaml`
  - QM9 regression DOE.
  - Useful for local CSV regression defaults, scaler choices, feature branches, and CV fanout shape.

- `config/doe_pgp.yaml`
  - PGP classification DOE.
  - Useful for binary label handling and comparing SMILES-native, RDKit, and Morgan branches.

- `config/doe_urease.yaml`
  - Urease/IC50 DOE.
  - Useful for ChEMBL-derived pIC50 experiments and full K-fold parent/child expansion.

## How To Use These

- Copy structure, not paths or labels blindly.
- Confirm the dataset columns before choosing `smiles_column`, `target_column`, and `label_column`.
- For one-off debugging, use one runtime config with one `fold_index`.
- For full K-fold results, use DOE fanout instead of manually writing many near-identical configs.
