# Config Examples

Use these examples as shape references before creating a new runtime config.

## Single Runtime Configs

- `tutorials/01_single_config_colab/configs/pgp_svm_cv_fold0.yaml`
  - PGP binary classification.
  - Local CSV, Morgan fingerprints, `preprocess.scaler: standard`.
  - `split.mode: cv`, `n_splits: 5`, `fold_index: 0`.
  - Good minimal example for one runnable fold slice.

## Canonical Submitted Runtime Config Examples

These are HPCC-submitted one-fold runtime configs from completed DOE families. Use them to understand model/task config shape, not as portable paths for a clean clone. They are all `rep0_fold0` examples on purpose, so fold choice is held constant while the model and task vary.

1. Regression, Chemprop, FLASH:
   `/mnt/home/f0113398/flash_doe/case_0026__reg_local_csv__chemprop__cv__random__rep0__fold0.yaml`
   - job `5628319`
   - SMILES-native config: `pipeline.feature_input: smiles_native`, `train.model.type: chemprop`

2. Regression, CheMeleon, FLASH:
   `/mnt/home/f0113398/flash_doe/case_0031__reg_local_csv__chemeleon__cv__random__rep0__fold0.yaml`
   - job `5628325`
   - SMILES-native config: `pipeline.feature_input: smiles_native`, `train.model.type: chemeleon`

3. Regression, random forest, FLASH:
   `/mnt/home/f0113398/flash_doe/case_0391__reg_local_csv__random_forest__cv__random__rep0__fold0.yaml`
   - job `5628344`
   - tabular molecular config, usually RDKit or Morgan features with `train.model.type: random_forest`

4. Regression, ensemble, FLASH:
   `/mnt/home/f0113398/flash_doe/case_0411__reg_local_csv__ensemble__cv__random__rep0__fold0.yaml`
   - job `5628365`
   - tabular molecular config, usually RDKit or Morgan features with `train.model.type: ensemble`

5. Classification, Chemprop, PGP:
   `/mnt/home/f0113398/pgp_doe/case_0026__clf_local_csv__chemprop__cv__random__rep0__fold0.yaml`
   - job `5342097`
   - SMILES-native config: `pipeline.feature_input: smiles_native`, `train.model.type: chemprop`

6. Classification, CheMeleon, PGP:
   `/mnt/home/f0113398/pgp_doe/case_0031__clf_local_csv__chemeleon__cv__random__rep0__fold0.yaml`
   - job `5342102`
   - SMILES-native config: `pipeline.feature_input: smiles_native`, `train.model.type: chemeleon`

7. Classification, random forest, PGP:
   `/mnt/home/f0113398/pgp_doe/case_0391__clf_local_csv__random_forest__cv__random__rep0__fold0.yaml`
   - job `5342124`
   - tabular molecular config, usually RDKit or Morgan features with `train.model.type: random_forest`

8. Classification, ensemble, PGP:
   `/mnt/home/f0113398/pgp_doe/case_0411__clf_local_csv__ensemble__cv__random__rep0__fold0.yaml`
   - job `5342144`
   - tabular molecular config, usually RDKit or Morgan features with `train.model.type: ensemble`

## Fold Families

The eight examples above are single execution slices. Full 5-fold CV for the same scientific parent is represented by sibling runtime configs, usually with adjacent case numbers and `rep0_fold0` through `rep0_fold4`.

Common submitted fold-family patterns:

- Chemprop: `case_0026` through `case_0030`
- CheMeleon: `case_0031` through `case_0035`
- Random forest: `case_0391` through `case_0395`
- Ensemble: `case_0411` through `case_0415`

For example, regression Chemprop FLASH uses:

- `/mnt/home/f0113398/flash_doe/case_0026__reg_local_csv__chemprop__cv__random__rep0__fold0.yaml`
- `/mnt/home/f0113398/flash_doe/case_0027__reg_local_csv__chemprop__cv__random__rep0__fold1.yaml`
- `/mnt/home/f0113398/flash_doe/case_0028__reg_local_csv__chemprop__cv__random__rep0__fold2.yaml`
- `/mnt/home/f0113398/flash_doe/case_0029__reg_local_csv__chemprop__cv__random__rep0__fold3.yaml`
- `/mnt/home/f0113398/flash_doe/case_0030__reg_local_csv__chemprop__cv__random__rep0__fold4.yaml`

Use `fold0` examples for ordinary config construction. Use the fold-family pattern only when explaining or generating full CV fanout.

## DOE Specs That Inform Single Configs

If local DOE specs are present, use them only to infer defaults and fanout shape. Useful patterns include:

- QM9-style regression DOE: local CSV regression defaults, scaler choices, feature branches, and CV fanout shape.
- PGP-style classification DOE: binary label handling and comparison of SMILES-native, RDKit, and Morgan branches.
- Urease/IC50-style DOE: ChEMBL-derived pIC50 experiments and full K-fold parent/child expansion.

## How To Use These

- Copy structure, not paths or labels blindly.
- Confirm the dataset columns before choosing `smiles_column`, `target_column`, and `label_column`.
- For a quick single run, prefer a holdout split and tell the user it trains once and evaluates once on one held-out test split.
- Tell the user a holdout split is not cross-validation. Offer DOE fanout when they need full K-fold evidence.
- For one-off debugging, use one runtime config with one `fold_index`.
- For full K-fold results, use DOE fanout instead of manually writing many near-identical configs.
- Do not call one `fold0` runtime config a complete 5-fold result.
