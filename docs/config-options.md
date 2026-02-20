# CheMLFlow Config Options Reference

This reference documents all configuration options for CheMLFlow pipelines. Use it to understand what each setting does, what values are valid, and what defaults apply.

**Quick start:** Copy an existing config from `config/` and modify it. This reference helps when you need to customize beyond the examples.

## Schema Rules

CheMLFlow validates your config before running. These rules prevent common mistakes:

| Rule | What it means |
|------|---------------|
| Blocks require nodes | Each config block (except `global` and `pipeline`) needs a matching node in `pipeline.nodes`. For example, a `split:` block requires `split` in the nodes list. |
| No top-level `model:` | Model settings go under `train.model.*` or `train_tdc.model.*`, not at root level. |
| `use.curated_features` is configless | This node takes no configuration. Do not add a `use:` block. |
| `preprocess.keep_all_columns` moved | Use `curate.keep_all_columns` instead. |
| `preprocess.exclude_columns` moved | Use `train.features.exclude_columns` instead. |
| `train` requires model type | When `train` is in your pipeline, you must specify `train.model.type`. |
| `train.tdc` requires model type | When `train.tdc` is in your pipeline, you must specify `train_tdc.model.type`. |

## Top-Level Blocks

Your config file can contain these top-level blocks:

| Block | When allowed |
|-------|--------------|
| `global` | Always (required) |
| `pipeline` | Always (required) |
| `get_data` | When `get_data` node is in pipeline |
| `curate` | When `curate` node is in pipeline |
| `label` | When `label.normalize` node is in pipeline |
| `split` | When `split` node is in pipeline |
| `featurize` | When any `featurize.*` node is in pipeline |
| `preprocess` | When `preprocess.features` or `select.features` node is in pipeline |
| `train` | When `train` node is in pipeline |
| `train_tdc` | When `train.tdc` node is in pipeline |

## `global` Block

The `global` block defines settings that apply across the entire pipeline.

**Example:**
```yaml
global:
  pipeline_type: qm9
  task_type: regression
  base_dir: data/qm9
  target_column: gap
  random_state: 42
  thresholds:
    active: 1000
    inactive: 10000
```

**Required keys:**

| Key | Description |
|-----|-------------|
| `pipeline_type` | Name for this pipeline/dataset (e.g., `qm9`, `urease`, `pgp`). Used in logging and artifact paths. |
| `base_dir` | Directory for data artifacts (curated files, splits, features). |
| `thresholds.active` | Activity threshold for curation (compounds below this are "active"). |
| `thresholds.inactive` | Inactivity threshold for curation (compounds above this are "inactive"). |

**Optional keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `task_type` | `regression` | Either `regression` or `classification`. |
| `target_column` | `pIC50` | Column name containing the prediction target. |
| `random_state` | `42` | Root seed for reproducibility. Other seeds inherit from this unless overridden. |
| `log_level` | (from `debug`) | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `debug` | `false` | When `true`, enables verbose logging (equivalent to `log_level: DEBUG`). |
| `run_dir` | (computed) | Explicit output directory. Overrides all other output path logic. |
| `runs.enabled` | `false` | When `true`, outputs go to `runs/<timestamp>/` instead of `results/`. |
| `runs.id` | (timestamp) | Explicit run ID when `runs.enabled: true`. |

**Output directory resolution:**

1. If `global.run_dir` is set → use that path directly
2. Else if `global.runs.enabled: true` → use `runs/<runs.id or timestamp>/`
3. Else → use `results/`

## `pipeline` Block

The `pipeline.nodes` list defines which steps run and in what order.

**Example:**
```yaml
pipeline:
  nodes:
    - get_data
    - curate
    - split
    - featurize.morgan
    - train
```

**Supported nodes:**

| Category | Nodes |
|----------|-------|
| Data loading | `get_data` |
| Data preparation | `curate`, `use.curated_features`, `label.normalize`, `label.ic50` |
| Splitting | `split` |
| Featurization | `featurize.lipinski`, `featurize.rdkit`, `featurize.rdkit_labeled`, `featurize.morgan` |
| Preprocessing | `preprocess.features`, `select.features` |
| Training | `train`, `train.tdc` |
| Analysis | `analyze.stats`, `analyze.eda`, `explain` |

**Order constraints:**

- Only one `split` node allowed per pipeline.
- Only one `train.tdc` node allowed per pipeline.
- `train` and `train.tdc` cannot both be in the same pipeline.
- If using `train.tdc`, it must be the last node.
- `split` must come **after** any of: `curate`, `label.normalize`, `label.ic50`.
- `split` must come **before** any of: `preprocess.features`, `select.features`, `train`, `explain`.

## `get_data` Block

Specifies where to load raw data from.

**Examples:**

```yaml
# From ChEMBL API
get_data:
  data_source: chembl
  source:
    target_name: urease

# From local CSV file
get_data:
  data_source: local_csv
  source:
    path: local_data/my_dataset.csv

# From TDC
get_data:
  data_source: tdc
  source:
    group: ADME
    name: Pgp_Broccatelli
```

**Keys:**

| Key | Required | Description |
|-----|----------|-------------|
| `data_source` | Yes | Source type: `chembl`, `local_csv`, `http_csv`, or `tdc`. |
| `source.target_name` | For chembl | ChEMBL target name to query. |
| `source.path` | For local_csv | Path to CSV file (relative to repo root). |
| `source.url` | For http_csv | URL to fetch CSV from. |
| `source.group` | For tdc | TDC dataset group (default: `ADME`). |
| `source.name` | For tdc | TDC dataset name within the group. |
| `max_rows` | No | Row limit for sampling raw CSV rows. Set under `get_data`, applied during `curate` when `pipeline_type: qm9`. |

## `curate` Block

Controls how raw data is cleaned and standardized.

**Example:**
```yaml
curate:
  smiles_column: SMILES
  dedupe_strategy: keep_first
  keep_all_columns: true
```

**Keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `properties` | (auto) | Column(s) to extract. Defaults to `target_column` for QM9/classification, otherwise `standard_value`. |
| `smiles_column` | (auto-detect) | Name of the SMILES column in raw data. |
| `dedupe_strategy` | (none) | How to handle duplicate SMILES: `keep_first`/`first`, `keep_last`/`last`, `drop_conflicts`, or `majority`. |
| `label_column` | `target_column` | Label column for classification tasks. |
| `require_neutral_charge` | `false` | When `true`, removes molecules with net charge. |
| `prefer_largest_fragment` | `true` | When `true`, keeps only the largest fragment of multi-fragment molecules. |
| `keep_all_columns` | `false` | When `true`, preserves all source columns through curation. When `false`, only essential columns (SMILES, target) are kept. |

## `label` Block (for `label.normalize`)

Normalizes categorical labels to binary (0/1) for classification tasks.

**Example:**
```yaml
label:
  source_column: Activity
  target_column: label
  positive: active, Active, 1
  negative: inactive, Inactive, 0
  drop_unmapped: true
```

**Keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `source_column` | — | (Required) Column containing original labels. |
| `target_column` | (from global) | Column name for normalized labels. |
| `positive` | — | (Required) Values to map to `1`. Can be a list or comma-separated string. |
| `negative` | — | (Required) Values to map to `0`. Can be a list or comma-separated string. |
| `drop_unmapped` | `true` | When `true`, rows with labels not in positive/negative are dropped. |

## `split` Block

Defines how data is divided into train/validation/test sets. This is critical for reproducibility and fair model comparison.

**Simple example (holdout):**
```yaml
split:
  mode: holdout
  strategy: scaffold
  test_size: 0.2
  val_size: 0.1
  random_state: 42
```

**Cross-validation example:**
```yaml
split:
  mode: cv
  strategy: scaffold
  cv:
    n_splits: 5
    fold_index: 0
  val_from_train:
    val_size: 0.1
```

### Common Keys

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `holdout` | Split protocol: `holdout`, `cv`, or `nested_holdout_cv`. |
| `strategy` | `random` | How to assign rows: `random`, `scaffold`, or `tdc_*`. |
| `test_size` | `0.2` | Fraction of data for test set (holdout mode). |
| `val_size` | `0.1` | Fraction of data for validation set. |
| `random_state` | (from global) | Seed for splitting. Inherits from `global.random_state` if not set. |
| `stratify` | `false` | When `true`, preserves class distribution across splits. |
| `stratify_column` | `target_column` | Column to stratify by. Must be low-cardinality (≤20 unique values). |
| `require_disjoint` | `false` | When `true`, fails if any overlap between train/val/test. |
| `allow_missing_val` | `true` | When `true`, accepts TDC splits that lack a validation set. |

### Coverage Enforcement

These settings prevent "winning by dropping hard rows" — ensuring that model comparisons are fair.

| Key | Default | Description |
|-----|---------|-------------|
| `min_coverage` | (none) | Minimum fraction of curated rows that must be assigned to splits. |
| `require_full_test_coverage` | `false` | When `true`, fails if any test rows are lost during featurization. Equivalent to `min_test_coverage: 1.0`. |
| `min_test_coverage` | (none) | Minimum fraction of original test rows that must remain after cleaning. |
| `min_train_coverage` | (none) | Minimum fraction of original train rows that must remain. |
| `min_val_coverage` | (none) | Minimum fraction of original val rows that must remain. |

### `mode: holdout`

Standard single train/val/test split. Best for quick experiments.

**Outputs:**
- `data/<dataset>/splits/<split_file>.json` — reusable split definition
- `<run_dir>/split_indices.json` — row indices for this run
- `<run_dir>/split_meta.json` — metadata (sizes, coverage, seeds)

### `mode: cv`

K-fold cross-validation. Run the pipeline once per fold (varying `fold_index`) to get robust metrics.

**Additional keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `cv.n_splits` | `5` | Number of folds. |
| `cv.repeats` | `1` | Number of times to repeat the k-fold (with different random shuffles). |
| `cv.fold_index` | `0` | Which fold to use as test (0 to n_splits-1). |
| `cv.repeat_index` | `0` | Which repeat to use (0 to repeats-1). |
| `cv.random_state` | (from split) | Seed for fold assignment. |
| `val_from_train.val_size` | (from split) | Fraction of train_pool to hold out as validation. |
| `val_from_train.stratify` | (from split) | Whether to stratify the val split. |
| `val_from_train.random_state` | (computed) | Defaults to `cv.random_state + repeat_index × 1000 + fold_index`. |

**Behavior:**
1. Rows are divided into `n_splits` folds.
2. Fold at `fold_index` becomes the test set.
3. Remaining rows form `train_pool`.
4. Validation is sampled from `train_pool` using `val_from_train` settings.

### `mode: nested_holdout_cv`

Two-level splitting: an outer holdout test set (never touched during model selection) plus inner CV for hyperparameter tuning.

**Additional keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `stage` | `inner` | Which stage to run: `inner` (CV on dev set) or `outer` (final evaluation). |
| `outer.test_size` | (from split) | Fraction for outer test set. |
| `outer.random_state` | (from split) | Seed for outer split. |
| `inner.n_splits` | `5` | Number of inner CV folds. |
| `inner.repeats` | `1` | Number of inner CV repeats. |
| `inner.fold_index` | `0` | Which inner fold to run. |
| `inner.repeat_index` | `0` | Which inner repeat to run. |
| `inner.random_state` | (from split) | Seed for inner CV. |

**Behavior:**
1. Outer split creates `outer_test` (held out) and `dev` (for model selection).
2. `stage: inner` runs CV within `dev` — use this to tune hyperparameters.
3. `stage: outer` trains on all of `dev` and evaluates on `outer_test` — use this for final reporting.

## `featurize` Block

Controls molecular featurization (converting SMILES to numeric descriptors).

**Example:**
```yaml
featurize:
  radius: 3
  n_bits: 4096
```

**Keys (for `featurize.morgan`):**

| Key | Default | Description |
|-----|---------|-------------|
| `radius` | `2` | Morgan fingerprint radius (higher = more structural context). |
| `n_bits` | `2048` | Fingerprint bit vector length. |

> **Note:** `featurize.rdkit` and `featurize.lipinski` use fixed settings and do not accept additional config keys.

## `preprocess` Block

Controls feature preprocessing (variance filtering, correlation removal, scaling).

**Example:**
```yaml
preprocess:
  variance_threshold: 0.1
  corr_threshold: 0.9
  stable_features_k: 100
```

**Keys (for `preprocess.features` and `select.features`):**

| Key | Default | Description |
|-----|---------|-------------|
| `variance_threshold` | `0.16` | Remove features with variance below this threshold. |
| `corr_threshold` | `0.95` | Remove one of each pair of features with correlation above this. |
| `clip_range` | `[-1e10, 1e10]` | Clip feature values to this range before processing. |
| `stable_features_k` | `50` | Number of top features to keep in `select.features`. |
| `random_state` | (from global) | Seed for any randomized preprocessing steps. |

## `train` Block

Configures model training. This is where you select your model type and set hyperparameters.

**Example:**
```yaml
train:
  random_state: 123  # optional: override global seed for training only
  model:
    type: random_forest
    params:
      n_estimators: 500
      max_depth: 20
  tuning:
    method: fixed
  reporting:
    plot_split_performance: true
  features:
    exclude_columns:
      - mol_weight
```

**Sub-blocks:**

| Block | Required | Description |
|-------|----------|-------------|
| `train.model` | Yes | Model type and hyperparameters. |
| `train.tuning` | No | Hyperparameter search settings. |
| `train.reporting` | No | Output/plotting options. |
| `train.early_stopping` | No | Early stopping for DL models. |
| `train.features` | No | Feature selection/exclusion. |

**Direct keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `train.random_state` | (from global) | Seed for training. Place directly under `train:`, not under `train.model:`. |

### `train.model`

**Required key:** `train.model.type`

**Supported model types:**

| Task | Model types |
|------|-------------|
| Regression | `random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble` |
| Regression (DL) | `dl_simple`, `dl_deep`, `dl_gru`, `dl_resmlp`, `dl_tabtransformer`, `dl_aereg` |
| Classification | `catboost_classifier` |
| Classification (SMILES-native) | `chemprop` |

**General keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `train.model.params` | `{}` | Model-specific hyperparameters (passed to the underlying model). |
| `train.model.n_jobs` | `-1` | Parallelism for sklearn/joblib. Use `1` to disable parallelism. |

**Chemprop-specific keys:**

Chemprop is a graph neural network that works directly on SMILES (no descriptor generation needed).

| Key | Default | Description |
|-----|---------|-------------|
| `smiles_column` | `canonical_smiles` | Column containing SMILES strings. |
| `foundation` | `none` | Foundation model: `none` or `chemeleon`. |
| `foundation_checkpoint` | — | Path to checkpoint file (required when `foundation: chemeleon`). |
| `freeze_encoder` | `false` | When `true`, freezes the message-passing encoder during fine-tuning. |

**Chemprop hyperparameters** (under `train.model.params`):

| Key | Default | Description |
|-----|---------|-------------|
| `max_epochs` | `30` | Maximum training epochs. |
| `batch_size` | `64` | Batch size. |
| `max_lr` or `lr` | `1e-3` | Maximum learning rate. |
| `init_lr` | `max_lr/10` | Initial learning rate for warmup. |
| `final_lr` | `max_lr/10` | Final learning rate after decay. |
| `mp_hidden_dim` | `300` | Message-passing hidden dimension. |
| `mp_depth` | `3` | Message-passing depth (number of layers). |
| `ffn_hidden_dim` | `300` | Feed-forward network hidden dimension. |

> **Note:** Chemprop training sets global random state via Lightning's `seed_everything()` for reproducibility.

### `train.tuning`

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `fixed` | Tuning method: `fixed` (use provided params) or `train_cv` (sklearn RandomizedSearchCV). |
| `cv_folds` | `5` | Number of CV folds for `train_cv` method. |
| `search_iters` | `100` | Number of random search iterations for `train_cv`. |
| `use_hpo` | `false` | Enable Optuna hyperparameter optimization (DL models only). |
| `hpo_trials` | `30` | Number of Optuna trials when `use_hpo: true`. |

### `train.reporting`

| Key | Default | Description |
|-----|---------|-------------|
| `plot_split_performance` | `false` | When `true`, generates per-split metrics and plots (train/val/test comparison). |

### `train.early_stopping`

| Key | Default | Description |
|-----|---------|-------------|
| `patience` | `20` | Number of epochs without improvement before stopping (DL models only). |

### `train.features`

| Key | Default | Description |
|-----|---------|-------------|
| `exclude_columns` | `[]` | Columns to drop from the feature matrix before training. Can be a list or comma-separated string. |
| `categorical_features` | `[]` | Columns to treat as categorical (for one-hot encoding). Can be a list or comma-separated string. |

## `train_tdc` Block (`train.tdc` node)

Runs TDC (Therapeutics Data Commons) benchmark evaluation. This trains across multiple seeds and reports aggregated metrics for leaderboard comparison.

**Example:**
```yaml
train_tdc:
  benchmarks:
    - Pgp_Broccatelli
  seeds: [1, 2, 3, 4, 5]
  model:
    type: catboost_classifier
```

**Keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `group` | `ADMET_Group` | TDC benchmark group. Currently only `ADMET_Group` is supported. |
| `benchmarks` | — | List of benchmark names to evaluate (required). Can also use `benchmark` (singular). |
| `split_type` | `default` | TDC split type to use. |
| `seeds` | `[1,2,3,4,5]` | Random seeds for multi-seed evaluation. |
| `path` | (from global) | Directory for TDC data downloads. |
| `model.type` | — | Model type (required). Must be `catboost_classifier`. |
| `tuning.cv_folds` | `5` | CV folds for hyperparameter search. |
| `tuning.search_iters` | `100` | Random search iterations. |
| `tuning.use_hpo` | `false` | Enable Optuna HPO. |
| `tuning.hpo_trials` | `30` | Optuna trials if enabled. |
| `early_stopping.patience` | `20` | Early stopping patience. |
| `featurize.radius` | `2` | Morgan fingerprint radius. |
| `featurize.n_bits` | `2048` | Morgan fingerprint bit count. |

## Nodes Without Configuration

These nodes run with fixed behavior and do not accept config blocks:

| Node | What it does |
|------|--------------|
| `use.curated_features` | Uses the curated CSV directly as features (no featurization). |
| `label.ic50` | Converts IC50 values to pIC50 with activity classes. |
| `analyze.stats` | Runs statistical tests on the dataset. |
| `analyze.eda` | Generates exploratory data analysis plots. |
| `explain` | Generates feature importance and SHAP explanations. |

## Seed Inheritance

CheMLFlow uses a hierarchical seed system for reproducibility:

```
global.random_state (default: 42)
    │
    ├── split.random_state (inherits from global)
    │       │
    │       ├── split.cv.random_state (inherits from split)
    │       ├── split.outer.random_state (inherits from split)
    │       └── split.inner.random_state (inherits from split)
    │
    ├── train.random_state (inherits from global, can override)
    │
    └── preprocess.random_state (inherits from global, can override)
```

**Key points:**
- Set `global.random_state` once to control all randomness.
- Override specific seeds only when you need different behavior for specific stages.
- CV/nested fold seeds are deterministically derived from the base seed plus fold/repeat indices.

## Split Artifacts

Each run with a `split` node writes two files:

**`<run_dir>/split_indices.json`**
```json
{
  "train": [0, 1, 2, 5, 7, ...],
  "val": [3, 8, 12, ...],
  "test": [4, 6, 9, ...]
}
```
Row indices into the curated dataset for each split.

**`<run_dir>/split_meta.json`**
```json
{
  "mode": "cv",
  "strategy": "scaffold",
  "fold_index": 0,
  "repeat_index": 0,
  "random_state": 42,
  "sizes": {"train": 800, "val": 100, "test": 100},
  "coverage": {"assigned_fraction": 1.0},
  ...
}
```
Full metadata including mode, strategy, seeds, fold/repeat indices, sizes, and coverage statistics.

**Use these files to:**
- Debug unexpected split behavior
- Reproduce exact fold membership in future runs
- Aggregate metrics across CV folds

---

## Troubleshooting Validation Errors

| Error Code | Meaning | How to Fix |
|------------|---------|------------|
| `CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE` | Config block exists without a matching node in `pipeline.nodes`. | Add the node to the pipeline, or remove the config block. |
| `CFG_MISSING_TRAIN_MODEL` | `train` node is in the pipeline but `train.model` is missing. | Add a `train.model` section. |
| `CFG_MISSING_TRAIN_MODEL_TYPE` | `train.model` exists but `type` is not specified. | Set `train.model.type` to a valid model name. |
| `CFG_CONFIGLESS_NODE_HAS_BLOCK` | `use.curated_features` is in the pipeline and a `use:` block exists. | Remove the `use:` block (this node doesn't accept config). |
| `CFG_LEGACY_MODEL_BLOCK_FORBIDDEN` | A top-level `model:` block was found. | Move settings to `train.model.*` or `train_tdc.model.*`. |
| `CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN` | `preprocess.keep_all_columns` or `preprocess.exclude_columns` was found. | Use `curate.keep_all_columns` or `train.features.exclude_columns` instead. |
