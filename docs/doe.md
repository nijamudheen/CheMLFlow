# DOE Config Generation

CheMLFlow supports generating many **runtime-valid** configs from one DOE YAML file.
The generator expands your search space, filters invalid combinations, and writes:

- `manifest.jsonl` (one row per attempted case, including skipped reasons)
- `summary.json` (counts, profile/task, selection metadata)
- one config YAML per valid case

Script:

```bash
python scripts/generate_doe.py --doe config/doe.example.yaml
```

## Why this exists

- Keeps split/evaluation policy consistent across models.
- Prevents known invalid combinations from reaching cluster jobs.
- Produces auditable case metadata (`status`, `issues`, `config_hash`).

## Supported profiles

- `reg_local_csv`
- `reg_chembl_ic50`
- `clf_local_csv`
- `clf_tdc_benchmark`

If `dataset.profile` is omitted, the generator infers a profile from:

- `dataset.task_type`
- `dataset.source.type`

For `task_type: auto`, set `dataset.auto_confirmed: true`.

## DOE schema (v1)

Top-level keys:

- `version` (must be `1`)
- `dataset`
- `search_space`
- `defaults` (optional)
- `constraints` (optional)
- `selection` (optional)
- `output`

Important conventions:

- `search_space` keys use dotted paths (for example `split.mode`, `train.model.type`).
- Values can be a scalar or list; scalar is treated as a single-value axis.
- `defaults` uses the same dotted-path style and is applied before `search_space` factors.
- For `split.mode: cv` or `nested_holdout_cv`, include fold axes in `search_space`
  (`split.cv.fold_index` / `split.cv.repeat_index` or nested equivalents) when you want multi-fold evaluation.
- By default, generated cases are isolated per case id:
  - `global.base_dir` becomes `<base_dir>/<case_id>`
  - `global.run_dir` becomes `<run_dir>/<case_id>` (or `<output.dir>/runs/<case_id>`)
  - `global.runs.id` is set to `case_id`
  Set `constraints.isolate_case_artifacts: false` only if you intentionally want shared artifacts.

## Required dataset fields (typical local CSV)

```yaml
dataset:
  task_type: classification   # or regression
  target_column: label
  source:
    type: local_csv
    path: local_data/my_data.csv
```

For classification with non-binary raw labels, provide a label map:

```yaml
dataset:
  label_source_column: Activity
  label_map:
    positive: [active, "1", 1]
    negative: [inactive, "0", 0]
```

## Compatibility checks (skipped with reason codes)

Examples:

- `DOE_MODEL_TASK_MISMATCH`
- `DOE_SPLIT_STRATEGY_MODE_INVALID`
- `DOE_VALIDATION_SPLIT_REQUIRED`
- `DOE_SELECT_REQUIRES_PREPROCESS`
- `DOE_FEATURE_INPUT_REQUIRED`
- `DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS`
- `DOE_CHEMPROP_PREPROCESS_UNSUPPORTED`
- `DOE_SPLIT_PARAM_INVALID`
- `DOE_DATASET_COLUMN_MISSING`
- `DOE_RUNTIME_SCHEMA_INVALID`

`manifest.jsonl` contains these codes per skipped case.

## Selection metric defaults

- Classification default: `auc`
- Regression default: `r2`

You can override in `selection.primary_metric`.

## Scientific selection guidance

- Avoid selecting the "best" config by comparing many configs on one fixed test split.
- Prefer `split.mode: nested_holdout_cv` (or repeated CV) and aggregate metrics across folds/repeats.
- Use the final untouched holdout only once for final reporting.
