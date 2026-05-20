# Design of Experiments

Design of Experiments (DOE) is the CheMLFlow layer for defining a set of
scientific comparisons before you run them. A DOE file describes the dataset,
the evaluation policy, the model and featurization axes to compare, and the
output location. CheMLFlow expands that design into runtime configs, filters
known-invalid combinations, and records enough metadata to audit what was run.

Use DOE when you want to compare models, featurizers, split strategies, or CV
folds under one consistent policy. The point is not only to make many configs
quickly; it is to make the experiment easier to trust. A good DOE keeps train,
validation, and test policy consistent, prevents hand-edited config drift, keeps
artifacts separated, and preserves the reasons that cases were skipped.

## Minimal Workflow

Start from the example DOE file:

```text
config/doe.example.yaml
```

Set at least:

- `dataset.source.path`
- `dataset.target_column`
- `dataset.smiles_column` for local CSV datasets
- the `search_space` values you want to compare
- `output.dir`

For real experiments, use a fresh `output.dir` for each DOE run. If you are
rerunning the tutorial output and do not need the old generated files, clean it
first:

```bash
rm -rf config/generated/flash_doe
```

Generate the DOE configs:

```bash
conda activate chemlflow_env
python scripts/generate_doe.py --doe config/doe.example.yaml
```

Inspect the generated design:

```bash
cat config/generated/flash_doe/summary.json
head -n 20 config/generated/flash_doe/manifest.jsonl
head -n 20 config/generated/flash_doe/parent_manifest.jsonl
ls config/generated/flash_doe/*.yaml
```

Run one generated case:

```bash
CHEMLFLOW_CONFIG=config/generated/flash_doe/<case_file>.yaml python main.py
```

Use the `config_path` values in `manifest.jsonl` or `ls` output rather than
assuming a specific generated filename.

## DOE YAML Shape

Top-level keys:

- `version`: must be `1`
- `dataset`: fixed facts about the data
- `defaults`: fixed runtime settings applied to every case
- `search_space`: experiment axes to vary
- `constraints`: generation and artifact-safety settings
- `selection`: metric preference metadata
- `output`: generated artifact location

Typical local CSV DOE:

```yaml
version: 1

dataset:
  task_type: regression
  target_column: label
  smiles_column: SMILES
  source:
    type: local_csv
    path: local_data/my_data.csv

defaults:
  global.base_dir: data/my_doe
  global.random_state: 42
  global.runs.enabled: true
  split.mode: holdout
  split.test_size: 0.2
  split.val_size: 0.1
  train.tuning.method: fixed

search_space:
  train.model.type: [random_forest, xgboost]
  split.strategy: [random, scaffold]
  pipeline.feature_input: [featurize.rdkit, featurize.morgan]

constraints:
  isolate_case_artifacts: true

output:
  dir: config/generated/my_doe
```

`search_space` and `defaults` use dotted paths. Scalar values are treated as
single-value axes; lists expand into a grid.

## Generated Artifacts

DOE generation writes these files into `output.dir`:

- `summary.json`: counts, profile/task, selection metadata, DOE spec hash,
  snapshot path, and git provenance
- `manifest.jsonl`: one row per attempted execution child, including skipped
  cases and issue codes
- `parent_manifest.jsonl`: one row per scientific parent config
- `*.yaml`: one runnable runtime config per valid execution child
- `doe_spec.input.yaml`: the exact DOE spec snapshot used for generation

Generated runtime configs are case-isolated by default. CheMLFlow scopes
`global.base_dir` and `global.run_dir` under the DOE spec hash and `case_id`, and
sets `global.runs.id` to the case id. Leave
`constraints.isolate_case_artifacts: true` unless you intentionally want shared
artifacts.

## Parent And Child Cases

DOE separates the scientific comparison from execution slices:

- A parent case is the logical scientific config, such as one model,
  featurizer, and split strategy.
- A child case is one concrete execution, such as a specific CV fold and repeat.
- Holdout designs usually have one child per parent.
- CV and nested designs can have many children per parent.

For `split.mode: cv` or `nested_holdout_cv`, keep `split.cv.n_splits` and
`split.cv.repeats` in `defaults`. If fold and repeat indices are omitted, DOE
expands all folds and repeats automatically. Set fold or repeat indices only
when debugging or rerunning a specific execution slice.

## Supported Profiles

Supported profiles are:

- `reg_local_csv`
- `reg_local_csv_ic50`
- `reg_chembl_ic50`
- `clf_local_csv`
- `clf_tdc_benchmark`

If `dataset.profile` is omitted, CheMLFlow infers it from
`dataset.task_type` and `dataset.source.type`. For `task_type: auto`, set
`dataset.auto_confirmed: true`.

## Correctness Checks

DOE skips combinations that are known to be invalid before they reach runtime.
Common examples include:

- model/task mismatches
- split strategy and split mode mismatches
- missing validation split settings
- missing target or SMILES columns
- unsupported feature input for a model
- Chemprop or Chemeleon preprocessing combinations that are not meaningful
- duplicate rendered runtime configs
- runtime schema validation errors

Skipped child cases stay in `manifest.jsonl` with issue codes such as
`DOE_MODEL_TASK_MISMATCH`, `DOE_SMILES_COLUMN_MISSING`, or
`DOE_RUNTIME_SCHEMA_INVALID`. This is intentional: skipped cases are part of the
audit trail.

If `constraints.max_cases` is omitted and the expanded design exceeds the
default safety limit of 10,000 execution cases, DOE generation fails fast.

## Scientific Use

DOE is most useful when it protects the scientific comparison:

- Use the same split policy, random seed, target column, and metric policy across
  cases.
- Compare models on matched splits where possible.
- Prefer scaffold CV for chemistry generalization claims.
- Avoid picking a winner from many models on one fixed test split.
- Use repeated CV or nested holdout CV when the conclusion needs to be robust.
- Keep a final untouched holdout for final reporting, not iterative selection.

Selection metadata defaults to `auc` for classification and `r2` for regression.
Override with `selection.primary_metric` when the scientific question requires a
different primary metric.

## Practical Defaults

- `clf_local_csv` with non-Chemprop models defaults to
  `pipeline.feature_input: featurize.morgan`.
- `chemprop` and `chemeleon` default to `pipeline.feature_input: smiles_native`
  when that axis is omitted.
- `reg_chembl_ic50` defaults `global.target_column` to `pIC50`.
- For comparisons across pipelines, strongly consider
  `split.require_disjoint: true` and `split.require_full_test_coverage: true`.

## Best Practices

Keep one DOE file per split mode. Mixing `holdout`, `cv`, and
`nested_holdout_cv` in one cartesian grid makes some settings meaningless for
some rows.

Keep only true experiment axes in `search_space`; put fixed choices in
`defaults`. Large mixed grids grow quickly and are harder to interpret.

For chemistry model comparison, prefer `split.mode: cv` with
`split.strategy: scaffold`. Scaffold CV requires a usable SMILES column that can
produce `canonical_smiles` in curated data.

Separate hyperparameter search from evaluation design.
`train.tuning.method: train_cv` is inner tuning; `split.mode: cv` or
`nested_holdout_cv` is the outer evaluation design.

Run a small pilot before the full DOE. Confirm metrics files, split metadata,
runtime, and memory behavior before spending the full compute budget.

Preserve generated configs, manifests, `summary.json`, and the emitted
`git_worktree.patch` if generation happened from a dirty working tree. These
files make the experiment reproducible after the run has finished.
