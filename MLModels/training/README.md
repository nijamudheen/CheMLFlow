# MLModels Training Package

This package contains modular training utilities used by `MLModels/train_models.py`.

## Files

- `api.py`
  - Public API wrappers for training workflows.
  - Provides `DatasetSplit`, `TrainSpec`, `train`, `train_from_frames`, `load`, and `run_explainability`.
- `cli.py`
  - Command-line interface for standalone use (`train`, `predict`, `explain`) built on top of the API.
- `config.py`
  - Runtime config parsing and normalization helpers.
  - Includes `RuntimeTrainingOptions`, `as_bool`, `resolve_n_jobs`, and Chemprop foundation config parsing.
- `metrics.py`
  - Classification and regression metric helpers and validation.
  - Includes safe metric wrappers (AUC/AUPRC/R2/MAE) and split metric assembly helpers.
- `plots.py`
  - Plot and artifact writers for ROC/PR/confusion/split-metrics/parity plots.
- `persistence.py`
  - Model/metrics/params persistence helpers (`joblib`, JSON, torch state dict).
- `torch_models.py`
  - PyTorch DL helpers for device selection, deterministic seeding, training, prediction, and Optuna tuning.
- `sklearn_models.py`
  - Tabular model constructors and hyperparameter-search wrappers (RF/SVM/DT/XGBoost/ensemble).
- `dl_registry.py`
  - DL model registry for `DLSearchConfig` builders (`dl_simple`, `dl_deep`, `dl_gru`, etc.).
- `model_factory.py`
  - Unified model initialization dispatcher used by `train_models.py`.
- `orchestrator.py`
  - Core `train_model` orchestration (runtime options, fit/predict, metrics/artifact persistence).
- `model_loader.py`
  - Core `load_model` implementation for DL/ML/CatBoost reload paths.
- `train_helpers.py`
  - Shared utility helpers (label coercion, feature-name sanitization, split-index mapping, classification output shaping).
- `chemprop_models.py`
  - Chemprop/Chemeleon training implementation used by the compatibility facade.
- `explainability.py`
  - Permutation importance and SHAP explainability implementation.
- `__init__.py`
  - Package exports for the modules above.

## Usage

Current workflows still call through `MLModels.train_models`. You can also call the package API directly:

### Quickstart (CLI)

Use the bundled CLI for a Chemprop-style workflow:

```bash
python -m MLModels.training.cli train \
  --data-path MLModels/training/examples/regression.csv \
  --target-column target \
  --model-type random_forest \
  --task-type regression \
  --output-dir runs/cli_quickstart
```

```bash
python -m MLModels.training.cli predict \
  --test-path MLModels/training/examples/regression.csv \
  --target-column target \
  --model-path runs/cli_quickstart/random_forest_best_model.pkl \
  --model-type random_forest \
  --task-type regression \
  --preds-path runs/cli_quickstart/predictions.csv
```

Top-level CLI help:

```bash
python -m MLModels.training.cli --help
```

### Usage (Python API)

```python
from MLModels.training.api import DatasetSplit, TrainSpec, train

dataset = DatasetSplit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_val=X_val,
    y_val=y_val,
)

spec = TrainSpec(
    model_type="random_forest",
    output_dir="runs/example",
    task_type="regression",
)

model, result = train(dataset, spec)
```

## Compatibility

`MLModels/train_models.py` remains the compatibility surface for existing callers while logic is moved into this package.
