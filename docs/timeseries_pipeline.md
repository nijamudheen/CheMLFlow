# CheMLFlow Time-Series Pipeline (Adaptive NVAR / Connectome NVAR)

This branch adds a **time-series forecasting pipeline** to CheMLFlow,
parallel to the existing tabular and SMILES-native branches. It targets
chaotic dynamical systems (Mackey–Glass, etc.) using the user's
**Adaptive NVAR** and **Adaptive Connectome NVAR** architectures.

## What's new

| Layer | What you get |
|-------|--------------|
| `pipeline_type: timeseries` | New profile that bypasses curate / featurize / split / preprocess / explain |
| `train.timeseries` node | New terminal node, mutually exclusive with `train` and `train.tdc` |
| Models | `dl_adaptive_nvar`, `dl_connectome_nvar` |
| Data sources | `local_npy`, `local_ts_csv` |
| DOE profile | `ts_forecast` (regression, source `local_npy`, models above) |
| Trainer | Two-phase Adam → L-BFGS, windowed autoregressive rollout, multi-horizon RMSE |
| Metrics | `<model>_metrics.json` + `<model>_split_metrics.json` + per-window-per-horizon CSV |

Architectures are ported faithfully from the user's notebooks
(`Optuna_MG_Adaptive_NVAR_0percent_noise.ipynb`,
 `Adaptive_NVAR_optuna_connectome.ipynb`); training, splitting, and rollout
are isolated into pipeline-friendly modules so the same trainer can serve
future autoregressive models.

## Quickstart

### 1. Drop your time series in place

The series is a 1-D or 2-D `.npy` (or CSV). For the user-supplied
Mackey–Glass series:

```bash
mkdir -p data
cp /path/to/ground_truth.npy data/ground_truth.npy
```

### 2. Single fixed-config run

```bash
CHEMLFLOW_CONFIG=config/timeseries_quick_demo.yaml python main.py
```

This trains an `AdaptiveNVAR` for ~3 seconds on CPU and writes:

```
artifacts/runs/ts_quick_demo/
  dl_adaptive_nvar_metrics.json                       # primary RMSE@max_horizon + diagnostics
  dl_adaptive_nvar_split_metrics.json                 # train/val/test horizon RMSEs
  dl_adaptive_nvar_rollout_per_window_per_horizon.csv # rich table
  dl_adaptive_nvar_predictions.npz                    # per-window pred/true/noisy
  dl_adaptive_nvar_best_model.pth                     # torch state_dict
  dl_adaptive_nvar_best_params.pkl                    # hyperparameters
  run_config.yaml, run_status.json, run.log
```

For the full notebook-quality config (5000 Adam + 50000 L-BFGS epochs):

```bash
CHEMLFLOW_CONFIG=config/timeseries_mg_demo.yaml python main.py
```

### 3. Connectome NVAR

```bash
mkdir -p data
cp /path/to/connectome_hermaphrodite.xlsx data/connectome_hermaphrodite.xlsx
```

Then craft a runtime config like `timeseries_mg_demo.yaml` but set:

```yaml
train:
  model:
    type: dl_connectome_nvar
    params:
      k: 5
      n_connectome: 100
      connectome_xlsx: data/connectome_hermaphrodite.xlsx
      connectome_sheet: hermaphrodite chemical
      connectome_mode: connectome              # or connectome_randomized
      connectome_selection_mode: top_degree    # or random
      connectome_normalization: maxabs         # or none, spectral
      input_scaling: 0.10
      lr_adam: 1.0e-3
      lr_lbfgs: 1.0
      horizons: [25, 50, 75, 100]
      num_windows: 10
```

### 4. DOE sweeps

Two DOE specs ship with this branch:

```bash
# AdaptiveNVAR sweep (mirrors notebook 2's Optuna grid)
python scripts/generate_doe.py --doe config/doe_timeseries.yaml

# ConnectomeNVAR sweep (mirrors notebook 1's Optuna grid + connectome_mode axis)
python scripts/generate_doe.py --doe config/doe_timeseries_connectome.yaml
```

Then dispatch the generated `case_*.yaml` configs through `main.py` however
your environment runs them.

## YAML schema for `pipeline_type: timeseries`

```yaml
global:
  pipeline_type: timeseries
  task_type: regression
  base_dir: ...
  run_dir: ...
  target_column: x        # cosmetic; not consumed by the trainer
  random_state: 2025
  thresholds: { active: 1000, inactive: 10000 }   # ignored, schema-required

pipeline:
  nodes: [get_data, train.timeseries]
  feature_input: none

get_data:
  data_source: local_npy           # or local_ts_csv
  source:
    path: data/ground_truth.npy
    time_axis: cols                # rows | cols | auto

split:                              # time-series segment lengths
  warmup_len: 500                   # context provided to rollout, never trained on
  train_len: 7500
  val_len: 1000
  test_len: 1000

train:
  tuning:
    method: fixed                   # or `optuna` to run an in-loop search
    n_trials: 30                    # only used when method=optuna
    trial_epoch_cap: 500            # cap Adam+L-BFGS epochs per trial
  model:
    type: dl_adaptive_nvar          # or dl_connectome_nvar
    params:
      k: 5
      hidden_dim: 200               # AdaptiveNVAR only
      n_connectome: 100             # ConnectomeNVAR only
      connectome_xlsx: ...          # ConnectomeNVAR only
      lr_adam: 1.0e-3
      lr_lbfgs: 1.0
      horizons: [25, 50, 75, 100]
      num_windows: 10
      train_noise_scale: 0.05
      dataset_noise_scale: 0.0
```

When `tuning.method == "optuna"`, the searched axes (k, hidden_dim,
lr_adam, lr_lbfgs for AdaptiveNVAR; k, n_connectome, input_scaling,
lr_adam, lr_lbfgs, weight_decay for ConnectomeNVAR) come from
`MLModels.training.dl_registry.build_timeseries_dl_search_config(...)`,
not from the YAML. Any value the user sets in `train.model.params` for a
searched axis is overridden by Optuna's choice. To change what's searched,
edit `dl_registry.py`; to vary scientific factors across experiments, use
the DOE.

The full parameter list is documented in
`MLModels/training/timeseries_nvar.py::TrainingConfig`.

## Running the test suite

```bash
pytest tests/test_timeseries_nvar.py -v
```

4 tests cover end-to-end training artifacts, model_type validation, split
config parsing, and time-series slicing. They run in <5 s on CPU.

## Architecture notes

* **No tabular preprocessing.** The pipeline validator forbids `curate`,
  `split`, `preprocess.features`, `select.features`, `featurize.*`,
  `label.*`, and `explain` when `train.timeseries` is in `pipeline.nodes`.
  Mixing them would only hide bugs.
* **`split.*` lengths, not `split.test_size`.** The `split:` block here
  carries `warmup_len`, `train_len`, `val_len`, `test_len` (integers, in
  timesteps). It is unrelated to the molecule-index `split` *node*.
* **Rollout is always windowed and autoregressive.** RMSE is computed per
  `(window, horizon)` pair, then averaged across windows for the
  aggregate metric. The number of windows is configurable via
  `train.model.params.num_windows`.
* **Training noise is applied only to the train segment.** A separate
  `dataset_noise_scale` adds noise globally before splitting; useful for
  measurement-noise robustness studies.
* **Connectome adjacency is loaded at runtime, not shipped.** The path
  goes in `train.model.params.connectome_xlsx`. The repo never ships
  third-party data.

## Compatibility

* `analysis.py` discovers the new metrics files automatically (it looks
  for `<run_dir>/<model_type>_metrics.json` and a `split_metrics_path`
  inside it — both produced by the timeseries trainer).
* All existing CheMLFlow pipelines continue to work unchanged. The patch
  was validated against the existing test suite: 173/173 non-rdkit tests
  pass after these changes (rdkit-dependent tests skip in environments
  without the optional rdkit install — same as before this patch).
