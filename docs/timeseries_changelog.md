# Changelog — CheMLFlow time-series patch

All changes here are scoped to the time-series Adaptive NVAR / Connectome NVAR
integration. Earlier baseline behavior of CheMLFlow is unchanged.

## v6 — 2026-05-15

Four targeted fixes prompted by the first real HPCC run: silent CPU fallback
on a GPU node, log lines that didn't distinguish Optuna trials from test-runs,
stale single-run numbers in metrics.json during a 25-run sweep, and a
continuous `lr_adam` axis that diverged from the notebook protocol.

### Fixed

- **Silent CPU fallback on the GPU node.** v1–v5 used `torch.device("cuda"
  if torch.cuda.is_available() else "cpu")`. When PyTorch was built without
  CUDA support (or Slurm failed to attach the GPU), training fell back to
  CPU and the only sign was a single "Training on device: cpu" line buried
  in a 35-minute Optuna log. v6 adds an explicit `train.model.params.device`
  knob with three values:
    - `cuda` (strict): raise `RuntimeError` if CUDA isn't actually available.
      This is the recommended HPCC setting — it converts a silent multi-hour
      CPU run into a fast, loud failure. Set in `config/doe_timeseries.yaml`
      and `config/doe_timeseries_connectome.yaml` defaults.
    - `auto`: pre-v6 behavior — CUDA if available, else CPU, no raise.
    - `cpu`: force CPU even on a GPU node (useful for debugging).
  Invalid values (e.g. `gpu`) are rejected at parse time with a clear error.

- **GPU diagnostic at startup.** A one-shot block now logs PyTorch version,
  `torch.version.cuda`, `torch.cuda.is_available()`, `torch.cuda.device_count()`,
  `CUDA_VISIBLE_DEVICES`, requested device, and resolved device. Fires once
  per process, before training, so a misconfigured environment is obvious
  from the first 10 lines of the log instead of being a 35-minute mystery.

- **Repeated-final UX (the "25 test runs" confusion).** The v3 patch already
  retrains 25 times after Optuna picks hyperparameters, but the user
  experience made it hard to tell what was happening:
    - Each iteration logged the same `"Training on device: cpu"` line as
      Optuna trials, with no visible boundary between "we're still searching"
      and "we're now doing final test-runs".
    - The metrics.json on disk carried the *single-run* number until the
      25th iteration finished and the post-loop rewrite landed. If the job
      was killed mid-sweep, the metrics.json was stale and misleading.
    - No way to see results stream in as runs completed.

  v6 fixes all three:
    - A header line `Final test protocol: N independent retrain+evaluate
      runs ...` separates Optuna from final-test.
    - Each iteration logs `[final test K/N] seed=... output=...`,
      `[final test K/N] done in 4.8s; rmse_h100=0.20 (running mean=0.20,
      std=0.001)` — running mean and std after every iteration.
    - A new per-case file `<model>_test_runs_progress.csv` appends a row
      after each iteration with run_index, seed, wall_seconds, and all
      horizon RMSEs. Tail-friendly.
    - `metrics.json` is incrementally rewritten after each iteration with
      the partial aggregate stats plus `partial: true`. The final rewrite
      sets `partial: false` and adds `test_num_runs_target`. If the job
      is killed at run 7 of 25, the metrics.json still has a meaningful
      mean±std over the completed 7 runs and clearly marks itself partial.

- **Optuna search space for `dl_adaptive_nvar` is now notebook-faithful.**
  `lr_adam` was a continuous log-uniform float `(1e-4, 1e-2)`. The notebook
  uses `suggest_categorical("lr_adam", [1e-4, 1e-3, 1e-2])`. v6 matches.
  `dl_connectome_nvar` was already all-categorical; verified to match
  the connectome notebook (`k`, `n_connectome`, `input_scaling`, `lr_adam`,
  `lr_lbfgs`, `weight_decay` — exact lists).

### Added

- `MLModels.training.timeseries_nvar._resolve_device(...)` and
  `_log_device_diagnostics(...)`. Single shared device-selection path
  used by both Optuna trials and final test-runs.
- `TrainingConfig.device: str = "auto"`.
- New tests:
    - `test_device_strict_cuda_raises_when_unavailable`
    - `test_device_auto_falls_back_to_cpu_silently`
    - `test_device_param_rejects_typos`
    - `test_dl_registry_adaptive_nvar_lr_adam_is_categorical`
    - `test_repeated_final_writes_progress_csv_and_partial_flag`

### Notes on test_num_runs semantics

We considered changing the 25-run protocol to "train once, evaluate 25
times" (which would have been faster). Decided against: the user's notebooks
retrain 25 times with different seeds, which is what reports `mean ± std`
of *training stochasticity*. v6 preserves this and only fixes the UX/
persistence around it. If you ever want the cheaper "single train, repeated
evaluation" protocol, that's a separate feature (with different scientific
meaning) — happy to add it under a different knob.

### Verified

- End-to-end run against `data/ground_truth.npy`: device diagnostic prints
  cleanly, `[final test 1/3] … [final test 3/3]` boundaries are obvious,
  running mean/std updates after every iteration, `partial: true` flips
  to `false` after the last run, progress CSV has 1 header + 3 data rows.
- 11/11 new time-series tests pass.
- 173/173 non-rdkit existing tests pass — 179 total.

---

## v3 — 2026-05-13

Architectural correction. Earlier versions confused two distinct concerns
(scientific experiment design vs hyperparameter tuning) by baking the
Optuna grid into the DOE. v3 separates them properly:

  * **DOE** = scientifically distinct experiments (noise levels,
    connectome_mode, selection_mode, …).
  * **Optuna** = hyperparameter search *within* each DOE case
    (k, hidden_dim, lr_adam, lr_lbfgs, n_connectome, input_scaling, …).
    Each case runs its own Optuna study in-loop.

### Added

- `MLModels.training.dl_registry.build_timeseries_dl_search_config(...)`.
  Separate registry entries for `dl_adaptive_nvar` and `dl_connectome_nvar`,
  each declaring its own `search_space` and `default_params`. The two
  architectures intentionally do *not* share axes — Adaptive NVAR has
  `hidden_dim`, Connectome NVAR has `n_connectome` and `input_scaling`,
  etc. The search-space spec format (`{"type": "categorical|float|int", ...}`)
  is identical to the tabular DL path, so it reads the same way.
- `MLModels.training.timeseries_nvar._run_optuna_timeseries(...)`. Wraps
  Optuna's `TPESampler` study around the existing trainer. Each trial:
  sample → merge with registry defaults + user YAML → build → train
  Adam→L-BFGS → score on val rollout RMSE at the largest horizon.
  Returns the best trial's params, which the outer trainer then merges
  into the final retrain with the full epoch budget.
- New tuning controls in YAML:
    - `train.tuning.method`: `"fixed"` (default) or `"optuna"`.
    - `train.tuning.n_trials`: trial count when method=optuna (default 30).
    - `train.tuning.trial_epoch_cap`: per-trial Adam+L-BFGS epoch cap, so
      sweeps finish quickly; the final retrain uses the full epoch budget.
    - `train.tuning.timeout`: wall-clock timeout (seconds), passed through
      to `study.optimize`.
    - `train.tuning.verbose`: surface Optuna's INFO chatter.
- `optuna>=3.5` added to optional deps in `requirements.txt`. The base
  install does not pull it in; `tuning.method: optuna` raises a clear
  ImportError if the package is missing.
- `metrics.json` now carries a `"tuning"` block when method=optuna:
  `method`, `n_trials_requested`, `n_trials_completed`, `best_trial_number`,
  `best_value`, `best_params`, `search_axes`, `trial_epoch_cap`. Reflects
  exactly what was searched and chosen.
- Two new tests:
    - `test_dl_registry_timeseries_search_configs` verifies that
      `dl_adaptive_nvar` and `dl_connectome_nvar` have distinct, well-typed
      search spaces (and that the wrong model_type fails clearly).
    - `test_train_timeseries_optuna_runs_and_writes_artifacts` runs 3
      trials end-to-end on a synthetic series and asserts the tuning
      summary lands in `metrics.json` and is consistent with the final
      retrained config.

### Changed

- `config/doe_timeseries.yaml` shrunk from 252 cases to **6**. It now
  varies only scientific factors (`dataset_noise_scale`,
  `train_noise_scale`); the architecture/optimization axes from
  notebook 2 (k, hidden_dim, lr_adam, lr_lbfgs) moved to dl_registry.
- `config/doe_timeseries_connectome.yaml` similarly shrunk to **24** cases
  covering `connectome_mode × connectome_selection_mode × dataset_noise_scale
  × train_noise_scale`. The notebook-1 axes (k, n_connectome, input_scaling,
  lr_adam, lr_lbfgs, weight_decay) moved to dl_registry.
- The "unsupported model" error in `dl_registry.build_dl_search_config` for
  `dl_adaptive_nvar` / `dl_connectome_nvar` now also points users to
  `build_timeseries_dl_search_config` in the same module.

### Verified

- End-to-end run of a generated DOE case (`case_0001`) against the user's
  `ground_truth.npy`: Optuna explored 3 trials (axes: k, hidden_dim,
  lr_adam, lr_lbfgs), picked the winner, retrained, and wrote a
  `metrics.json` whose `tuning` block matched the chosen `config`.
- 6/6 new time-series tests pass. 173/173 non-rdkit existing tests pass.

---

## v2 — 2026-05-09

Two bug fixes and supporting documentation, prompted by user feedback after
the first running attempt of v1.

### Fixed

- **`openpyxl` missing produced a cryptic pandas stack trace.**
  `dl_connectome_nvar` requires `openpyxl` (pandas' xlsx engine). When it was
  not installed, the failure surfaced four frames deep inside pandas
  (`pandas.compat._optional.import_optional_dependency`). The connectome
  loader now imports `openpyxl` up front in `load_connectome_xlsx` and raises
  a one-line `ImportError` with the exact install command:

  > Reading the connectome workbook requires the optional `openpyxl` package,
  > which is not installed in this environment. Install with `pip install
  > openpyxl` (or `conda install -c conda-forge openpyxl`) and re-run. This
  > is only needed for `dl_connectome_nvar`; the rest of CheMLFlow does not
  > depend on it.

  Files: `utilities/connectome_loader.py`.

- **DOE generation marked every `ts_forecast` case as
  `DOE_MODEL_NOT_SUPPORTED_FOR_PROFILE`.**
  `ProfileSpec.allows_model` had a special case for `dl_*` model names: it
  *only* accepted them when the profile listed the wildcard `"dl_*"` in
  `allowed_models`. The `ts_forecast` profile lists exact names
  (`"dl_adaptive_nvar"`, `"dl_connectome_nvar"`) because we deliberately do
  not want every tabular DL model to slip into a time-series sweep — and the
  exact-name path was unreachable for `dl_*`-prefixed names. Fixed so
  `allows_model` accepts a `dl_*` name that is either listed by exact name
  *or* covered by the `"dl_*"` wildcard. Verified end-to-end:

  - `generate_doe(...)` on `config/doe_timeseries.yaml` → **252 valid, 0
    skipped, 0 issues** (was 0/252 in v1).
  - One generated case ran to completion against the user's `ground_truth.npy`
    and wrote all expected artifacts.

  Files: `utilities/doe.py`.

### Added

- `requirements.txt` gained an "Optional: dl_connectome_nvar" comment block
  documenting `openpyxl` and `networkx` as commented-out lines, so users
  who hit the new ImportError can also enable both deps in one shot.

### Notes & verification

- 4/4 new time-series tests pass.
- 173/173 non-rdkit existing tests pass — the `allows_model` change is
  non-regressive (other profiles still accept any `dl_*` model via the
  wildcard, and the new exact-name match only widens behavior; nothing
  previously valid is rejected).
- The cosmetic `pipeline_type: mackey_glass` in generated cases (inherited
  from the `dataset.name` field) is benign: the `train.timeseries` node does
  not gate on `pipeline_type`. Left as-is for v2.

---

## v1 — 2026-05-09 (initial release)

Full integration of Adaptive NVAR and Adaptive Connectome NVAR as a
time-series-native pipeline branch in CheMLFlow ("Path B" from the design
discussion). Adds:

- `pipeline_type: timeseries` and the new terminal node
  `train.timeseries`, mutually exclusive with `train` and `train.tdc`.
- Two new model types: `dl_adaptive_nvar`, `dl_connectome_nvar`.
- Two new data sources: `local_npy`, `local_ts_csv`.
- New DOE profile `ts_forecast` with corresponding case-config emission.
- Adam → L-BFGS trainer with windowed autoregressive rollout and
  multi-horizon RMSE; rich per-window-per-horizon CSV.
- Sample runtime configs (`config/timeseries_mg_demo.yaml`,
  `config/timeseries_quick_demo.yaml`) and DOE specs
  (`config/doe_timeseries.yaml`,
  `config/doe_timeseries_connectome.yaml`).
- 4 new tests (`tests/test_timeseries_nvar.py`).
- Prose documentation: `docs/timeseries_pipeline.md`.

See `docs/timeseries_pipeline.md` for the full schema and quickstart.
