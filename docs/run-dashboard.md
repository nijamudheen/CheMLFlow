# Live run dashboard

CheMLFlow includes a local, read-only dashboard that derives every displayed
value from run artifacts. It supports a single runtime config and generated DOE
studies through the same normalized case model.

## Launch a run

Single config:

```bash
python -m chemlflow_dashboard run --config config/config.chembl.yaml
```

DOE:

```bash
python -m chemlflow_dashboard run \
  --doe doe/pgp_dashboard_demo.yaml \
  --max-workers 1
```

The DOE command performs a model dependency preflight, generates the DOE, starts
the local runner, and leaves the dashboard open for inspection when execution
finishes. Use `--no-open` on a headless machine. The dashboard binds to
`127.0.0.1` and a free port by default.

Run these commands from the prepared CheMLFlow Python environment. Attach mode
requires PyYAML; run mode additionally requires the libraries selected by the
config or DOE.

## Attach without launching work

```bash
python -m chemlflow_dashboard attach --config path/to/runtime.yaml
python -m chemlflow_dashboard attach --doe-dir config/generated/pgp_dashboard_demo
```

Attach mode does not modify or restart a study.

## Keep an attached dashboard running

Use persistent mode when an agent, script, or short-lived terminal command must
hand the dashboard URL back and exit:

```bash
python -m chemlflow_dashboard start --config path/to/runtime.yaml
python -m chemlflow_dashboard start \
  --doe-dir config/generated/pgp_dashboard_demo
```

The command waits for a healthy server, prints its URL, and then returns. The
server has an instance ID and source-specific metadata outside the study
directory. Repeating `start` reuses the same healthy server.

Inspect or stop that exact server with the same source argument:

```bash
python -m chemlflow_dashboard status \
  --doe-dir config/generated/pgp_dashboard_demo
python -m chemlflow_dashboard stop \
  --doe-dir config/generated/pgp_dashboard_demo
```

`stop` validates the server PID, instance ID, and source path before signaling
it. Foreground `attach` remains available for interactive terminal use.

## Truthfulness contract

- The latest per-case `execution_manifest.jsonl` attempt defines local process state;
  current-attempt `run_status.json` is the primary scientific terminal truth,
  with current `run_progress.json` as a best-effort fallback.
- `run_progress.json` is written atomically and heartbeats while a process is
  alive. It records pipeline nodes and any training scopes exposed by an adapter.
- PyTorch, Chemprop/Lightning, Optuna, and Adaptive NVAR expose real epoch or
  trial counts.
- Opaque estimators such as most sklearn fits show an indeterminate active state
  and heartbeat. CheMLFlow never invents a percentage.
- `execution_manifest.jsonl` preserves every local attempt. A new attempt is
  added as `RUNNING` before its subprocess starts, and only that attempt's row
  is atomically updated to its terminal state. Each launched subprocess has its
  own log under `local_logs/<case_id>/`.
- Launch-time process errors finalize the active attempt as `FAILED`; an
  interruption finalizes it as `CANCELLED` before propagating the interrupt.
- Local analysis accepts a terminal `run_status.json` for a `RUNNING` attempt
  only when its start timestamp belongs to that attempt. Older success
  artifacts cannot complete a live retry.
- Compatibility skips are displayed separately and excluded from the valid-run
  progress denominator. A skipped record can contain multiple issue codes, so
  issue-code counts are occurrences and must not be summed as case counts.
- Leaderboard rows require every valid fold in a parent to succeed and produce
  the selected metric. Rankings remain labeled provisional until the study is
  settled.
- Fold spread is population standard deviation across the completed required
  folds, matching `analysis.py` (`ddof=0`).
- CPU and GPU utilization display as unavailable in v1 because CheMLFlow does
  not yet emit trustworthy resource samples.

## PGP Broccatelli demo

`doe/pgp_dashboard_demo.yaml` uses the tracked
`tutorials/data/pgp_broccatelli.csv` dataset. It takes a seeded 25% stratified
sample and compares random forest, `dl_simple`, and Chemprop under the same
three-fold random CV protocol. The three compatible parent configurations yield
nine valid executions. Incompatible model/input cartesian combinations remain
visible as validation skips.

Chemprop and Lightning must be installed for the complete demo. The dashboard
launcher reports a failed preflight rather than silently omitting that model.

## PGP TabPFN foundation-model extension

`doe/pgp_tabpfn_foundation_demo.yaml` is a separate six-job study so the
established dashboard demo remains unchanged. It compares two scientific
parents across the same three-fold CV protocol:

- TabPFN 2.6 + RDKit2D;
- TabPFN 2.6 + frozen CheMeleonFP.

Both branches fit standardization on the training fold, preserve all 2,048
CheMeleon dimensions, and clip transformed values to `[-6, 6]`. Launch it on a
TabPFN-compatible host with:

```bash
python -m chemlflow_dashboard run \
  --doe doe/pgp_tabpfn_foundation_demo.yaml \
  --max-workers 1
```

The launch preflight checks PyTorch, Chemprop, TabPFN, the 2.6 API, and the
configured CheMeleon checkpoint before generating or starting jobs. It does not
review TabPFN's model license or authenticate on the user's behalf. Complete the
one-time TabPFN cache-priming step in the
[installation guide](installation.md#full-foundation-model-installation-in-the-same-environment)
before launching the dashboard, especially on a headless execution host.
