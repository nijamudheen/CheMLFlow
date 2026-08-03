---
name: chemlflow-run-monitor
description: Launch or attach to CheMLFlow's local read-only live dashboard for a single runtime config or generated DOE, verify that displayed progress and metrics match run artifacts, and diagnose queued, running, stale, failed, completed, or compatibility-skipped cases. Use when a user asks to run CheMLFlow with a dashboard, monitor a job in real time, inspect live epochs/trials/logs, attach to an existing local study, or validate dashboard truthfulness across sklearn, PyTorch, Chemprop/Lightning, Optuna, or Adaptive NVAR runs.
---

# CheMLFlow Run Monitor

## Purpose

Operate the artifact-backed local dashboard without creating a second source of
truth. Treat one runtime config as a one-case study and a DOE as its valid
execution children plus separate compatibility skips.

## Choose the operation

1. Attach when a run or DOE already exists. Do not restart active work.
2. For an approval-gated study, generate the config or DOE, attach read-only,
   verify the planned cases are queued, and wait for explicit approval before
   launching training.
3. Launch a single config when the user approved exactly one resolved runtime case.
4. Launch a DOE for approved comparisons, CV folds, or several model families.
5. Use the normal CheMLFlow Slurm workflow for remote execution. Dashboard v1 is
   a local backend; do not fabricate scheduler state.

Run commands from the CheMLFlow repository root:

```bash
python -m chemlflow_dashboard run --config <runtime.yaml>
python -m chemlflow_dashboard run --doe <doe.yaml> --max-workers 1
python -m chemlflow_dashboard attach --config <runtime.yaml>
python -m chemlflow_dashboard attach --doe-dir <generated-doe-dir>
python -m chemlflow_dashboard start --doe-dir <generated-doe-dir>
```

Use persistent `start` when the launching agent or command will exit before the
user is finished with the dashboard. Check or stop it later with `status` and
`stop` using the same `--config` or `--doe-dir` argument. Use foreground
`attach` when a human-owned terminal will remain open.

The combined `run` commands generate and execute immediately. Use them only
after approval. To preview a DOE first, run `scripts/generate_doe.py --doe
<doe.yaml>`, launch persistent monitoring with `start --doe-dir
<generated-doe-dir>`, report the queued run shape, and launch
`scripts/run_doe_local.py` only after the user says to proceed.

Use the project's prepared Python environment. At minimum, attach mode requires
PyYAML; verify it with `python -c "import yaml"` before launching. Run mode also
requires every dependency used by the selected cases.

Use `--no-open` in headless environments. Keep `--max-workers 1` for a simple
demo or when Chemprop stability has not been proven. Never combine parallel
workers with `--stop-on-failure`.

## Verify before reporting

1. Read the dashboard URL printed by the launcher.
2. Inspect `/api/v1/snapshot` or the UI execution ledger.
3. Cross-check a selected case against:
   - `<run_dir>/run_status.json` for current-attempt terminal truth;
   - `<run_dir>/run_progress.json` for heartbeat, pipeline node, and exposed
     training scopes;
   - `<generated-doe-dir>/execution_manifest.jsonl` for local attempt state;
   - the model metrics JSON and split metrics for displayed scores;
   - `manifest.jsonl` and `parent_manifest.jsonl` for valid/skipped cases and
     fold membership.
4. Derive the progress denominator from valid execution-child records in
   `manifest.jsonl`; use `summary.json` as a consistency check, not as an
   override.
5. Rank only parents whose valid folds all succeeded and emitted the selected
   metric. Call the rank provisional until the study is settled.

## Interpret telemetry faithfully

- A known epoch, trial, phase, or repeated-run total may display a percentage.
- An opaque sklearn or native fit must remain indeterminate while its heartbeat
  is fresh. Never infer progress from elapsed time.
- Mark a run stale only when the dashboard's heartbeat threshold is exceeded;
  stale is not the same as failed.
- Treat DOE compatibility skips as design audit records, not run failures.
- A skipped record may contain several issue codes. Report both skipped-record
  count and issue-code occurrences; never sum code counts as the number skipped.
- A generated DOE with no execution manifest or run artifacts has valid cases
  queued, zero settled cases, and an empty leaderboard.
- Resolve conflicting lifecycle artifacts in this order: an explicit local
  attempt failure wins; otherwise use only `run_status.json` and
  `run_progress.json` whose start times belong to the latest attempt. A current
  success artifact completes the case, while a completed process without one
  is failed. `SKIPPED` with `already_successful` intentionally reuses the prior
  success. Current `run_progress.state` is a best-effort terminal fallback, not
  a reason to reuse stale telemetry.
- Apply the same current-attempt timestamp rule to `analysis.py --backend
  local`: a live `RUNNING` retry must not inherit a prior success artifact.
- Leave CPU and GPU utilization unavailable unless a future artifact explicitly
  records trustworthy samples.
- Use the case drawer for the resolved config, artifacts, split metadata, metrics,
  and bounded log tail. The dashboard is read-only: do not claim it paused,
  pruned, retried, approved, or killed work.

## PGP example workflow

For an already approved tracked PGP Broccatelli study:

```bash
python -m chemlflow_dashboard run \
  --doe doe/pgp_dashboard_demo.yaml \
  --max-workers 1
```

Expect the launcher to preflight PyTorch, Chemprop, and Lightning. If preflight
fails, report the missing module and stop; do not silently present a reduced
three-model comparison. The generated study should contain nine valid execution
children: three compatible model/input parents across three CV folds.

## Completion gate

Before calling monitoring complete, report the URL or attach command, current
case counts, any stale/failed cases, compatibility-skip count, and whether the
study leaderboard is provisional or final. An empty leaderboard is still
provisional when settled valid cases are fewer than total valid cases. If
numbers disagree, trust the artifacts, capture the conflicting paths and
values, and diagnose the normalizer rather than editing scientific outputs.
