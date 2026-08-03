# Agent-Guided DOE Demo Acceptance Rehearsal — 2026-08-02

## Verdict

The core workflow works end to end. A fresh agent profiled the PGP Broccatelli
dataset, asked the scientific questionnaire, generated a DOE without launching
compute, opened a DOE-level queued dashboard, honored a pilot-only approval,
ran Chemprop, recovered from a native crash, and audited the partial result. A
separate presentation-sized rehearsal then completed all nine execution
children and produced a final, audited three-parent leaderboard.

The result is a **conditional pass**, not yet a clean unattended-demo pass.
Three product issues should be fixed before treating the canonical prompt as a
polished live demo:

1. a dashboard launched inside the fresh Codex turn exited when that turn
   ended, requiring the supervising session to reattach it;
2. rerunning a failed child replaced its earlier execution-attempt record, so
   retry provenance was lost; and
3. two dashboard values disagree with analysis/provenance semantics: fold
   spread uses sample standard deviation while `analysis.py` emits population
   standard deviation, and a resumed successful pilot displays the near-zero
   duration of the `already_successful` skip rather than its original runtime.

The full 65-execution scientific study was intentionally not launched. Only
its Chemprop fold-0 pilot ran. The complete nine-execution presentation DOE is
a workflow rehearsal on a 25% sample and is not a final scientific result.

## Test boundary

- Candidate repository: isolated worktree at commit `315afa8`, with the current
  uncommitted dashboard, progress, study-runner, and demo changes overlaid.
- Fresh agent thread: `019fc2f2-3f61-7b51-86a6-b8c86a547a3e`.
- Runtime: Python 3.12.13, RDKit 2026.03.4, scikit-learn 1.9.0,
  Chemprop 2.2.4, Torch 2.2.2, and Lightning 2.6.5.
- Hardware: CPU-only macOS; four Torch threads before the local workaround;
  CUDA and MPS unavailable.
- Main worktree: no training or generated configs were written into it. The
  run occurred in `/private/tmp/chemlflow-agent-demo.Yr3Y2k`.
- Test dependency: `pytest 9.1.1` was added to `chemlflow_env` during this
  rehearsal because the prepared environment did not contain pytest.

## Scenario A: canonical natural-language agent flow

### Intake and questionnaire

The controller sent only:

```text
I want to build a model for the PGP Broccatelli dataset.
```

The fresh agent profiled the data before making changes. It identified 1,275
rows, `SMILES`, binary `Activity`, the supplied Training/Internal
Validation/External Validation groups, likely leakage columns, invalid SMILES,
and cross-partition duplicates. It then asked six material questions covering
endpoint meaning, curation, supplied partitions, scaffold versus random CV,
model/representation scope, and the primary metric. No DOE or run artifact
existed and no training began in this turn.

The controller replied:

```text
Accept the recommendations.
```

### Generated scientific study

The agent generated a Training-only, five-fold scaffold-CV study with AUROC as
the primary metric. Internal and External Validation remained untouched. The
valid scientific design contained 13 parents:

- random forest, SVM, CatBoost, and `dl_simple`, each crossed with Morgan,
  RDKit, and ECFP4+RDKit inputs; and
- one SMILES-native Chemprop parent.

CheMeleon was omitted because no checkpoint was present. DOE generation
produced 65 valid children and 135 compatibility-skipped children from 200
attempted Cartesian records. Every valid parent had five children, every child
used `Set: Training`, and every child had a unique config and run directory.

Before approval, the dashboard reported 65 queued, 0 running, 0 completed, and
135 compatibility skips. There was no execution manifest, run status, metric,
or analysis output. This proves the pre-approval compute boundary.

### Pilot-only approval and execution

The controller deliberately narrowed approval:

```text
Start only the Chemprop fold 0 pilot (case_0196). Do not start the remaining
64 executions unless I explicitly approve them after reviewing the pilot.
```

The agent created a one-record manifest, verified it contained only
`case_0196`, and ran it serially. The live dashboard correctly changed from
65 queued to 1 running plus 64 queued and exposed the five-node pipeline,
Chemprop epoch telemetry, and split integrity.

The first attempt reached all 30 training epochs but segfaulted while restoring
checkpoints for prediction (`return_code=-11`). No second case started. The
single protocol-permitted retry used `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`; it succeeded in 89.2 seconds.

The successful scaffold fold used 734 curated Training rows:

| Split | Rows | AUC | AUPRC | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Train | 528 | 0.8613 | 0.8616 | 0.7784 | 0.7944 |
| Validation | 59 | 0.8000 | 0.7400 | 0.8136 | 0.8254 |
| Test | 147 | 0.7624 | 0.6715 | 0.6599 | 0.5902 |

The partial audit exited successfully with no integrity issues for the one
completed fold. It correctly reported `ranking_ready: false` and
`final_claim_ready: false`. The dashboard showed Chemprop at 1/5, left the
leaderboard empty, and kept the other 64 children queued.

## Scenario B: complete presentation rehearsal

The second lane exercised final settlement using
`doe/pgp_dashboard_demo.yaml`. This DOE uses a seeded 25% stratified sample,
three random-CV folds, and three compatible parents: random forest + Morgan,
`dl_simple` + Morgan, and Chemprop + native SMILES.

Generation produced the expected shape:

- 18 attempted execution records;
- 9 valid children across 3 valid parents; and
- 9 compatibility-skipped children across 3 invalid parents.

The queued dashboard showed 0/9 and no leaderboard. A thread-limited Chemprop
fold-0 pilot then completed and passed a partial audit, which correctly kept
ranking disabled. The runner resumed the remaining children serially. Its
final execution manifest contained eight `COMPLETED` attempts and one
`SKIPPED/already_successful` record that reused the pilot's successful
`run_status.json`; all nine run-status artifacts were successful.

Final analysis and audit results:

- 9 completed execution rows;
- 3 completed aggregate parent rows;
- `slice_count=3`, `completed_slices=3`, and `failed_slices=0` for every
  parent;
- no failed cases, missing metrics, missing split metrics, mapping mismatches,
  or non-finite AUROC values;
- `ranking_ready: true`; and
- `final_claim_ready: true`.

The dashboard's aggregate means exactly matched `all_runs_metrics.csv`:

| Rank | Presentation candidate | Mean AUC | Analysis AUC SD |
|---:|---|---:|---:|
| 1 | Random forest + Morgan | 0.9258 | 0.0076 |
| 2 | `dl_simple` + Morgan | 0.8711 | 0.0131 |
| 3 | Chemprop + native SMILES | 0.6837 | 0.0919 |

These values describe only the compact presentation sample. They are not the
answer to the full scientific PGP study.

## Acceptance matrix

| Acceptance criterion | Result | Evidence |
|---|---|---|
| Fresh natural-language intake | Pass | Dataset profile and six-question response |
| No compute before approval | Pass | 65 queued, no execution/run artifacts |
| DOE-level dashboard denominator | Pass | 13 parent rows and 65 child rows |
| SMILES-native pilot | Pass after one documented workaround | Chemprop fold 0 completed after thread-capped retry |
| Queued/running/settled dashboard truth | Pass with duration caveat | API snapshots and screenshots at all three states |
| Compatibility skips separate from failures | Pass | 0 run failures; skip-record counts remain separate |
| Incomplete parent excluded from ranking | Pass | Scientific pilot stayed 1/5 with empty leaderboard |
| Final aggregate leaderboard | Pass | Presentation DOE settled 9/9 and three parents ranked |
| Final analysis audit | Pass | `ranking_ready` and `final_claim_ready` both true |
| Retry-attempt provenance | Fail | First SIGSEGV attempt was overwritten on retry |
| Dashboard survives agent turn boundary | Fail in fresh CLI harness | Supervisor had to reattach read-only server |
| Complete reproducible transcript bundle | Partial | Final messages saved; full CLI JSONL trace was not persisted |
| Full 65-child scientific result | Not run | Requires separate explicit compute approval |

## Findings to fix

### P0 — Preserve every local execution attempt

`scripts/run_doe_local.py` rewrites `execution_manifest.jsonl` on each
invocation. The successful retry therefore removed the first `-11` attempt and
its failure record. A retry-safe manifest should preserve both attempts and
identify the latest/current attempt explicitly. The per-case log was also
reused, which makes the first crash difficult to audit after recovery.

### P0 — Keep the dashboard alive across agent turns

The attach server worked and its API was correct while the fresh Codex turn was
active, but the process exited with the turn. The supervising process could
reattach immediately, proving this was lifecycle ownership rather than a
dashboard-state failure. The product flow needs a persistent launcher or
supervisor-owned process before handing the URL to the user.

### P1 — Align dashboard spread with analysis

`chemlflow_dashboard/static/app.js` divides variance by `n - 1`, so the top
tile displayed random forest as approximately `0.9258 ± 0.0093`. Final
`all_runs_metrics.csv` reports `auc_std=0.007629`, which is population standard
deviation over the three folds. The UI and analysis must use the same defined
statistic and label it clearly.

### P1 — Reuse the original runtime for `already_successful`

After `--resume`, the reused Chemprop pilot appeared completed but with a
runtime of roughly zero seconds. The dashboard used the near-instantaneous
`SKIPPED/already_successful` attempt timestamps rather than the successful
`run_status.json` duration. This also understates the Chemprop wall-time card.

### P1 — Reduce agent latency and context use

The three fresh-agent turns reported approximately 17.9 million input tokens
and 91,678 output tokens in total. Much of the input was cached, but the agent
still reread broad source ranges and skill files repeatedly. The functional
behavior was correct; the interaction is too slow and expensive for a live
demo. A compact repository orientation artifact, narrower commands, and a
purpose-built acceptance controller would materially improve it.

### P2 — Make environment activation deterministic

The default `python` resolved through a broken pyenv selection, and
`conda run -n chemlflow_env python` also encountered that shim. The agent
recovered by using the environment's absolute interpreter path. The demo
launcher should activate the environment or resolve its interpreter once,
rather than requiring discovery during the conversation.

### P2 — Persist the full controller transcript

The evidence bundle contains each turn's final message and all artifact/browser
checkpoints, but not the full Codex JSONL stream. The next automated controller
should save stdout for every turn so commands, intermediate reasoning messages,
and tool outcomes survive independently of the supervising conversation.

## Verification performed

```text
Focused pytest suite: 8 passed in 7.19s
Browser console errors: 0 at queued, running, and settled checkpoints
Presentation final audit: exit 0, issues=[], ranking_ready=true,
  final_claim_ready=true
Scientific pilot audit: exit 0 with --allow-partial, issues=[],
  ranking_ready=false, final_claim_ready=false
git diff --check: passed in the isolated candidate worktree
```

The focused suite covered DOE generation, dashboard state, dashboard HTTP
serving, local execution-manifest behavior, and run-progress telemetry. This
was not a run of the repository's entire test suite.

## Evidence

Fresh-agent checkpoints:

- [Queued scientific dashboard](../output/agent-demo-acceptance-20260802/agent-flow/queued-dashboard.png)
- [Running scientific pilot](../output/agent-demo-acceptance-20260802/agent-flow/running-dashboard.png)
- [Completed scientific pilot](../output/agent-demo-acceptance-20260802/agent-flow/pilot-completed-dashboard.png)
- [Questionnaire response](../output/agent-demo-acceptance-20260802/agent-flow/turn1_last_message.md)
- [Pre-approval study contract](../output/agent-demo-acceptance-20260802/agent-flow/turn2_last_message.md)
- [Pilot report](../output/agent-demo-acceptance-20260802/agent-flow/turn3_last_message.md)
- [Scientific pilot analysis](../output/agent-demo-acceptance-20260802/scientific-pilot/analysis/report.json)
- [Scientific pilot audit summary](../output/agent-demo-acceptance-20260802/scientific-pilot/audit-summary.json)

Complete presentation checkpoints:

- [Queued presentation dashboard](../output/agent-demo-acceptance-20260802/presentation/queued-dashboard.png)
- [Running presentation pilot](../output/agent-demo-acceptance-20260802/presentation/running-dashboard.png)
- [Settled presentation dashboard](../output/agent-demo-acceptance-20260802/presentation/settled-dashboard.png)
- [Settled API snapshot](../output/agent-demo-acceptance-20260802/presentation/settled-snapshot.json)
- [Final analysis report](../output/agent-demo-acceptance-20260802/presentation/analysis_final/report.json)
- [Final audit summary](../output/agent-demo-acceptance-20260802/presentation/audit-summary.json)
- [Aggregate metrics](../output/agent-demo-acceptance-20260802/presentation/analysis_final/all_runs_metrics.csv)
- [Execution-level metrics](../output/agent-demo-acceptance-20260802/presentation/analysis_final/all_runs_metrics_by_execution.csv)
- [Final execution manifest](../output/agent-demo-acceptance-20260802/presentation/execution_manifest.jsonl)
- [Verification summary](../output/agent-demo-acceptance-20260802/verification-summary.json)

The original isolated worktree remains available at
`/private/tmp/chemlflow-agent-demo.Yr3Y2k` for deeper inspection.
