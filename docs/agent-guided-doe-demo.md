# Agent-Guided DOE Demo and Acceptance Test

The 2026-08-02 isolated rehearsal is recorded in
[Agent-guided DOE demo acceptance rehearsal](agent-guided-doe-demo-acceptance-20260802.md).

## Goal

The full CheMLFlow demo begins with a researcher's natural-language request and
ends with an audited, DOE-level model comparison. The researcher should not
need to write YAML, calculate the execution fanout, launch individual folds, or
interpret raw run directories.

This is an agent-guided workflow. The questionnaire is conversational rather
than a separate web form, while the dashboard is a read-only view of the real
CheMLFlow artifacts.

## Canonical opening prompt

```text
I want to build a model for the PGP Broccatelli dataset.
```

The user may explicitly request a DOE, but should not need to know that term for
the study runner to recommend one when model comparison or cross-validation is
appropriate.

## Full demo flow

1. The user provides the natural-language request.
2. The agent locates and profiles the named dataset. It reports the likely
   target, task type, molecular input column, row count, class balance, missing
   values, duplicates or conflicts, supplied validation groups, and likely
   leakage columns.
3. The agent asks a concise study-design questionnaire. It covers at least:
   endpoint meaning, curation policy, intended generalization boundary, model
   and molecular-representation scope, validation depth, and selection metric.
   Each question includes a recommendation and tradeoff.
4. The user answers the questions or says `Accept the recommendations`.
5. The agent presents one study contract: dataset and endpoint, curation,
   split/validation protocol, model and representation space, primary metric,
   expected scientific parents, execution children, compatibility skips,
   assumptions, and limitations.
6. The agent generates the DOE without starting training.
7. The agent starts a persistent, attach-only dashboard. The dashboard opens
   at the DOE level and shows the valid execution children as queued,
   compatibility skips separately, and an empty/provisional leaderboard.
8. The agent reports the exact run shape and asks one explicit approval
   question, such as: `This will run three model candidates across three folds,
   producing nine valid executions. Start the pilot and, if it passes, the full
   study?`
9. The user says `Start`, `Run it`, or another clear affirmative response.
10. The agent runs a pilot execution while the same dashboard updates from
    run artifacts. If SMILES-native models are included, the pilot must exercise
    one of those models rather than proving only the tabular path.
11. The agent analyzes and audits the pilot. If the approved pilot gate passes,
    it resumes the remaining DOE children. A material change to the study stops
    the flow and requires renewed approval.
12. The dashboard displays DOE-level progress: completed/active/failed cases,
    model-by-fold coverage, real training scope where available, compatibility
    skips, aggregate fold metrics, and a provisional leaderboard.
13. After all required folds settle, the agent runs final analysis and audits
    `report.json`, execution-level metrics, aggregate metrics, failed cases, and
    fold completeness.
14. The dashboard marks the leaderboard final only when every required fold for
    a ranked parent succeeded and emitted the selected metric.
15. The agent reports the winning model using aggregate CV results, including
    fold spread, failures, skips, limitations, and links or paths to the study
    artifacts.

## Presentation-sized PGP study

The tracked presentation DOE is `doe/pgp_dashboard_demo.yaml`. It uses a seeded
25% stratified sample of `tutorials/data/pgp_broccatelli.csv` and compares:

- random forest with Morgan fingerprints;
- `dl_simple` with Morgan fingerprints;
- Chemprop with native SMILES input.

Each compatible scientific parent has three random-CV folds. The manifest
therefore contains three valid parents and nine valid execution children. The
three invalid model/input parents produce nine compatibility-skipped children;
these are design-audit records, not training failures.

The presentation DOE is a workflow demonstration, not a claim that a 25%
sample and three folds constitute the final scientific study.

The optional foundation-model extension is tracked separately at
`doe/pgp_tabpfn_foundation_demo.yaml`. It contributes two valid parents and six
children (TabPFN 2.6 over RDKit2D and CheMeleonFP) with training-fold-only
standardization and `[-6, 6]` clipping. Keeping it separate avoids changing the
accepted nine-job baseline demo and makes its heavier, gated dependency explicit.

## What can be tested automatically

Most of the demo can be tested without a human driving it. The testing strategy
has three layers.

### 1. Deterministic repository tests

These tests do not invoke an agent and should run on every relevant change:

- DOE generation produces the expected parent, execution, and skip counts.
- Every valid execution child has an isolated config and artifact directory.
- The local runner preserves every attempt in `execution_manifest.jsonl`,
  updates only the active attempt from `RUNNING` to terminal, and keeps a
  distinct log for each retry.
- A persistent dashboard remains healthy after the launching agent command
  exits and can be stopped only through matching lifecycle metadata.
- Dashboard snapshots classify queued, running, stale, failed, completed, and
  compatibility-skipped cases correctly.
- Dashboard API values agree with `run_status.json`, `run_progress.json`, model
  metrics, split metadata, and DOE manifests.
- A leaderboard row appears only after every required fold succeeds.
- Single-config views use run/fold language while DOE views use study/aggregate
  language.

The repository already has focused coverage in
`tests/test_pgp_dashboard_demo.py`, `tests/test_dashboard_state.py`,
`tests/test_dashboard_server.py`, `tests/test_local_doe_live_manifest.py`, and
`tests/test_run_progress.py`.

### 2. Scripted agent acceptance test

Run the agent under test in a fresh worktree with a controller acting as the
user. The controller feeds a fixed transcript and records every response,
command, process, file change, and browser URL.

1. Send the canonical opening prompt.
2. Assert that the agent profiles the dataset and asks the required scientific
   questions before creating or launching a study.
3. Reply `Accept the recommendations`.
4. Assert that the proposed study contract contains the endpoint, curation,
   split, models, representations, metric, run counts, skips, and limitations.
5. Assert that the agent generates the DOE and opens an attach-only dashboard.
6. Before approval, assert that no training process has started, no execution
   attempt is marked `RUNNING`, and the dashboard reports queued valid cases.
7. Reply `Start`.
8. Assert that a pilot runs first, produces current-attempt status and metrics,
   and passes analysis/audit before the remaining work starts.
9. Wait for the approved DOE to settle and assert that all expected execution
   children are accounted for as completed, failed, or intentionally skipped.
10. Assert that the final answer uses aggregate parent metrics and agrees with
    the audited analysis artifacts.

Conversational assertions should be rubric-based, because exact wording may
vary. Launch boundaries, process state, manifest counts, metric values, and
artifact paths should use deterministic assertions.

### 3. Real presentation rehearsal

Run the exact PGP demo in the prepared `chemlflow_env` and capture evidence at
four checkpoints:

1. questionnaire and study contract;
2. queued DOE dashboard before approval;
3. active pilot or full execution;
4. settled dashboard and final agent report.

At each dashboard checkpoint, save both a screenshot and the corresponding
`/api/v1/snapshot` response. The final evidence bundle should also contain the
agent transcript, DOE input/spec snapshot, manifests, launcher log, run status
and progress artifacts, analysis output, and audit result. Screenshots show the
experience; artifacts prove that the display was truthful.

## Suggested automation cadence

1. **Pull request:** Run deterministic generation, state-normalization, API,
   and tiny-run tests. Avoid the full Chemprop demo on every code change.
2. **Nightly or pre-release:** Run a small real DOE through generation,
   dashboard attachment, execution, analysis, and audit.
3. **Demo rehearsal:** Run the exact nine-execution PGP presentation DOE in a
   fresh output directory and capture the complete evidence bundle.
4. **Agent-contract changes:** Run the scripted conversation through the
   pre-approval boundary at minimum; run the full execution when launch or
   monitoring behavior changed.

## Acceptance criteria

The full demo passes only when all of the following are true:

- the agent asks the material scientific questions rather than inventing
  unspoken choices;
- no compute starts before explicit approval;
- the dashboard is DOE-level for a DOE and displays the planned denominator;
- a pilot proves every required runtime family, including a SMILES-native path
  when selected;
- the dashboard agrees with source artifacts at queued, running, and settled
  checkpoints;
- compatibility skips are not reported as failures;
- rankings remain provisional until all required folds are complete;
- the final model claim uses audited aggregate metrics rather than a single
  fold; and
- the transcript and evidence bundle are sufficient for another person to
  reproduce or audit the run.

## Operator commands

Generate and preview without launching training:

```bash
python scripts/generate_doe.py --doe doe/pgp_dashboard_demo.yaml
python -m chemlflow_dashboard start \
  --doe-dir config/generated/pgp_dashboard_demo
```

After explicit approval, run a serial pilot and then resume the study while the
attached dashboard remains open. Do not use the generic `--limit 1` as the only
pilot for this DOE: valid manifest order starts with random forest, so that
would not prove the Chemprop runtime. Select one valid Chemprop child into a
separate pilot manifest first:

```bash
jq -c \
  'select(.status == "valid" and .factors["train.model.type"] == "chemprop")' \
  config/generated/pgp_dashboard_demo/manifest.jsonl \
  | sed -n '1p' \
  > config/generated/pgp_dashboard_demo/chemprop_pilot_manifest.jsonl

python scripts/run_doe_local.py \
  --doe-dir config/generated/pgp_dashboard_demo \
  --manifest config/generated/pgp_dashboard_demo/chemprop_pilot_manifest.jsonl \
  --stop-on-failure

python analysis.py \
  --backend local \
  --doe-dir config/generated/pgp_dashboard_demo \
  --output-dir config/generated/pgp_dashboard_demo/analysis_pilot

python skills/chemlflow-analysis-curator/scripts/audit_analysis.py \
  config/generated/pgp_dashboard_demo/analysis_pilot

python scripts/run_doe_local.py \
  --doe-dir config/generated/pgp_dashboard_demo \
  --max-workers 1 \
  --resume
```

For a rehearsed demo that has already been approved and does not require the
queued preview gate, the combined launcher is:

```bash
python -m chemlflow_dashboard run \
  --doe doe/pgp_dashboard_demo.yaml \
  --max-workers 1
```

See [Live run dashboard](run-dashboard.md) for dashboard behavior and
[Design of Experiments](doe.md) for DOE artifact semantics.
