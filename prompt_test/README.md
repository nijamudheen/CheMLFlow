# CheMLFlow agent-forecasting experiment: evidence package

This package holds the records of 30 AI-agent studies run on 9 November 2026. Each study
asked an agent to build a univariate forecasting model for the time series
`dataset2/viT.npy`. The design compares agents working **with** CheMLFlow (`CheM_*`)
against agents working **without** it (`NOCheM_*`), at three prompt levels
(easy, medium, technical), with 5 repetitions each: 2 × 3 × 5 = 30 runs.
The exact prompts are in `Prompts_2nd_round.docx`.

Runs with a `_2` suffix (CheM_easy_rep2_2, CheM_easy_rep3_2, NOCheM_medium_rep5_2)
replace earlier attempts and fill repetitions 2, 3 and 5 of their groups.
`conclusions.html` describes how this was handled.

## Contents

| File | What it contains |
|---|---|
| `conclusions.html` | Summary report: results, token usage, DOE comparison, model choice and RMSE, interactions, errors, conclusions and limitations |
| `chats.zip` | Full logs of all 30 agent sessions. Each session has a readable HTML page and a machine-readable JSONL file; `index.html` lists them and `manifest.json` maps them to their sessions |
| `CheMLFlow_evaluation.zip` | Per-run evaluation on the hidden test block for the 15 CheMLFlow runs: `results.csv`, `baselines.csv` (persistence and ridge-AR), forecast plots, and saved predictions (`.npz`/`.npy`) |
| `No_CheMFLow_evaluation.zip` | The same files for the 15 runs without CheMLFlow |
| `Prompts_2nd_round.docx` | The exact prompts for each condition and level |
| `REDACTION_LOG.csv` | Record of every change made for privacy (see below) |

## How to open the report with working links

1. Create a folder (e.g. `Evidence/`). Put `chats.zip` in it and unzip it there,
   so you get `Evidence/chats/index.html`.
2. Create a subfolder (e.g. `Evidence/analysis/`) and put `conclusions.html` in it
   <, together with the analysis data files>.
3. Open `analysis/conclusions.html` in a web browser.

The report needs `chats/` to be one level above it. Don't rename `chats/` or move
`conclusions.html`, or the links will break. The evaluation zips can be unzipped
anywhere; the report doesn't link to them.

Note: each run folder in the evaluation zips contains a `runs/<case>` entry. It is a
shortcut (symbolic link) to the original project folder on the author's computer,
so it won't open on other machines. This is expected.

## Privacy redactions

Before release, these changes were made and nothing else was altered:

- The local computer username was replaced by `REDACTED_USER` in file paths,
  file listings and shortcut targets (9,610 occurrences).
- A cluster account ID was replaced by `REDACTED_HPC_ID` (16 occurrences, in
  CheMLFlow source code quoted in four chat logs).
- The author fields were removed from the hidden document properties of
  `Prompts_2nd_round.docx`. The visible text is unchanged.
- macOS system files (`.DS_Store`, `__MACOSX/`) were deleted.

All other content is byte-for-byte identical to the original records, including
every number, CSV, prediction array and plot. Each redacted file was checked to
confirm that reversing the replacement gives back exactly the original bytes.
`REDACTION_LOG.csv` lists every file with the action taken, the number of
replacements, and SHA-256 checksums before and after redaction.
The author list in the citation of the Adaptive NVAR paper (arXiv:2507.08738),
which appears in some logs, was left unchanged because it is public.

## Known gaps

The analysis data files linked from `conclusions.html` (CSV/JSON tables such as
`case_summary.csv` and `tokens_by_turn.csv`) are not included in this release.
Those links will not open. The report itself contains the key tables.

The development series `viT.npy` and the hidden test block `hiT.npz` used for
scoring can be generated with make_thomas_dataset.py with noise 10%.

