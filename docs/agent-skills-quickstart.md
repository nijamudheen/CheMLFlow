# Agent Skills Quickstart

CheMLFlow skills are small operating manuals for agents. They teach an agent how to do
CheMLFlow-specific work consistently: review DOE files, audit analysis outputs, and avoid
common manifest, row-count, scaler, and split-balance mistakes.

## 1. What is included

```text
skills/
+-- chemlflow-doe-designer/
+-- chemlflow-analysis-curator/
```

Each skill has:

- `SKILL.md`: concise instructions and trigger description
- `references/`: details loaded only when needed
- `scripts/`: deterministic helper checks

## 2. Use a skill in a prompt

Ask your agent to use the skill by path:

```text
Use the CheMLFlow DOE Designer skill in skills/chemlflow-doe-designer to review config/doe_pgp.yaml.
```

```text
Use the CheMLFlow Analysis Curator skill in skills/chemlflow-analysis-curator to audit pah/pah_analysis_6689856.
```

## 3. Run the helper checks

Summarize generated DOE artifacts:

```bash
python skills/chemlflow-doe-designer/scripts/summarize_doe.py tmp/pgp_hpcc_analysis/pgp_doe
```

Audit analysis outputs:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py pah/pah_analysis_6689856
```

## 4. What the agent should check

For DOE work, the agent should inspect:

- `summary.json`
- `manifest.jsonl`
- `parent_manifest.jsonl`
- model, feature, scaler, and split compatibility
- valid, skipped, and parent case counts

For analysis work, the agent should inspect:

- `report.json`
- `all_runs_metrics.csv`
- `all_runs_metrics_by_execution.csv`
- raw vs aggregated row counts
- `scaler`, Morgan/RDKit, model, and split balance
- failed or incomplete folds before discussing model performance

## 5. Optional auto-discovery

Keep `skills/` in this repo as the source of truth. If your agent supports automatic skill
discovery, copy or symlink the skill folders into that agent's personal or project skill
directory.

Example:

```bash
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/chemlflow-doe-designer" ~/.codex/skills/chemlflow-doe-designer
ln -s "$(pwd)/skills/chemlflow-analysis-curator" ~/.codex/skills/chemlflow-analysis-curator
```

After installing or symlinking skills, restart the agent session so it can reload available
skills.
