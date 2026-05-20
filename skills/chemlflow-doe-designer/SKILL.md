---
name: chemlflow-doe-designer
description: Design and review CheMLFlow DOE YAMLs and generated DOE artifacts. Use when Codex is asked to create, modify, or audit CheMLFlow DOE specs, search spaces, model/feature/scaler/split compatibility, manifest skip reasons, parent/child CV shape, or expected valid/skipped case counts.
---

# CheMLFlow DOE Designer

## Overview

Use this skill as a small operating manual for CheMLFlow DOE work. Keep the focus on experiment validity: compatible axes, predictable manifest shape, auditable configs, and scientifically meaningful split/evaluation design.

## Workflow

1. Locate the DOE spec, usually `config/doe_*.yaml`, `doe/doe_*.yaml`, or a user-provided YAML.
2. Read `docs/doe.md` only if the repo behavior is unfamiliar or the DOE uses a less common profile.
3. Inspect `dataset`, `defaults`, `search_space`, `constraints`, `selection`, and `output`.
4. Check model/feature/scaler/split compatibility before recommending a run.
5. If generated artifacts exist, inspect `summary.json`, `manifest.jsonl`, and `parent_manifest.jsonl`.
6. Report expected run shape: total attempted children, valid children, skipped children, valid scientific parents, and major skip reasons.
7. Call out scientific risks separately from syntax risks.

## Checks

- Keep fixed choices in `defaults`; keep only true experiment axes in `search_space`.
- Treat DOE as parent/child shaped: one scientific parent can expand to many execution children, usually CV folds.
- For CV runs, expect all folds/repeats to be generated unless fold/repeat indices are intentionally fixed for debugging.
- Prefer separate DOE specs for holdout, CV, and nested holdout CV.
- Treat `smiles_native` as reserved for SMILES-native models such as `chemprop` and `chemeleon`.
- Expect tabular models to use `featurize.rdkit`, `featurize.morgan`, or curated numeric features, not raw SMILES.
- Expect `chemprop` and `chemeleon` to reject ordinary preprocessing/scaler branches except meaningful no-op branches.
- For comparison studies, check that Morgan/RDKit/scaler/split rows are balanced across non-native models.
- For final claims, prefer CV or nested holdout CV over selecting many configs on one fixed test split.

## Useful Commands

Summarize generated DOE artifacts:

```bash
python skills/chemlflow-doe-designer/scripts/summarize_doe.py <generated-doe-dir>
```

Generate DOE configs from a spec only when the user has asked for execution or validation that requires it:

```bash
python scripts/generate_doe.py --doe config/doe_example.yaml
```

## References

- For detailed review prompts and expected red flags, read `references/doe-review.md`.
- For canonical repo docs, prefer `docs/doe.md`, `docs/doe_quickstart.md`, and `docs/dataset_profile_support_matrix.md`.
