# Tutorial 01: Single Config in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/01_single_config_colab/CheMLFlow_Tutorial_01_Single_Config.ipynb)

This is the easiest real CheMLFlow tutorial.

It runs one PGP classification config end to end with:

- local CSV input
- Morgan fingerprints
- standard scaling
- SVM classifier
- 5-fold CV design, running fold `0` of repeat `0`

## Why this tutorial is only one fold

CheMLFlow's CV runtime executes one fold at a time. That keeps the runtime config simple and makes DOE expansion clean later.

So this tutorial gives you:

- a real 5-fold CV config shape
- one concrete slice you can run immediately
- the exact config pattern that tutorial 2 will fan out into all folds

## Files

- `configs/pgp_svm_cv_fold0.yaml`
- `CheMLFlow_Tutorial_01_Single_Config.ipynb`

## What the config does

- dataset: `tutorials/data/pgp_broccatelli.csv`
- target column: existing binary `Activity`
- split: `mode: cv`, `n_splits: 5`, `repeats: 1`
- preprocess: `scaler: standard`
- model: `svm`

## Local Run

From the repo root:

```bash
CHEMLFLOW_CONFIG=tutorials/01_single_config_colab/configs/pgp_svm_cv_fold0.yaml python main.py
```

## Colab Run

Open in Colab:

- [CheMLFlow_Tutorial_01_Single_Config.ipynb](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/01_single_config_colab/CheMLFlow_Tutorial_01_Single_Config.ipynb)

The notebook:

1. clones the repo
2. installs the minimum dependencies for this tutorial
3. runs the config
4. prints the metrics JSON

## Expected Outputs

This tutorial writes artifacts under:

- `tutorials/01_single_config_colab/artifacts/data/pgp_svm_cv_fold0`
- `tutorials/01_single_config_colab/artifacts/runs/pgp_svm_cv_fold0`

Look for:

- `svm_metrics.json`
- `svm_split_metrics.json`
- `run.log`
- `run_config.yaml`
