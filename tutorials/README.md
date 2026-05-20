# Tutorials

This folder is the low-friction onboarding path for CheMLFlow.

The goal is simple:

- start with one runnable config
- move to DOE generation once the config shape makes sense
- move to phase-2 style tuning only after the DOE workflow is clear

## Tutorial Roadmap

1. `00_dataset_eda`
   Load a local CSV and generate generic EDA outputs with a two-node workflow:
   - `get_data`
   - `analyze.eda`
   - Colab: [Open tutorial 00](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/00_dataset_eda/CheMLFlow_Tutorial_00_Dataset_EDA.ipynb)

2. `01_single_config_colab`
   Run one PGP classification config in a Colab-style workflow.
   This tutorial uses:
   - the bundled `tutorials/data/pgp_broccatelli.csv` dataset
   - Morgan fingerprints
   - `preprocess.scaler: standard`
   - `split.mode: cv` with `n_splits: 5`
   - a single runnable slice: `fold_index: 0`, `repeat_index: 0`
   - Colab: [Open tutorial 01](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/01_single_config_colab/CheMLFlow_Tutorial_01_Single_Config.ipynb)

3. `02_submit_doe`
   Planned. Generate a DOE from one dataset profile and inspect the manifest/summary outputs.

4. `03_submit_phase2`
   Planned. Launch a focused phase-2 tuning DOE from a parent winner.

## Important CV Note

CheMLFlow treats `split.mode: cv` as one fold per run.

That means tutorial 1 is still a "single config" tutorial, but the config is one execution slice from a 5-fold CV design. The full fold fanout belongs in the DOE tutorial.
