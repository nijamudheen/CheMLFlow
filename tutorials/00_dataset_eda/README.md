# Tutorial 00: Dataset EDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/00_dataset_eda/CheMLFlow_Tutorial_00_Dataset_EDA.ipynb)

This tutorial is the shortest path to "load a dataset and make plots" in CheMLFlow.

It uses a two-node workflow:

- `get_data`
- `analyze.eda`

For the first example, it uses the bundled tutorial PGP dataset:

- `tutorials/data/pgp_broccatelli.csv`

## Config

- `configs/pgp_raw_eda.yaml`
- `CheMLFlow_Tutorial_00_Dataset_EDA.ipynb`

## What it makes

For this raw PGP dataset, the notebook now starts with an interactive molecular landscape view, then walks through:

- interactive structure-first molecular map
- raw dataset EDA outputs
- target analysis
- SMILES-derived property analysis

The generic EDA flow itself should create:

- dataset overview
- missingness plot
- numeric histograms
- correlation heatmap
- target distribution
- class balance plot

## Run

From the repo root:

```bash
CHEMLFLOW_CONFIG=tutorials/00_dataset_eda/configs/pgp_raw_eda.yaml python main.py
```

Outputs go to:

- `tutorials/00_dataset_eda/artifacts/run/eda`

## Colab

Open in Colab:

- [CheMLFlow_Tutorial_00_Dataset_EDA.ipynb](https://colab.research.google.com/github/bsmith24/CheMLFlow/blob/phase2_doe_support/tutorials/00_dataset_eda/CheMLFlow_Tutorial_00_Dataset_EDA.ipynb)

The notebook keeps the config visible in a code cell, writes it to disk, then launches:

```bash
python main.py
```
