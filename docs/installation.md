# Installation

CheMLFlow is usually installed from a local source checkout. We recommend using a fresh conda or
mamba environment so compiled scientific packages, RDKit, and optional deep-learning dependencies
do not collide with other projects.

The commands below assume Python 3.12 and should be run from the `CheMLFlow/` repository root.

## Option 1. Standard source install

Clone the repository and create an environment:

```bash
git clone https://github.com/nijamudheen/CheMLFlow.git
cd CheMLFlow

conda create -n chemlflow_env python=3.12
conda activate chemlflow_env
```

Install compiled dependencies from conda-forge first:

```bash
conda install -c conda-forge \
  numpy scipy scikit-learn matplotlib-base seaborn \
  lightgbm xgboost catboost rdkit shap numba llvmlite
```

Then install the Python requirements without asking pip to re-resolve the compiled stack:

```bash
pip install -r requirements.txt --no-deps
```

This is the most reliable path on macOS and works well on Linux clusters where compiled packages
are easier to manage through conda.

## Option 2. Editable developer install

Use this when you plan to modify CheMLFlow code and import it as a package:

```bash
git clone https://github.com/nijamudheen/CheMLFlow.git
cd CheMLFlow

conda create -n chemlflow_dev python=3.12
conda activate chemlflow_dev

conda install -c conda-forge \
  numpy scipy scikit-learn matplotlib-base seaborn \
  lightgbm xgboost catboost rdkit shap numba llvmlite

pip install -r requirements.txt --no-deps
pip install -e . --no-deps
```

For tests:

```bash
pip install pytest
```

## Optional. Deep-learning models

Install PyTorch and Optuna if you want to run neural network model families such as `dl_simple`,
`dl_deep`, `dl_gru`, `dl_resmlp`, `dl_tabtransformer`, or `dl_aereg`.

For CUDA systems, choose the PyTorch index URL that matches your driver/CUDA stack. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning optuna
```

For CPU-only use:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning optuna
```

For Apple Silicon or other CPU-only systems, install the standard PyTorch wheels:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning optuna
```

## Optional. TDC datasets

Install `pytdc` only if you need Therapeutic Data Commons datasets, such as the PGP benchmark
export workflow:

```bash
pip install pytdc
python -c "from tdc.benchmark_group import admet_group; print('pytdc ok')"
```

If `pytdc` pulls in conflicting dependencies, use a small separate export environment and write the
dataset to CSV before running CheMLFlow in the main environment.

## Check the install

From the repository root:

```bash
python -m MLModels.training.cli --help
python scripts/generate_doe.py --help
```

For a short train-and-predict workflow, continue with the [Quickstart](quickstart.md).

## Troubleshooting

- Run commands from the repo root so relative paths resolve.
- If `conda activate` is unavailable, run `conda init zsh` and restart the shell.
- If NumPy, RDKit, SHAP, numba, or llvmlite report binary compatibility errors, reinstall those
  packages from conda-forge in the active environment.
- CatBoost is stable on Python 3.12; Python 3.13 is not recommended for this project yet.
- `pytdc` is optional and is not installed by `requirements.txt`.
