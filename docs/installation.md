# Installation

CheMLFlow is usually installed from a local source checkout. A CheMLFlow installation uses one
conda or mamba environment, named `chemlflow_env` below. Install the model backends you need into
that same environment; individual models and DOE studies do not require separate environments.
The environment remains isolated from unrelated projects so compiled scientific packages, RDKit,
and deep-learning dependencies do not collide with them.

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

conda create -n chemlflow_env python=3.12
conda activate chemlflow_env

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

Install PyTorch if you want to run neural network model families such as `dl_simple`,
`dl_deep`, `dl_gru`, `dl_resmlp`, `dl_tabtransformer`, or `dl_aereg`.

For CUDA systems, choose the PyTorch index URL that matches your driver/CUDA stack. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning
```

For CPU-only use:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
```

For Apple Silicon or other CPU-only systems, install the standard PyTorch wheels:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
```

## Full foundation-model installation in the same environment

The paper-aligned TabPFN path uses `tabpfn==7.0.1` and the separately downloaded
CheMeleon message-passing checkpoint. Install both backends into the already
activated `chemlflow_env`. On Intel macOS, install PyTorch 2.5.1 from
conda-forge before the editable install because PyPI does not publish a
compatible wheel for that platform:

```bash
conda activate chemlflow_env
# Intel macOS only:
conda install -c conda-forge pytorch=2.5.1

python -m pip install -e ".[chemprop,tabpfn,molecular_eda]"
```

TabPFN 7.0.1 requires PyTorch 2.5 or newer. Linux, Windows, and Apple Silicon can
use the appropriate PyTorch distribution for their hardware. Intel macOS can
use conda-forge's native `osx-64` build, but it is CPU-only and the full DOE will
be materially slower than on a CUDA host. Every platform still uses one
`chemlflow_env` for CheMLFlow, Chemprop, CheMeleonFP, TabPFN, the dashboard, and
the DOE runner.

Model weights are a separate one-time setup step. The Python installation does
not bundle either checkpoint:

- CheMeleonFP reads the explicit local `models/chemeleon_mp.pt` checkpoint.
- TabPFN downloads and caches its TabPFN 2.6 classifier weights on first use.
  The classifier checkpoint is approximately 43 MB. The separate 51.6 MB
  regressor checkpoint is not needed by the PGP classification DOE. These
  weights have their own license: review it at
  https://huggingface.co/Prior-Labs/tabpfn_2_6 before use. If Hugging Face
  requests authentication, run `hf auth login` in the same environment. On a
  headless host, provide `HF_TOKEN` with read access; do not commit the token.

The CheMeleon checkpoint remains local and explicit:

```bash
mkdir -p models
curl -L https://zenodo.org/records/15460715/files/chemeleon_mp.pt \
  -o models/chemeleon_mp.pt
```

After reviewing the TabPFN terms (and authenticating if requested), prime and
verify the classifier cache from the same environment before launching a
background DOE:

```bash
conda activate chemlflow_env
python - <<'PY'
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

X = np.arange(40, dtype=float).reshape(20, 2)
y = np.array([0, 1] * 10)
model = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2_6,
    device="cpu",
    n_estimators=1,
)
model.fit(X, y)
print(f"TabPFN 2.6 ready: {model.model_path}")
PY
```

The first run downloads the classifier checkpoint to TabPFN's user cache.
Subsequent CheMLFlow runs in `chemlflow_env` reuse it. The full PGP DOE needs
both this cached TabPFN checkpoint and `models/chemeleon_mp.pt`.

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
