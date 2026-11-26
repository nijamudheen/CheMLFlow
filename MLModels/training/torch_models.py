"""PyTorch helpers shared by the neural models.

Device setting
--------------
Neural models read ``train.model.params.device``. Allowed values:

- ``auto`` (default): CUDA if available, else Apple MPS, else CPU.
- ``cuda``: require a CUDA GPU; raise if none is visible.
- ``mps``:  require Apple MPS; raise if unavailable.
- ``cpu``:  always use the CPU.

The setting applies to the ``dl_*`` tabular models, Chemprop/Chemeleon and the
time-series models. Do not set it for scikit-learn models (they reject unknown
parameters). XGBoost has its own native ``device`` parameter with different
values (``cpu``, ``cuda``, ``cuda:<n>``); ``auto`` and ``mps`` are not valid there.

Configs should say ``auto`` or ``cuda``, never a GPU index. Choosing *which* GPU
is the launcher's job: ``CUDA_VISIBLE_DEVICES``, Slurm, or
``scripts/run_doe_local.py --gpus``. Inside a process, ``cuda`` means the first
GPU that process can see.

Prediction runs on whatever device the model is already on. Models loaded from
disk are on the CPU.
"""

from __future__ import annotations

import gc
import logging
import os
import random
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cuda", "cpu", "mps")

_DEVICE_DIAGNOSTIC_EMITTED = False


def normalize_device(value: Any) -> str:
    """Validate a device setting without importing torch. ``None``/empty means ``auto``."""
    req = str(value if value is not None else "auto").strip().lower() or "auto"
    if req not in DEVICE_CHOICES:
        raise ValueError(
            "train.model.params.device must be one of "
            + ", ".join(repr(c) for c in DEVICE_CHOICES)
            + f"; got {value!r}. Choose which GPU with CUDA_VISIBLE_DEVICES "
            "(or run_doe_local.py --gpus), not in the config."
        )
    return req


def log_device_diagnostics(requested: str, resolved) -> None:
    """One-shot dump of torch device visibility info. Fires once per process."""
    global _DEVICE_DIAGNOSTIC_EMITTED
    if _DEVICE_DIAGNOSTIC_EMITTED:
        return
    _DEVICE_DIAGNOSTIC_EMITTED = True

    import torch

    LOGGER.info("=" * 60)
    LOGGER.info("Device diagnostic (emitted once per process):")
    LOGGER.info("  torch.__version__         = %s", torch.__version__)
    LOGGER.info("  torch.version.cuda        = %s", getattr(torch.version, "cuda", None))
    LOGGER.info("  torch.cuda.is_available() = %s", torch.cuda.is_available())
    try:
        LOGGER.info("  torch.cuda.device_count() = %d", torch.cuda.device_count())
    except Exception as exc:
        LOGGER.info("  torch.cuda.device_count() raised: %s", exc)
    LOGGER.info(
        "  CUDA_VISIBLE_DEVICES env  = %s",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    LOGGER.info("  requested device          = %s", requested)
    LOGGER.info("  resolved device           = %s", resolved)
    LOGGER.info("=" * 60)


def resolve_device(requested: Any = "auto"):
    """Return a torch.device for ``requested`` (see the module docstring)."""
    import torch

    req = normalize_device(requested)
    if req == "cpu":
        device = torch.device("cpu")
    elif req == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            log_device_diagnostics(req, "REQUESTED CUDA BUT UNAVAILABLE")
            raise RuntimeError(
                "train.model.params.device='cuda' was requested but no CUDA "
                "device is visible to PyTorch. See the device diagnostic "
                "above. Common causes: (1) PyTorch was installed without "
                "CUDA support — `python -c 'import torch; print(torch.version.cuda)'` "
                "shows None; (2) Slurm did not allocate a GPU — `nvidia-smi` "
                "shows no devices, or CUDA_VISIBLE_DEVICES is empty; (3) the "
                "job ran on a CPU-only node. Re-submit with --gpus=1 and a "
                "CUDA-enabled PyTorch in the env, or set device: cpu/auto to "
                "fall back."
            )
        device = torch.device("cuda")
    elif req == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("train.model.params.device='mps' was requested but MPS is unavailable.")
        device = torch.device("mps")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    log_device_diagnostics(req, device)
    return device


def get_device(requested: Any = "auto"):
    """Backward-compatible alias for :func:`resolve_device`."""
    return resolve_device(requested)


def model_device(model):
    """Device of a model's first parameter or buffer; CPU if it has neither."""
    import torch

    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def release_accelerator_memory() -> None:
    """Drop unreachable tensors and return cached GPU memory to the driver."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    backends_mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and hasattr(mps, "empty_cache") and backends_mps is not None and backends_mps.is_available():
        mps.empty_cache()


def seed_dl_runtime(seed: int) -> None:
    """Seed Python/NumPy/PyTorch for deterministic DL training."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def train_dl(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    random_state: int = 42,
    task_type: str = "regression",
    device: Any = "auto",
) -> dict[str, Any]:
    """Train a PyTorch model. Returns dict with model, best_params and device.

    Data stays on the CPU; only the current batch is moved to ``device``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = resolve_device(device)
    model = model.to(device)
    batch_size = max(1, int(batch_size))

    X_t = torch.tensor(X_train, dtype=torch.float32)

    if X_val is None or y_val is None:
        raise ValueError("DL training requires X_val/y_val for early stopping.")
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    if task_type == "classification":
        y_t = torch.tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
    else:
        y_t = torch.tensor(np.asarray(y_train).reshape(-1, 1), dtype=torch.float32)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32)
        criterion = nn.MSELoss()

    dl_generator = torch.Generator()
    dl_generator.manual_seed(int(random_state))
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        generator=dl_generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            if task_type == "classification":
                loss = criterion(out.view(-1), by.view(-1))
            else:
                loss = criterion(out.view(-1, 1), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_n = 0
            for i in range(0, len(X_val_t), batch_size):
                bx_val = X_val_t[i : i + batch_size].to(device)
                by_val = y_val_t[i : i + batch_size].to(device)
                outv = model(bx_val)
                if task_type == "classification":
                    batch_loss = criterion(outv.view(-1), by_val.view(-1))
                else:
                    batch_loss = criterion(outv.view(-1, 1), by_val)
                batch_n = int(bx_val.shape[0])
                total_loss += float(batch_loss.item()) * batch_n
                total_n += batch_n
            val_loss = total_loss / total_n if total_n else float("inf")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if epoch % 20 == 0:
                logging.info("[Epoch %d] New best val_loss=%.4f", epoch, val_loss)
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_params": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
        "device": str(device),
    }


def predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Predict with a PyTorch model on the device it is already on."""
    import torch

    device = model_device(model)
    model = model.eval()
    X_arr = np.asarray(X, dtype=np.float32)
    if not X_arr.flags.writeable:
        X_arr = X_arr.copy()
    batch_size = max(1, int(batch_size))

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), batch_size):
            bx = torch.as_tensor(X_arr[i : i + batch_size], dtype=torch.float32, device=device)
            out = model(bx).cpu().numpy().flatten()
            preds.append(out)
    if not preds:
        return np.array([], dtype=float)
    return np.concatenate(preds)
