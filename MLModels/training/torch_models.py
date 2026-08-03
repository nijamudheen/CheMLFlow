from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from .progress import emit


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    progress_reporter: Any | None = None,
) -> dict[str, Any]:
    """Train a PyTorch model. Returns dict with model and best_params."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device()
    model = model.to(device)
    batch_size = max(1, int(batch_size))

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    if X_val is None or y_val is None:
        raise ValueError("DL training requires X_val/y_val for early stopping.")
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    if task_type == "classification":
        y_t = torch.tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32, device=device)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        y_t = torch.tensor(np.asarray(y_train).reshape(-1, 1), dtype=torch.float32, device=device)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32, device=device)
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

    emit(
        progress_reporter,
        "training_started",
        "epoch",
        unit="epoch",
        total=int(epochs),
        phase="training",
        message="PyTorch training started.",
    )

    epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for bx, by in loader:
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
                    bx_val = X_val_t[i : i + batch_size]
                    by_val = y_val_t[i : i + batch_size]
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

            emit(
                progress_reporter,
                "training_update",
                "epoch",
                current=epoch,
                total=int(epochs),
                phase="early_stopping" if wait else "training",
                message=f"Validation epoch {epoch} complete.",
                metrics={
                    "val_loss": val_loss,
                    "best_val_loss": best_loss,
                    "patience_wait": wait,
                },
            )
            if wait >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break
    except BaseException:
        emit(
            progress_reporter,
            "training_scope_finished",
            "epoch",
            status="failed",
            message="PyTorch training failed.",
        )
        raise

    emit(
        progress_reporter,
        "training_scope_finished",
        "epoch",
        message=(
            f"PyTorch training stopped after epoch {epoch}."
            if epoch < epochs
            else "PyTorch training completed."
        ),
        metrics={"best_val_loss": best_loss},
    )

    if best_state:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_params": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
    }


def predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Predict with a PyTorch model."""
    import torch

    device = get_device()
    model = model.to(device).eval()
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
