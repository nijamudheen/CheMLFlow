from __future__ import annotations

import gc
import inspect
import logging
import random
from typing import Any, Callable

import numpy as np
from sklearn.metrics import r2_score

from .metrics import (
    safe_auc,
    validate_classification_metric_inputs,
    validate_regression_metric_inputs,
)


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


def _call_predict_fn(
    predict_fn: Callable[..., np.ndarray],
    model: object,
    X: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    try:
        signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return predict_fn(model, X, batch_size=batch_size)

    parameters = signature.parameters.values()
    supports_batch_size = "batch_size" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if supports_batch_size:
        return predict_fn(model, X, batch_size=batch_size)
    return predict_fn(model, X)


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
            if wait >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

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


def run_optuna(
    config: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_evals: int,
    random_state: int,
    patience: int,
    task_type: str = "regression",
    *,
    seed_fn: Callable[[int], None] | None = None,
    train_fn: Callable[..., dict[str, Any]] | None = None,
    predict_fn: Callable[..., np.ndarray] | None = None,
) -> tuple[object, dict[str, Any]]:
    try:
        import optuna
    except Exception as exc:
        raise ImportError("optuna is required for DL hyperparameter search.") from exc

    if X_val is None or y_val is None or len(y_val) == 0:
        raise ValueError("DL hyperparameter search requires an explicit validation split (X_val/y_val).")

    seed_fn = seed_fn or seed_dl_runtime
    train_fn = train_fn or train_dl
    predict_fn = predict_fn or predict_dl

    best_model = None
    best_score = float("-inf")
    pruned_reasons: list[str] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_model, best_score

        params = {}
        for name, spec in config.search_space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )

        trial_seed = int(random_state) + int(trial.number) + 1
        seed_fn(trial_seed)
        model = config.model_class(params)

        result = train_fn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 32)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            patience=patience,
            random_state=trial_seed,
            task_type=task_type,
        )

        trained_model = result["model"]
        y_pred = _call_predict_fn(
            predict_fn,
            trained_model,
            X_val,
            batch_size=max(1, int(params.get("batch_size", 32))),
        )

        if task_type == "classification":
            try:
                y_true, y_score = validate_classification_metric_inputs(
                    y_val,
                    np.asarray(y_pred).reshape(-1),
                    context="optuna validation scoring",
                )
            except ValueError as exc:
                pruned_reasons.append(str(exc))
                raise optuna.exceptions.TrialPruned(str(exc)) from exc
            y_proba = 1.0 / (1.0 + np.exp(-y_score))
            score = safe_auc(y_true, y_proba)
            if score is None:
                message = "AUC undefined (single-class val set)"
                pruned_reasons.append(message)
                raise optuna.exceptions.TrialPruned(message)
        else:
            try:
                y_true, y_hat = validate_regression_metric_inputs(
                    y_val,
                    y_pred,
                    context="optuna validation scoring",
                )
            except ValueError as exc:
                pruned_reasons.append(str(exc))
                raise optuna.exceptions.TrialPruned(str(exc)) from exc
            score = float(r2_score(y_true, y_hat))

        if score > best_score:
            best_score = score
            best_model = trained_model
            logging.info("New best score=%.4f with params: %s", score, params)

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        del model, result, y_pred
        gc.collect()

        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)

    if best_model is None:
        detail = pruned_reasons[-1] if pruned_reasons else "no completed trials"
        raise ValueError(
            "DL hyperparameter search completed no valid trials; "
            f"last prune reason: {detail}"
        )

    best_params = study.best_params
    logging.info("Optuna complete. Best score=%.4f, params=%s", best_score, best_params)
    return best_model, best_params
