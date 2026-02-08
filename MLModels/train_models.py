import os
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    model_path: str
    params_path: str
    metrics_path: str

@dataclass
class DLSearchConfig:
    model_class: Callable
    search_space: Dict[str, Any]
    default_params: Dict[str, Any]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _resolve_n_jobs(model_config: Dict[str, Any] | None = None) -> int:
    """Resolve parallelism level for scikit-learn/joblib.

    Default to single-thread under pytest (subprocesses inherit PYTEST_CURRENT_TEST),
    since some sandboxes/CI environments disallow the syscalls loky uses to size its pool.
    """
    model_config = model_config or {}

    value = model_config.get("n_jobs")
    if value is None:
        env = os.environ.get("CHEMLFLOW_N_JOBS")
        if env:
            try:
                value = int(env)
            except ValueError:
                logging.warning("Invalid CHEMLFLOW_N_JOBS=%r; falling back to defaults.", env)

    if value is None:
        value = 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1

    try:
        value_int = int(value)
    except (TypeError, ValueError):
        logging.warning("Invalid n_jobs=%r; using 1.", value)
        return 1
    if value_int == 0:
        logging.warning("n_jobs=0 is invalid; using 1.")
        return 1

    # Some environments (notably sandboxed macOS setups) raise PermissionError on
    # os.sysconf("SC_SEM_NSEMS_MAX"), which joblib/loky calls when starting a
    # process pool. Fall back to single-thread execution to keep pipelines runnable.
    if value_int != 1:
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            logging.warning(
                "Parallel joblib backend is not permitted in this environment; forcing n_jobs=1."
            )
            return 1
        except Exception:
            # If sysconf is unavailable/unsupported, let joblib decide.
            pass
    return value_int


def _safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        logging.warning("AUROC undefined for single-class test set.")
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        logging.warning("AUPRC undefined for single-class test set.")
        return None
    return float(average_precision_score(y_true, y_score))


def _save_roc_curve(output_dir: str, model_type: str, y_true, y_score) -> str | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    roc_path = os.path.join(output_dir, f"{model_type}_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    return roc_path


def _ensure_binary_labels(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("Expected a single label column for classification.")
        series = series.iloc[:, 0]
    values = series.dropna().unique()
    if len(values) == 0:
        raise ValueError("Empty label series for classification.")
    if len(values) > 2:
        raise ValueError(f"Expected binary labels; got {len(values)} classes.")

    if set(values).issubset({0, 1}):
        return series.astype(int)

    def _coerce_numeric(val):
        if isinstance(val, (int, float, np.generic)) and val in {0, 1}:
            return int(val)
        if isinstance(val, str):
            token = val.strip().lower()
            if token in {"0", "1"}:
                return int(token)
            try:
                parsed = float(token)
            except ValueError:
                return None
            if parsed in {0.0, 1.0}:
                return int(parsed)
        return None

    def _normalize(v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    normalized = {_normalize(v) for v in values}
    if normalized == {"active", "inactive"}:
        mapping = {"active": 1, "inactive": 0}
    elif normalized == {"inactive", "active"}:
        mapping = {"active": 1, "inactive": 0}
    else:
        coerced = series.map(_coerce_numeric)
        if coerced.notna().all():
            return coerced.astype(int)
        raise ValueError("Classification labels must be 0/1 or active/inactive; add label.normalize.")

    return series.map(lambda v: mapping.get(_normalize(v)))

def _run_optuna(
    config: DLSearchConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_evals: int,
    random_state: int,
    patience: int,
) -> Tuple[object, dict]:
    try:
        import optuna
    except Exception as exc:
        raise ImportError("optuna is required for DL hyperparameter search.") from exc
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )
    
    best_model = None
    best_score = float("-inf")
    
    def objective(trial: optuna.Trial) -> float:
        nonlocal best_model, best_score
        
        # Sample hyperparameters from search space
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
        
        model = config.model_class(params)
        
        # Train
        result = _train_dl(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            random_state=random_state,
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 32)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            patience=patience,
        )
        
        # Evaluate on validation set
        trained_model = result["model"]
        y_pred = _predict_dl(trained_model, X_val)
        r2 = r2_score(y_val, y_pred)
        
        # Track best
        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            logging.info(f"New best R2={r2:.4f} with params: {params}")
        
        return r2  # Optuna maximizes by default when direction="maximize"
    
    # Create and run study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)
    
    best_params = study.best_params
    logging.info(f"Optuna complete. Best R2={best_score:.4f}, params={best_params}")
    
    return best_model, best_params

def _initialize_model(
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    input_dim: int = None,
    n_jobs: int = -1,
):
    if model_type == "random_forest":
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
        base_rf = RandomForestRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "svm":
        param_grid_svm = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5],
        }
        return GridSearchCV(
            estimator=SVR(kernel="rbf"),
            param_grid=param_grid_svm,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
        )
    if model_type == "decision_tree":
        param_dist_dt = {
            "max_depth": [int(x) for x in np.linspace(5, 50, num=10)] + [None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
        base_dt = DecisionTreeRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_dt,
            param_distributions=param_dist_dt,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "xgboost":
        param_dist_xgb = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.1, 1, 10, 100],
        }
        base_xgb = XGBRegressor(objective="reg:squarederror", random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist_xgb,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "ensemble":
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            max_features="sqrt",
            bootstrap=False,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=random_state,
        )
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state,
        )
        return VotingRegressor([("rf", rf), ("xgb", xgb)], n_jobs=n_jobs)
    
    if model_type == "dl_simple":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.simpleregressionnn import SimpleRegressionNN
        return DLSearchConfig(
            model_class=lambda params: SimpleRegressionNN(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_dim", 256),
                use_tropical=params.get("use_tropical", False),
            ),
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                "use_tropical": {"type": "categorical", "choices": [True, False]},
            },
            default_params={
                "hidden_dim": 256,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    if model_type == "dl_deep":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.deepregressionnn import DeepRegressionNN
        return DLSearchConfig(
            model_class=lambda params: DeepRegressionNN(
                input_dim=input_dim,
                hidden_dims=[params.get("hidden_dim", 128)] * params.get("num_layers", 3),
                dropout_rate=params.get("dropout_rate", 0.2),
                use_tropical=params.get("use_tropical", False),
            ),
            search_space={
                "num_layers": {"type": "categorical", "choices": [2, 3, 4, 5]},
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                "use_tropical": {"type": "categorical", "choices": [True, False]},
            },
            default_params={
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout_rate": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type == "dl_gru":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.gruregressor import GRURegressor
        return DLSearchConfig(
            model_class=lambda params: GRURegressor(
                seq_len=input_dim,
                input_size=params.get("input_size", 1),
                hidden_size=params.get("hidden_size", 512),
                num_layers=params.get("num_layers", 2),
                bidirectional=params.get("bidirectional", True),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "num_layers": {"type": "categorical", "choices": [1, 2]},
                "bidirectional": {"type": "categorical", "choices": [True, False]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "input_size": 1,
                "hidden_size": 512,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    if model_type == "dl_resmlp":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.resmlp import ResMLP
        return DLSearchConfig(
            model_class=lambda params: ResMLP(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_dim", 512),
                n_blocks=params.get("n_blocks", 4),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [128, 256, 512, 1024]},
                "n_blocks": {"type": "categorical", "choices": [2, 3, 4, 6, 8]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "hidden_dim": 512,
                "n_blocks": 4,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    
    if model_type == "dl_tabtransformer":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.tabtransformer import TabTransformer
        return DLSearchConfig(
            model_class=lambda params: TabTransformer(
                input_dim=input_dim,
                embed_dim=params.get("embed_dim", 128),
                n_heads=params.get("n_heads", 8),
                n_layers=params.get("n_layers", 4),
                dropout=params.get("dropout", 0.1),
            ),
            search_space={
                "embed_dim": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
                "n_layers": {"type": "categorical", "choices": [2, 3, 4, 6]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.4, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "embed_dim": 128,
                "n_heads": 8,
                "n_layers": 4,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    
    if model_type == "dl_aereg":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.aeregressor import Autoencoder, AERegressor
        return DLSearchConfig(
            model_class=lambda params: AERegressor(
                pretrained_encoder=Autoencoder(
                    input_dim=input_dim,
                    bottleneck=params.get("bottleneck", 64),
                ).encoder,
                bottleneck=params.get("bottleneck", 64),
                dropout=params.get("dropout", 0.1),
            ),
            search_space={
                "bottleneck": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "bottleneck": 64,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    raise ValueError(f"Unsupported model type: {model_type}")

# DL training helper functions
def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _is_dl_model(model_type: str) -> bool:
    return model_type.startswith("dl_")
def _train_dl(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    ) -> Dict[str, Any]:
    """Train a PyTorch model. Returns dict with model and best_params."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = _get_device()
    model = model.to(device)
    
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    
    # Train/val split
    n_val = max(1, int(0.1 * len(X_t)))
    gen = torch.Generator().manual_seed(random_state)
    idx = torch.randperm(len(X_t), generator=gen)
    X_tr, X_val = X_t[idx[n_val:]], X_t[idx[:n_val]]
    y_tr, y_val = y_t[idx[n_val:]], y_t[idx[:n_val]]
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_loss, best_state, wait = float('inf'), None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).view(-1, 1), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).view(-1, 1), y_val).item()
        
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if epoch % 20 == 0:
                logging.info(f"[Epoch {epoch}] New best val_loss={val_loss:.4f}")
        else:
            wait += 1
            if wait >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return {"model": model, "best_params": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate}}

def _predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Predict with a PyTorch model."""
    import torch
    device = _get_device()
    model = model.to(device).eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out = model(X_t[i:i + batch_size]).cpu().numpy().flatten()
            preds.append(out)
    return np.concatenate(preds)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    random_state: int = 42,
    cv_folds: int = 5,
    search_iters: int = 100,
    use_hpo: bool = False,        
    hpo_trials: int = 30,          
    patience: int = 20,
    task_type: str = "regression",
    model_config: Dict[str, Any] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[object, TrainResult]:
    _ensure_dir(output_dir)
    is_dl = _is_dl_model(model_type)
    model_config = model_config or {}
    n_jobs = _resolve_n_jobs(model_config)

    logging.info(
        "Training start: model=%s task=%s X_train=%s X_test=%s",
        model_type,
        task_type,
        X_train.shape,
        X_test.shape,
    )

    if task_type == "classification" and model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        y_train = _ensure_binary_labels(y_train)
        y_test = _ensure_binary_labels(y_test)
        if y_val is not None:
            y_val = _ensure_binary_labels(y_val)

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": random_state,
            "verbose": False,
        }
        params.update(model_config.get("params", {}))
        estimator = CatBoostClassifier(**params)
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = (X_val, y_val)
        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True
        estimator.fit(X_train, y_train, **fit_kwargs)
        y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        estimator.save_model(model_path)
        best_params = estimator.get_params()
        metrics = {
            "auc": _safe_auc(y_test, y_pred_proba),
            "auprc": _safe_auprc(y_test, y_pred_proba),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        roc_path = _save_roc_curve(output_dir, model_type, y_test, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path
        params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
        joblib.dump(best_params, params_path)
        pd.Series(metrics).to_json(metrics_path)
        logging.info("Training complete (classification): metrics=%s", metrics)
        logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)
        return estimator, TrainResult(model_path, params_path, metrics_path)

    if task_type == "classification":
        raise ValueError(f"Unsupported classification model type: {model_type}")

    model = _initialize_model(
        model_type,
        random_state,
        cv_folds,
        search_iters,
        input_dim=X_train.shape[1] if is_dl else None,
        n_jobs=n_jobs,
    )
    if isinstance(model, DLSearchConfig):
        # ── DL Training ──
        if use_hpo:
            logging.info(f"Running optuna for {model_type} ({hpo_trials} evals)")
            estimator, best_params = _run_optuna(
                model, X_train.values, y_train.values, hpo_trials, random_state, patience
            )
        else:
            logging.info(f"Training DL model: {model_type} (default params)")
            nn_model = model.model_class(model.default_params)
            result = _train_dl(
                nn_model, X_train.values, y_train.values, random_state,
                epochs=model.default_params["epochs"],
                batch_size=model.default_params["batch_size"],
                learning_rate=model.default_params["learning_rate"],
                patience=patience,
            )
            estimator = result["model"]
            best_params = result["best_params"]
        
        y_pred = _predict_dl(estimator, X_test.values)
        import torch
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        torch.save(estimator.state_dict(), model_path)
    else:

        logging.info(f"Training ML model: {model_type}")
        model.fit(X_train, y_train)

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        joblib.dump(estimator, model_path)
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    joblib.dump(best_params, params_path)   
    pd.Series(metrics).to_json(metrics_path)
    logging.info("Training complete (regression): metrics=%s", metrics)
    logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)

    return estimator, TrainResult(model_path, params_path, metrics_path)

def load_model(model_path: str, model_type: str, input_dim: int = None) -> object:
    is_dl = _is_dl_model(model_type)
    
    if is_dl:
        import torch
        if input_dim is None:
            raise ValueError("input_dim required to load DL models")
        
        params_path = model_path.replace("_best_model.pth", "_best_params.pkl")
        if os.path.exists(params_path):
            saved_params = joblib.load(params_path)
        else:
            saved_params = {}

        config = _initialize_model(model_type, random_state=42, cv_folds=5, 
                                   search_iters=100, input_dim=input_dim)
        
        model_params = {**config.default_params, **saved_params}
        
        model = config.model_class(model_params)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    if model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    return joblib.load(model_path)

def run_explainability(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    ) -> None:
    _ensure_dir(output_dir)
    is_dl = model_type.startswith("dl_")
    n_jobs = _resolve_n_jobs()

    try:
        import shap
    except Exception as exc:
        logging.warning("SHAP is not available; skipping SHAP explainability. %s", exc)
        return

    if is_dl:
        import torch
        class _SklearnWrapper:
            def __init__(self, model):
                self.model = model
            def fit(self, X, y):
                return self
            def predict(self, X):
                return _predict_dl(self.model, X.values if hasattr(X, 'values') else X)
            def score(self, X, y):
                y_pred = self.predict(X)
                return r2_score(y, y_pred)
        
        wrapped_estimator = _SklearnWrapper(estimator)
        result = permutation_importance(
            wrapped_estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
        )
    else:
        result = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=n_jobs
        )


    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance": result.importances_mean}
    ).sort_values(by="importance", ascending=False)
    importance_path = os.path.join(output_dir, f"{model_type}_permutation_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    plt.figure(figsize=(10, 6))
    importance_df.head(20).plot.bar(x="feature", y="importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_permutation_importance.png"))
    plt.close()

    try:
        if model_type in ["random_forest", "decision_tree", "xgboost", "catboost_classifier"]:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
        
        elif model_type == "ensemble":
            explainer = shap.Explainer(estimator.predict, X_test.iloc[:background_samples])
            shap_values = explainer(X_test)
        
        elif model_type == "svm":
            background = shap.sample(X_test, min(background_samples, len(X_test)))
            explainer = shap.KernelExplainer(estimator.predict, background)
            X_explain = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_explain)
            X_test = X_explain 
        
        elif model_type.startswith("dl_"): 
            device = _get_device()
            estimator.to(device).eval()
            
            n_bg = min(background_samples, len(X_train))
            n_ex = min(100, len(X_test))
            
            # Convert to numpy arrays (not tensors) for SHAP
            X_bg_np = X_train.iloc[:n_bg].values.astype(np.float32)
            X_ex_np = X_test.iloc[:n_ex].values.astype(np.float32)
            
            # Use KernelExplainer instead (more reliable for PyTorch)
            def model_predict(X):
                return _predict_dl(estimator, X)
            
            explainer = shap.KernelExplainer(model_predict, X_bg_np)
            shap_values = explainer.shap_values(X_ex_np, nsamples=100)
            
            # Keep DataFrame for feature names in plot
            X_test = X_test.iloc[:n_ex]
        
        else:
            background = shap.sample(X_test, min(background_samples, len(X_test)))
            explainer = shap.KernelExplainer(estimator.predict, background)
            X_explain = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_explain)
            X_test = X_explain

        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, f"{model_type}_shap_summary.png"), bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logging.warning("SHAP explainability failed: %s", exc)
