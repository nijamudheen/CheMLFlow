from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from MLModels import train_models
from MLModels.training import api as training_api


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def _resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    target_lower = str(target_column).strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == target_lower:
            return str(column)
    raise ValueError(f"Target column {target_column!r} not found in {list(df.columns)!r}.")


def _load_model_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model-config JSON must be an object: {config_path}")
    return payload


def _coerce_numeric_features(X: pd.DataFrame, *, context: str) -> pd.DataFrame:
    try:
        numeric = X.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError(
            f"{context}: non-numeric feature columns detected. "
            "Provide numeric descriptors/features for non-chemprop models."
        ) from exc
    if numeric.shape[1] == 0:
        raise ValueError(f"{context}: no feature columns remain after dropping target.")
    return numeric


def _split_row_indices(
    y: pd.Series,
    *,
    task_type: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test-size must be between 0 and 1.")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val-size must be >= 0 and < 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test-size + val-size must be < 1.")

    all_idx = np.arange(len(y))
    y_array = np.asarray(y)
    stratify = y_array if task_type == "classification" and len(np.unique(y_array)) > 1 else None
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if val_size <= 0.0:
        return np.asarray(train_idx), np.asarray([], dtype=int), np.asarray(test_idx)

    rel_val_size = float(val_size) / float(1.0 - test_size)
    y_train = y_array[np.asarray(train_idx)]
    stratify_train = y_train if task_type == "classification" and len(np.unique(y_train)) > 1 else None
    train_idx, val_idx = train_test_split(
        np.asarray(train_idx),
        test_size=rel_val_size,
        random_state=random_state,
        stratify=stratify_train,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _frame_splits_from_indices(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    if len(val_idx) == 0:
        return X_train, y_train, X_test, y_test, None, None
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    return X_train, y_train, X_test, y_test, X_val, y_val


def _cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data_path)
    target_col = _resolve_target_column(df, args.target_column)
    y = df[target_col]
    model_config = _load_model_config(args.model_config_json)

    train_idx, val_idx, test_idx = _split_row_indices(
        y=y,
        task_type=args.task_type,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    if args.model_type == "chemprop":
        model_config = dict(model_config)
        model_config.setdefault("smiles_column", args.smiles_column)
        _, result = train_models.train_chemprop_model(
            curated_df=df,
            target_column=target_col,
            split_indices={
                "train": [int(i) for i in train_idx.tolist()],
                "val": [int(i) for i in val_idx.tolist()],
                "test": [int(i) for i in test_idx.tolist()],
            },
            output_dir=args.output_dir,
            random_state=args.random_state,
            task_type=args.task_type,
            model_config=model_config,
        )
    else:
        X = _coerce_numeric_features(df.drop(columns=[target_col]), context="train")
        y = y.reset_index(drop=True)
        X_train, y_train, X_test, y_test, X_val, y_val = _frame_splits_from_indices(
            X,
            y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        _, result = training_api.train_from_frames(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=args.model_type,
            output_dir=args.output_dir,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            search_iters=args.search_iters,
            use_hpo=args.use_hpo,
            hpo_trials=args.hpo_trials,
            patience=args.patience,
            task_type=args.task_type,
            model_config=model_config,
            X_val=X_val,
            y_val=y_val,
        )

    print(f"model: {result.model_path}")
    print(f"params: {result.params_path}")
    print(f"metrics: {result.metrics_path}")
    return 0


def _predict_regression(estimator: object, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type.startswith("dl_"):
        return np.asarray(train_models._predict_dl(estimator, X.values), dtype=float).reshape(-1)
    return np.asarray(estimator.predict(X), dtype=float).reshape(-1)


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.model_type == "chemprop":
        raise ValueError("predict for model_type=chemprop is not implemented in this CLI yet.")

    df = pd.read_csv(args.test_path)
    target_col = None
    if args.target_column:
        target_col = _resolve_target_column(df, args.target_column)

    feature_df = df.drop(columns=[target_col]) if target_col is not None else df.copy()
    X = _coerce_numeric_features(feature_df, context="predict")
    input_dim = args.input_dim if args.input_dim is not None else int(X.shape[1])
    estimator = training_api.load(
        model_path=args.model_path,
        model_type=args.model_type,
        input_dim=input_dim,
    )

    out_df = df.copy()
    if args.task_type == "classification":
        y_proba, y_pred, y_score = train_models._predict_classification_outputs(
            estimator=estimator,
            model_type=args.model_type,
            X=X,
        )
        out_df["pred_label"] = y_pred
        out_df["pred_proba"] = y_proba
        out_df["pred_score"] = y_score
    else:
        out_df["pred_0"] = _predict_regression(estimator, X, args.model_type)

    preds_path = Path(args.preds_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(preds_path, index=False)
    print(f"predictions: {preds_path}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    if args.model_type == "chemprop":
        raise ValueError("explain for model_type=chemprop is not implemented in this CLI yet.")

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    target_col_train = _resolve_target_column(train_df, args.target_column)
    target_col_test = _resolve_target_column(test_df, args.target_column)
    X_train = _coerce_numeric_features(train_df.drop(columns=[target_col_train]), context="explain train")
    X_test = _coerce_numeric_features(test_df.drop(columns=[target_col_test]), context="explain test")
    y_test = test_df[target_col_test]

    input_dim = args.input_dim if args.input_dim is not None else int(X_test.shape[1])
    estimator = training_api.load(
        model_path=args.model_path,
        model_type=args.model_type,
        input_dim=input_dim,
    )
    training_api.run_explainability(
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_type=args.model_type,
        output_dir=args.output_dir,
        background_samples=args.background_samples,
        task_type=args.task_type,
    )
    print(f"explainability_dir: {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemlflow-training",
        description=(
            "Train, predict, and explain models with CheMLFlow.\n"
            "This CLI is a thin wrapper over MLModels.training.api."
        ),
        formatter_class=_HelpFormatter,
        epilog=textwrap.dedent(
            """\
            Quickstart examples:
              python -m MLModels.training.cli train --data-path tests/fixtures/data/training_cli_quickstart_regression.csv --target-column target --model-type random_forest --task-type regression --output-dir runs/cli_quickstart
              python -m MLModels.training.cli predict --test-path tests/fixtures/data/training_cli_quickstart_regression.csv --target-column target --model-path runs/cli_quickstart/random_forest_best_model.pkl --model-type random_forest --task-type regression --preds-path runs/cli_quickstart/predictions.csv
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{train,predict,explain}")

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from a CSV file.",
        description=(
            "Train a model from one CSV. The target column is removed from features.\n"
            "For chemprop, this command uses row-index split IDs and forwards to train_chemprop_model."
        ),
        formatter_class=_HelpFormatter,
    )
    train_parser.add_argument("--data-path", required=True, help="Input CSV path.")
    train_parser.add_argument("--target-column", required=True, help="Target column name.")
    train_parser.add_argument("--model-type", required=True, help="Model type (e.g., random_forest, dl_simple, chemprop).")
    train_parser.add_argument("--task-type", default="regression", choices=["regression", "classification"])
    train_parser.add_argument("--output-dir", required=True, help="Output directory for model artifacts.")
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--cv-folds", type=int, default=5)
    train_parser.add_argument("--search-iters", type=int, default=100)
    train_parser.add_argument("--use-hpo", action="store_true", help="Enable HPO for DL models.")
    train_parser.add_argument("--hpo-trials", type=int, default=30)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--val-size", type=float, default=0.0)
    train_parser.add_argument("--model-config-json", default=None, help="Optional model config JSON file.")
    train_parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="SMILES column for chemprop training (used only when --model-type chemprop).",
    )
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction using a saved model.",
        description="Load a saved model and write predictions for a CSV file.",
        formatter_class=_HelpFormatter,
    )
    predict_parser.add_argument("--test-path", required=True, help="CSV file for inference.")
    predict_parser.add_argument("--model-path", required=True, help="Saved model path.")
    predict_parser.add_argument("--model-type", required=True, help="Model type used during training.")
    predict_parser.add_argument("--preds-path", required=True, help="Destination CSV path for predictions.")
    predict_parser.add_argument("--task-type", default="regression", choices=["regression", "classification"])
    predict_parser.add_argument("--target-column", default=None, help="Optional target column to exclude from features.")
    predict_parser.add_argument("--input-dim", type=int, default=None, help="Optional input dimension override for DL models.")
    predict_parser.set_defaults(func=_cmd_predict)

    explain_parser = subparsers.add_parser(
        "explain",
        help="Generate explainability artifacts.",
        description="Generate permutation importance and SHAP artifacts using a saved model.",
        formatter_class=_HelpFormatter,
    )
    explain_parser.add_argument("--train-path", required=True, help="Training CSV path for background/features.")
    explain_parser.add_argument("--test-path", required=True, help="Test CSV path for explainability.")
    explain_parser.add_argument("--target-column", required=True, help="Target column name.")
    explain_parser.add_argument("--model-path", required=True, help="Saved model path.")
    explain_parser.add_argument("--model-type", required=True, help="Model type used during training.")
    explain_parser.add_argument("--output-dir", required=True, help="Output directory for explainability artifacts.")
    explain_parser.add_argument("--task-type", default="regression", choices=["regression", "classification"])
    explain_parser.add_argument("--background-samples", type=int, default=100)
    explain_parser.add_argument("--input-dim", type=int, default=None, help="Optional input dimension override for DL models.")
    explain_parser.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
