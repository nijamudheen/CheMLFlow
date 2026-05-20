from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def train_chemprop_model(
    curated_df: pd.DataFrame,
    target_column: str,
    split_indices: dict[str, Any],
    output_dir: str,
    random_state: int = 42,
    task_type: str = "classification",
    model_config: dict[str, Any] | None = None,
    *,
    row_index_col: str,
    ensure_dir: Callable[[str], None],
    require_chemprop: Callable[[], None],
    resolve_chemprop_foundation_config: Callable[[dict[str, Any]], tuple[str, str | None, bool]],
    as_bool: Callable[[Any], bool],
    resolve_chemprop_split_positions: Callable[..., tuple[list[int], list[int], list[int], list[int]]],
    ensure_binary_labels: Callable[[pd.Series], pd.Series],
    resolve_chemprop_predictor_ctor: Callable[[Any, str], Any],
    validate_classification_score_values: Callable[..., np.ndarray],
    sigmoid: Callable[[np.ndarray | pd.Series | list[float]], np.ndarray],
    classification_metrics_from_outputs: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]],
    validate_regression_metric_inputs: Callable[..., tuple[np.ndarray, np.ndarray]],
    safe_r2: Callable[[Any, Any], float | None],
    safe_mae: Callable[[Any, Any], float | None],
    save_split_metrics_artifacts: Callable[[str, str, dict[str, dict[str, float | None]]], tuple[str | None, str | None]],
    save_classification_split_plots: Callable[[str, str, dict[str, dict[str, Any]]], dict[str, str]],
    save_regression_parity_plots: Callable[[str, str, dict[str, tuple[Any, Any]]], dict[str, str]],
    save_params: Callable[[dict[str, Any], str], None],
    save_metrics_json: Callable[[dict[str, Any], str], None],
    train_result_cls: Callable[[str, str, str], Any],
) -> tuple[object, Any]:
    """Train Chemprop in-process while delegating shared helpers to callbacks."""
    ensure_dir(output_dir)
    require_chemprop()
    model_config = model_config or {}
    foundation_mode, foundation_checkpoint, freeze_encoder = resolve_chemprop_foundation_config(model_config)

    smiles_column = model_config.get("smiles_column", "canonical_smiles")
    if smiles_column not in curated_df.columns:
        raise ValueError(
            f"chemprop requires smiles_column={smiles_column!r} in curated_df. "
            "Ensure curate emits canonical_smiles or override model.smiles_column."
        )
    if target_column not in curated_df.columns:
        raise ValueError(f"Target column {target_column!r} not found in curated_df.")

    df = curated_df.reset_index(drop=True).copy()
    allow_legacy_split_positions = as_bool(model_config.get("allow_legacy_split_positions", False))
    tr_idx, va_idx, te_idx, row_ids = resolve_chemprop_split_positions(
        df,
        split_indices,
        row_index_col=row_index_col,
        allow_legacy_positions=allow_legacy_split_positions,
    )
    if not tr_idx or not te_idx:
        raise ValueError("chemprop training requires non-empty train and test splits.")
    if not va_idx:
        raise ValueError(
            "chemprop training requires an explicit validation split from the split node. "
            "Set split.val_size > 0."
        )
    if task_type == "classification":
        y_all = ensure_binary_labels(df[target_column]).astype(int)
    else:
        y_all = pd.to_numeric(df[target_column], errors="coerce")
        if y_all.isna().any():
            raise ValueError(
                "chemprop regression requires numeric target values; "
                f"found non-numeric entries in target_column={target_column!r}."
            )

    df["_row_id"] = row_ids
    df["_label"] = y_all.reset_index(drop=True)

    params = dict(model_config.get("params", {}))
    batch_size = int(params.get("batch_size", 64))
    max_epochs = int(params.get("max_epochs", 30))
    max_lr = float(params.get("max_lr", params.get("lr", 1e-3)))
    init_lr = float(params.get("init_lr", max_lr / 10.0))
    final_lr = float(params.get("final_lr", max_lr / 10.0))
    ff_hidden = int(params.get("ffn_hidden_dim", 300))
    mp_hidden = int(params.get("mp_hidden_dim", 300))
    depth = int(params.get("mp_depth", 3))
    plot_split_performance = as_bool(model_config.get("plot_split_performance", False))

    if os.environ.get("PYTEST_CURRENT_TEST") and "max_epochs" not in params:
        max_epochs = 2

    logging.info(
        "Training start (chemprop): task=%s N=%s train=%s val=%s test=%s",
        task_type,
        len(df),
        len(tr_idx),
        len(va_idx),
        len(te_idx),
    )

    import numpy as _np
    import torch
    from lightning import pytorch as pl
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    from chemprop import data, featurizers, models, nn

    pl.seed_everything(int(random_state), workers=True)

    def _to_datapoints(rows: pd.DataFrame) -> list:
        points = []
        for smi, y in zip(rows[smiles_column].astype(str).tolist(), rows["_label"].tolist()):
            points.append(data.MoleculeDatapoint.from_smi(smi, y=_np.array([float(y)], dtype=_np.float32)))
        return points

    all_points = _to_datapoints(df)
    train_points = [all_points[i] for i in tr_idx]
    val_points = [all_points[i] for i in va_idx]
    test_points = [all_points[i] for i in te_idx]

    mol_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_data = data.MoleculeDataset(train_points, featurizer=mol_featurizer)
    val_data = data.MoleculeDataset(val_points, featurizer=mol_featurizer)
    test_data = data.MoleculeDataset(test_points, featurizer=mol_featurizer)
    if hasattr(data, "build_dataloader"):
        train_loader = data.build_dataloader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        train_eval_loader = data.build_dataloader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader = data.build_dataloader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = data.build_dataloader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = data.MolGraphDataLoader(
            train_data, mol_featurizer, batch_size=batch_size, shuffle=True, num_workers=0
        )
        train_eval_loader = data.MolGraphDataLoader(
            train_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )
        val_loader = data.MolGraphDataLoader(
            val_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = data.MolGraphDataLoader(
            test_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )

    import inspect

    foundation_hparams: dict[str, Any] | None = None
    if foundation_mode == "chemeleon":
        assert foundation_checkpoint is not None
        payload = torch.load(foundation_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: expected dict."
            )
        if "hyper_parameters" not in payload or "state_dict" not in payload:
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: expected keys 'hyper_parameters' and 'state_dict'."
            )
        hyper_parameters = payload["hyper_parameters"]
        state_dict = payload["state_dict"]
        if not isinstance(hyper_parameters, dict):
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: 'hyper_parameters' must be a dict."
            )
        foundation_hparams = hyper_parameters
        mp = nn.BondMessagePassing(**hyper_parameters)
        mp.load_state_dict(state_dict)
        if freeze_encoder:
            for param in mp.parameters():
                param.requires_grad = False
            mp.eval()
        logging.info(
            "Chemprop foundation init: mode=%s checkpoint=%s freeze_encoder=%s",
            foundation_mode,
            foundation_checkpoint,
            freeze_encoder,
        )
    else:
        mp = nn.BondMessagePassing(d_h=mp_hidden, depth=depth)

    mp_output_dim = int(getattr(mp, "output_dim", mp_hidden))
    configured_or_ckpt_depth = (
        foundation_hparams.get("depth")
        if isinstance(foundation_hparams, dict) and "depth" in foundation_hparams
        else depth
    )
    mp_depth_effective = int(getattr(mp, "depth", configured_or_ckpt_depth))
    agg = nn.MeanAggregation()

    predictor_ctor = resolve_chemprop_predictor_ctor(nn, task_type)
    ffn_sig = inspect.signature(predictor_ctor.__init__)
    ffn_kwargs: dict[str, Any] = {"n_tasks": 1}
    if "d_mp" in ffn_sig.parameters:
        ffn_kwargs["d_mp"] = mp_output_dim
    if "input_dim" in ffn_sig.parameters:
        ffn_kwargs["input_dim"] = mp_output_dim
    if "d_h" in ffn_sig.parameters:
        ffn_kwargs["d_h"] = ff_hidden
    if "hidden_dim" in ffn_sig.parameters:
        ffn_kwargs["hidden_dim"] = ff_hidden
    ffn = predictor_ctor(**ffn_kwargs)
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=None,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    checkpointing = ModelCheckpoint(
        dirpath=output_dir,
        filename="chemprop-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpointing]
    if foundation_mode == "chemeleon" and freeze_encoder:
        from lightning.pytorch.callbacks import Callback

        class _FrozenEncoderEvalCallback(Callback):
            def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
                for attr_name in ("message_passing", "mp", "encoder"):
                    module = getattr(pl_module, attr_name, None)
                    if isinstance(module, torch.nn.Module):
                        module.eval()

        callbacks.append(_FrozenEncoderEvalCallback())
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        deterministic=True,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def _extract_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)) and obj:
            return _extract_tensor(obj[0])
        if isinstance(obj, dict) and obj:
            if "preds" in obj:
                return _extract_tensor(obj["preds"])
            return _extract_tensor(next(iter(obj.values())))
        raise TypeError(f"Unsupported prediction batch type: {type(obj)!r}")

    def _predict_batches(loader):
        try:
            return trainer.predict(dataloaders=loader, ckpt_path="best", weights_only=False)
        except TypeError:
            return trainer.predict(dataloaders=loader, ckpt_path="best")

    def _predict_values(loader, *, context: str) -> np.ndarray:
        pred_batches = _predict_batches(loader)
        if isinstance(pred_batches, list):
            preds_t = torch.cat(
                [_extract_tensor(p).detach().cpu().reshape(-1, 1) for p in pred_batches],
                dim=0,
            )
            preds = preds_t.numpy().reshape(-1)
        else:
            preds = _extract_tensor(pred_batches).detach().cpu().numpy().reshape(-1)
        preds = preds.astype(float)
        if task_type == "classification":
            preds = validate_classification_score_values(
                preds,
                context=f"{context}: raw classification scores",
            )
            if preds.size > 0 and (np.nanmin(preds) < 0.0 or np.nanmax(preds) > 1.0):
                preds = sigmoid(preds)
            preds = validate_classification_score_values(
                preds,
                context=f"{context}: classification probabilities",
            )
        return preds

    preds = _predict_values(test_loader, context="chemprop test scoring")

    metrics = {
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "init_lr": float(init_lr),
        "max_lr": float(max_lr),
        "final_lr": float(final_lr),
        "mp_hidden_dim": int(mp_output_dim),
        "mp_depth": int(mp_depth_effective),
        "ffn_hidden_dim": int(ff_hidden),
        "foundation": foundation_mode,
        "foundation_checkpoint": foundation_checkpoint,
        "freeze_encoder": bool(freeze_encoder),
    }
    if task_type == "classification":
        y_true, preds, y_pred, classification_metrics = classification_metrics_from_outputs(
            df.loc[te_idx, "_label"].to_numpy(dtype=int),
            preds,
            (preds >= 0.5).astype(int),
            context="chemprop test scoring",
        )
        metrics.update(classification_metrics)
    else:
        y_true, y_pred = validate_regression_metric_inputs(
            df.loc[te_idx, "_label"].to_numpy(dtype=float),
            preds.astype(float),
            context="chemprop test scoring",
        )
        metrics.update(
            {
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )
    if plot_split_performance:
        train_preds = _predict_values(train_eval_loader, context="chemprop train scoring")
        val_preds = _predict_values(val_loader, context="chemprop val scoring")
        if task_type == "classification":
            train_y, train_preds, train_pred_labels, train_metrics = classification_metrics_from_outputs(
                df.loc[tr_idx, "_label"].to_numpy(dtype=int),
                train_preds,
                (train_preds >= 0.5).astype(int),
                context="chemprop train scoring",
            )
            val_y, val_preds, val_pred_labels, val_metrics = classification_metrics_from_outputs(
                df.loc[va_idx, "_label"].to_numpy(dtype=int),
                val_preds,
                (val_preds >= 0.5).astype(int),
                context="chemprop val scoring",
            )
            split_metrics: dict[str, dict[str, float | None]] = {
                "train": train_metrics,
                "val": val_metrics,
                "test": {
                    "auc": metrics["auc"],
                    "auprc": metrics["auprc"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                },
            }
        else:
            train_y = df.loc[tr_idx, "_label"].to_numpy(dtype=float)
            val_y = df.loc[va_idx, "_label"].to_numpy(dtype=float)
            split_metrics = {
                "train": {"r2": safe_r2(train_y, train_preds), "mae": safe_mae(train_y, train_preds)},
                "val": {"r2": safe_r2(val_y, val_preds), "mae": safe_mae(val_y, val_preds)},
                "test": {"r2": metrics["r2"], "mae": metrics["mae"]},
            }
        split_metrics_path, split_plot_path = save_split_metrics_artifacts(
            output_dir,
            "chemprop",
            split_metrics,
        )
        if split_metrics_path:
            metrics["split_metrics_path"] = split_metrics_path
        if split_plot_path:
            metrics["split_metrics_plot_path"] = split_plot_path
        if task_type == "classification":
            metrics.update(
                save_classification_split_plots(
                    output_dir,
                    "chemprop",
                    {
                        "train": {
                            "y_true": train_y,
                            "y_proba": train_preds,
                            "y_pred": train_pred_labels,
                        },
                        "val": {
                            "y_true": val_y,
                            "y_proba": val_preds,
                            "y_pred": val_pred_labels,
                        },
                        "test": {
                            "y_true": y_true,
                            "y_proba": preds,
                            "y_pred": y_pred,
                        },
                    },
                )
            )
        else:
            parity_paths = save_regression_parity_plots(
                output_dir,
                "chemprop",
                {
                    "train": (train_y, train_preds),
                    "val": (val_y, val_preds),
                    "test": (y_true, y_pred),
                },
            )
            for split_name, path in parity_paths.items():
                metrics[f"parity_plot_{split_name}_path"] = path

    model_path = os.path.join(output_dir, "chemprop_best_model.ckpt")
    best_model_path = checkpointing.best_model_path
    if best_model_path and os.path.isfile(best_model_path):
        import shutil

        shutil.copyfile(best_model_path, model_path)
    else:
        trainer.save_checkpoint(model_path)
    params_path = os.path.join(output_dir, "chemprop_best_params.pkl")
    metrics_path = os.path.join(output_dir, "chemprop_metrics.json")
    best_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "init_lr": init_lr,
        "max_lr": max_lr,
        "final_lr": final_lr,
        "mp_hidden_dim": mp_output_dim,
        "mp_depth": mp_depth_effective,
        "ffn_hidden_dim": ff_hidden,
        "foundation": foundation_mode,
        "foundation_checkpoint": foundation_checkpoint,
        "freeze_encoder": bool(freeze_encoder),
    }
    save_params(best_params, params_path)
    save_metrics_json(metrics, metrics_path)

    pred_path = os.path.join(output_dir, "chemprop_predictions.csv")
    out_pred = df.loc[te_idx, ["_row_id", smiles_column, target_column]].copy()
    out_pred.rename(columns={target_column: "y_true", smiles_column: "smiles"}, inplace=True)
    out_pred["y_pred"] = y_pred
    if task_type == "classification":
        out_pred["y_proba"] = preds
    out_pred.to_csv(pred_path, index=False)

    report_keys = ["auc", "auprc", "accuracy", "f1"] if task_type == "classification" else ["r2", "mae"]
    logging.info(
        "Training complete (chemprop): task=%s metrics=%s",
        task_type,
        {k: metrics[k] for k in report_keys},
    )
    logging.info("Artifacts: model=%s metrics=%s params=%s preds=%s", model_path, metrics_path, params_path, pred_path)

    return mpnn, train_result_cls(model_path, params_path, metrics_path)
