from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .metrics import safe_auprc


def save_roc_curve(output_dir: str, model_type: str, y_true: Any, y_score: Any) -> str | None:
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


def save_pr_curve(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true: Any,
    y_score: Any,
) -> str | None:
    y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
    y_score_arr = np.asarray(y_score).reshape(-1).astype(float)
    if y_true_arr.size == 0 or y_true_arr.size != y_score_arr.size:
        return None
    if len(np.unique(y_true_arr)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    ap = safe_auprc(y_true_arr, y_score_arr)
    prevalence = float(np.mean(y_true_arr))
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR")
    plt.hlines(
        prevalence,
        xmin=0.0,
        xmax=1.0,
        linestyles="--",
        colors="gray",
        label="Baseline",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    title = f"{model_type} {split_name} PR curve (N={len(y_true_arr)})"
    if ap is not None:
        title += f" AUPRC={ap:.3f}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(output_dir, f"{model_type}_pr_curve_{split_name}.png")
    plt.savefig(pr_path)
    plt.close()
    return pr_path


def save_confusion_matrix_plot(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true: Any,
    y_pred: Any,
) -> str | None:
    y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
    y_pred_arr = np.asarray(y_pred).reshape(-1).astype(int)
    if y_true_arr.size == 0 or y_true_arr.size != y_pred_arr.size:
        return None
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(f"{model_type} {split_name} confusion (N={len(y_true_arr)})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    cm_path = os.path.join(output_dir, f"{model_type}_confusion_{split_name}.png")
    fig.savefig(cm_path)
    plt.close(fig)
    return cm_path


def save_classification_split_plots(
    output_dir: str,
    model_type: str,
    split_outputs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    split_order = [name for name in ("train", "val", "test") if name in split_outputs]
    if not split_order:
        split_order = list(split_outputs.keys())
    if not split_order:
        return saved_paths

    combined_pr_fig, combined_pr_axes = plt.subplots(
        1, len(split_order), figsize=(6 * len(split_order), 5), squeeze=False
    )
    combined_cm_fig, combined_cm_axes = plt.subplots(
        1, len(split_order), figsize=(5.5 * len(split_order), 5), squeeze=False
    )
    any_pr = False
    any_cm = False

    for idx, split_name in enumerate(split_order):
        payload = split_outputs.get(split_name, {})
        y_true = payload.get("y_true")
        y_proba = payload.get("y_proba")
        y_pred = payload.get("y_pred")
        if y_true is None or y_proba is None:
            continue
        y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
        y_proba_arr = np.asarray(y_proba).reshape(-1).astype(float)
        if y_pred is None:
            y_pred_arr = (y_proba_arr >= 0.5).astype(int)
        else:
            y_pred_arr = np.asarray(y_pred).reshape(-1).astype(int)
        if (
            y_true_arr.size == 0
            or y_true_arr.size != y_proba_arr.size
            or y_true_arr.size != y_pred_arr.size
        ):
            continue

        pr_path = save_pr_curve(output_dir, model_type, split_name, y_true_arr, y_proba_arr)
        if pr_path:
            saved_paths[f"pr_curve_{split_name}_path"] = pr_path

        cm_path = save_confusion_matrix_plot(output_dir, model_type, split_name, y_true_arr, y_pred_arr)
        if cm_path:
            saved_paths[f"confusion_matrix_{split_name}_path"] = cm_path

        pr_ax = combined_pr_axes[0, idx]
        if len(np.unique(y_true_arr)) >= 2:
            precision, recall, _ = precision_recall_curve(y_true_arr, y_proba_arr)
            pr_ax.plot(recall, precision, label="PR")
            pr_ax.hlines(
                float(np.mean(y_true_arr)),
                xmin=0.0,
                xmax=1.0,
                linestyles="--",
                colors="gray",
                label="Baseline",
            )
            any_pr = True
        else:
            pr_ax.text(0.5, 0.5, "Single-class split", ha="center", va="center", transform=pr_ax.transAxes)
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.set_title(f"{split_name} PR (N={len(y_true_arr)})")
        pr_ax.grid(alpha=0.25)
        if pr_ax.get_legend_handles_labels()[0]:
            pr_ax.legend()

        cm_ax = combined_cm_axes[0, idx]
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        cm_im = cm_ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                cm_ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        cm_ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        cm_ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        cm_ax.set_title(f"{split_name} confusion (N={len(y_true_arr)})")
        any_cm = True
        combined_cm_fig.colorbar(cm_im, ax=cm_ax, fraction=0.046, pad=0.04)

    combined_pr_fig.tight_layout()
    combined_cm_fig.tight_layout()
    if any_pr:
        pr_all_path = os.path.join(output_dir, f"{model_type}_pr_curve_all_splits.png")
        combined_pr_fig.savefig(pr_all_path)
        saved_paths["pr_curve_all_splits_path"] = pr_all_path
    if any_cm:
        cm_all_path = os.path.join(output_dir, f"{model_type}_confusion_all_splits.png")
        combined_cm_fig.savefig(cm_all_path)
        saved_paths["confusion_matrix_all_splits_path"] = cm_all_path
    plt.close(combined_pr_fig)
    plt.close(combined_cm_fig)
    return saved_paths


def save_split_metrics_artifacts(
    output_dir: str,
    model_type: str,
    split_metrics: dict[str, dict[str, float | None]],
) -> tuple[str | None, str | None]:
    if not split_metrics:
        return None, None

    split_order = [name for name in ("train", "val", "test") if name in split_metrics]
    if not split_order:
        split_order = list(split_metrics.keys())

    metrics_json_path = os.path.join(output_dir, f"{model_type}_split_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(split_metrics, f, indent=2)

    df = pd.DataFrame.from_dict(split_metrics, orient="index")
    df = df.reindex(split_order)
    df = df.apply(pd.to_numeric, errors="coerce")
    metric_names = [name for name in df.columns if df[name].notna().any()]
    if not metric_names:
        return metrics_json_path, None

    fig, axes = plt.subplots(
        nrows=len(metric_names),
        ncols=1,
        figsize=(8, max(3, 2.6 * len(metric_names))),
        squeeze=False,
    )
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx, 0]
        values = df[metric_name]
        ax.bar(df.index.astype(str), values.values)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_type}_split_metrics.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return metrics_json_path, plot_path


def save_regression_parity_plots(
    output_dir: str,
    model_type: str,
    split_predictions: dict[str, tuple[Any, Any]],
) -> dict[str, str]:
    def _sanitize_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if y_true is None or y_pred is None:
            return None
        y_true_arr = np.asarray(y_true).reshape(-1)
        y_pred_arr = np.asarray(y_pred).reshape(-1)
        if y_true_arr.size == 0 or y_true_arr.size != y_pred_arr.size:
            return None
        finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        y_true_arr = y_true_arr[finite_mask]
        y_pred_arr = y_pred_arr[finite_mask]
        if y_true_arr.size == 0:
            return None
        return y_true_arr, y_pred_arr

    def _limits(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> tuple[float, float]:
        lo = float(min(np.min(y_true_arr), np.min(y_pred_arr)))
        hi = float(max(np.max(y_true_arr), np.max(y_pred_arr)))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        return lo - pad, hi + pad

    saved_paths: dict[str, str] = {}
    clean_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, payload in split_predictions.items():
        if payload is None or len(payload) != 2:
            continue
        y_true, y_pred = payload
        sanitized = _sanitize_pair(y_true, y_pred)
        if sanitized is None:
            continue
        y_true_arr, y_pred_arr = sanitized
        clean_pairs[split_name] = (y_true_arr, y_pred_arr)
        line_lo, line_hi = _limits(y_true_arr, y_pred_arr)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(y_true_arr, y_pred_arr, s=18, alpha=0.65, edgecolors="none")
        ax.plot([line_lo, line_hi], [line_lo, line_hi], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_type} {split_name} parity (N={len(y_true_arr)})")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        parity_path = os.path.join(output_dir, f"{model_type}_parity_{split_name}.png")
        fig.savefig(parity_path)
        plt.close(fig)
        saved_paths[split_name] = parity_path

    if clean_pairs:
        order = ["train", "val", "test"]
        all_true_parts = [clean_pairs[name][0] for name in order if name in clean_pairs]
        all_pred_parts = [clean_pairs[name][1] for name in order if name in clean_pairs]
        if all_true_parts and all_pred_parts:
            all_true = np.concatenate(all_true_parts)
            all_pred = np.concatenate(all_pred_parts)
            clean_pairs["all"] = (all_true, all_pred)

            line_lo, line_hi = _limits(all_true, all_pred)
            fig, axes = plt.subplots(2, 2, figsize=(11, 10), squeeze=False)
            panel_order = ["all", "train", "val", "test"]
            for idx, split_name in enumerate(panel_order):
                ax = axes[idx // 2, idx % 2]
                pair = clean_pairs.get(split_name)
                if pair is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{model_type} {split_name} parity (N=0)")
                    ax.set_xlabel("True")
                    ax.set_ylabel("Predicted")
                    ax.grid(alpha=0.25)
                    continue
                y_true_arr, y_pred_arr = pair
                ax.scatter(y_true_arr, y_pred_arr, s=16, alpha=0.6, edgecolors="none")
                ax.plot([line_lo, line_hi], [line_lo, line_hi], linestyle="--", color="gray", linewidth=1)
                ax.set_xlim(line_lo, line_hi)
                ax.set_ylim(line_lo, line_hi)
                ax.set_xlabel("True")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{model_type} {split_name} parity (N={len(y_true_arr)})")
                ax.grid(alpha=0.25)

            fig.tight_layout()
            combined_path = os.path.join(output_dir, f"{model_type}_parity_all_splits.png")
            fig.savefig(combined_path)
            plt.close(fig)
            saved_paths["all_splits"] = combined_path

    return saved_paths
