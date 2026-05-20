from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)

_DEFAULT_INCLUDE = {
    "overview": True,
    "missingness": True,
    "numeric_histograms": True,
    "correlation_heatmap": True,
    "target_distribution": True,
    "class_balance": True,
    "descriptor_scatter": False,
    "descriptor_boxplots": False,
}
_MAX_HIST_COLUMNS = 12
_MAX_BOXPLOT_COLUMNS = 6


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _resolve_include_flags(config: dict[str, Any]) -> dict[str, bool]:
    include_cfg = config.get("include", {}) if isinstance(config, dict) else {}
    resolved = dict(_DEFAULT_INCLUDE)
    if isinstance(include_cfg, dict):
        for key in resolved:
            if key in include_cfg:
                resolved[key] = _as_bool(include_cfg.get(key))
    return resolved


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).replace("\ufeff", "").strip() for col in cleaned.columns]
    return cleaned


def _coerce_numeric_frame(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    numeric_data: dict[str, pd.Series] = {}
    for col in df.columns:
        if col == target_column:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid_fraction = float(series.notna().mean()) if len(series) else 0.0
        if series.notna().sum() > 0 and valid_fraction >= 0.5:
            numeric_data[col] = series
    return pd.DataFrame(numeric_data, index=df.index)


def _save_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(
    df: pd.DataFrame,
    output_dir: str,
    *,
    task_type: str | None,
    target_column: str | None,
    numeric_columns: list[str],
    smiles_column: str | None,
) -> str:
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axis("off")
    lines = [
        "CheMLFlow Dataset Overview",
        "",
        f"Rows: {rows}",
        f"Columns: {cols}",
        f"Numeric columns: {len(numeric_columns)}",
        f"Missing cells: {missing_cells}",
        f"Task type: {task_type or 'unspecified'}",
        f"Target column: {target_column or 'not set'}",
        f"SMILES column: {smiles_column or 'not detected'}",
        "",
        "Columns:",
        ", ".join(list(df.columns[:12])) + (" ..." if len(df.columns) > 12 else ""),
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    path = os.path.join(output_dir, "dataset_overview.png")
    _save_figure(fig, path)
    return path


def _plot_missingness(df: pd.DataFrame, output_dir: str) -> str | None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None
    top = missing.head(20)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4C78A8")
    ax.set_title("Missing values by column")
    ax.set_xlabel("Missing count")
    ax.set_ylabel("Column")
    path = os.path.join(output_dir, "missingness.png")
    _save_figure(fig, path)
    return path


def _plot_numeric_histograms(df: pd.DataFrame, output_dir: str, numeric_columns: list[str]) -> str | None:
    if not numeric_columns:
        return None
    cols = numeric_columns[:_MAX_HIST_COLUMNS]
    n = len(cols)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.3 * nrows))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    for ax, col in zip(axes_list, cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        sns.histplot(series, bins=30, ax=ax, color="#59A14F")
        ax.set_title(col)
    for ax in axes_list[n:]:
        ax.axis("off")
    path = os.path.join(output_dir, "numeric_histograms.png")
    _save_figure(fig, path)
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: str, numeric_columns: list[str]) -> str | None:
    if len(numeric_columns) < 2:
        return None
    cols = numeric_columns[:20]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 4, 0.6 * len(cols) + 3))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation heatmap")
    path = os.path.join(output_dir, "correlation_heatmap.png")
    _save_figure(fig, path)
    return path


def _plot_target_distribution(
    df: pd.DataFrame,
    output_dir: str,
    *,
    task_type: str | None,
    target_column: str | None,
) -> str | None:
    if not target_column or target_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if task_type == "classification":
        counts = df[target_column].astype(str).value_counts(dropna=False)
        sns.barplot(x=counts.index.tolist(), y=counts.values.tolist(), ax=ax, color="#F28E2B")
        ax.set_ylabel("Count")
    else:
        series = pd.to_numeric(df[target_column], errors="coerce").dropna()
        if series.empty:
            plt.close(fig)
            return None
        sns.histplot(series, bins=30, ax=ax, color="#F28E2B")
        ax.set_ylabel("Frequency")
    ax.set_title(f"Target distribution: {target_column}")
    ax.set_xlabel(target_column)
    path = os.path.join(output_dir, "target_distribution.png")
    _save_figure(fig, path)
    return path


def _plot_class_balance(df: pd.DataFrame, output_dir: str, class_column: str | None) -> str | None:
    if not class_column or class_column not in df.columns:
        return None
    counts = df[class_column].astype(str).value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=counts.index.tolist(), y=counts.values.tolist(), ax=ax, color="#E15759")
    ax.set_title(f"Class balance: {class_column}")
    ax.set_xlabel(class_column)
    ax.set_ylabel("Count")
    path = os.path.join(output_dir, "class_balance.png")
    _save_figure(fig, path)
    return path


def _plot_descriptor_scatter(
    df: pd.DataFrame,
    output_dir: str,
    *,
    scatter_columns: list[str],
    color_column: str | None,
) -> str | None:
    if len(scatter_columns) < 2:
        return None
    x_col, y_col = scatter_columns[:2]
    if x_col not in df.columns or y_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_kwargs: dict[str, Any] = {
        "data": df,
        "x": x_col,
        "y": y_col,
        "ax": ax,
        "alpha": 0.75,
    }
    if color_column and color_column in df.columns:
        plot_kwargs["hue"] = color_column
    sns.scatterplot(**plot_kwargs)
    ax.set_title(f"Scatter: {x_col} vs {y_col}")
    path = os.path.join(output_dir, "descriptor_scatter.png")
    _save_figure(fig, path)
    return path


def _plot_descriptor_boxplots(
    df: pd.DataFrame,
    output_dir: str,
    *,
    boxplot_columns: list[str],
    class_column: str | None,
) -> str | None:
    if not class_column or class_column not in df.columns:
        return None
    columns = [col for col in boxplot_columns[:_MAX_BOXPLOT_COLUMNS] if col in df.columns]
    if not columns:
        return None
    n = len(columns)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    for ax, col in zip(axes_list, columns):
        sns.boxplot(data=df, x=class_column, y=col, ax=ax)
        ax.set_title(col)
    for ax in axes_list[n:]:
        ax.axis("off")
    path = os.path.join(output_dir, "descriptor_boxplots.png")
    _save_figure(fig, path)
    return path


def run_generic_eda(
    *,
    input_path: str,
    output_dir: str,
    task_type: str | None = None,
    target_column: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    os.makedirs(output_dir, exist_ok=True)
    df = _clean_dataframe(pd.read_csv(input_path))

    if target_column and target_column not in df.columns:
        target_column = None

    smiles_column = str(cfg.get("smiles_column", "")).strip() or None
    if smiles_column and smiles_column not in df.columns:
        smiles_column = None

    includes = _resolve_include_flags(cfg)
    numeric_df = _coerce_numeric_frame(df, target_column)
    numeric_columns = _as_str_list(cfg.get("numeric_columns")) or list(numeric_df.columns)
    numeric_columns = [col for col in numeric_columns if col in numeric_df.columns]
    scatter_columns = _as_str_list(cfg.get("scatter_columns"))
    if not scatter_columns:
        scatter_columns = numeric_columns[:2]
    boxplot_columns = _as_str_list(cfg.get("boxplot_columns"))
    if not boxplot_columns:
        boxplot_columns = numeric_columns[:_MAX_BOXPLOT_COLUMNS]
    class_column = str(cfg.get("class_column", "")).strip() or None
    if not class_column and task_type == "classification" and target_column in df.columns:
        class_column = target_column
    if class_column and class_column not in df.columns:
        class_column = None
    color_column = class_column or target_column

    created: dict[str, str] = {}
    skipped: dict[str, str] = {}

    if includes["overview"]:
        created["overview"] = _plot_overview(
            df,
            output_dir,
            task_type=task_type,
            target_column=target_column,
            numeric_columns=numeric_columns,
            smiles_column=smiles_column,
        )
    else:
        skipped["overview"] = "disabled"

    if includes["missingness"]:
        path = _plot_missingness(df, output_dir)
        if path:
            created["missingness"] = path
        else:
            skipped["missingness"] = "no missing values found"
    else:
        skipped["missingness"] = "disabled"

    if includes["numeric_histograms"]:
        path = _plot_numeric_histograms(numeric_df, output_dir, numeric_columns)
        if path:
            created["numeric_histograms"] = path
        else:
            skipped["numeric_histograms"] = "no numeric columns available"
    else:
        skipped["numeric_histograms"] = "disabled"

    if includes["correlation_heatmap"]:
        path = _plot_correlation_heatmap(numeric_df, output_dir, numeric_columns)
        if path:
            created["correlation_heatmap"] = path
        else:
            skipped["correlation_heatmap"] = "fewer than two numeric columns available"
    else:
        skipped["correlation_heatmap"] = "disabled"

    if includes["target_distribution"]:
        path = _plot_target_distribution(df, output_dir, task_type=task_type, target_column=target_column)
        if path:
            created["target_distribution"] = path
        else:
            skipped["target_distribution"] = "target column is missing or not plottable"
    else:
        skipped["target_distribution"] = "disabled"

    if includes["class_balance"]:
        path = _plot_class_balance(df, output_dir, class_column)
        if path:
            created["class_balance"] = path
        else:
            skipped["class_balance"] = "classification target column not available"
    else:
        skipped["class_balance"] = "disabled"

    if includes["descriptor_scatter"]:
        path = _plot_descriptor_scatter(df, output_dir, scatter_columns=scatter_columns, color_column=color_column)
        if path:
            created["descriptor_scatter"] = path
        else:
            skipped["descriptor_scatter"] = "scatter_columns are unavailable"
    else:
        skipped["descriptor_scatter"] = "disabled"

    if includes["descriptor_boxplots"]:
        path = _plot_descriptor_boxplots(
            df,
            output_dir,
            boxplot_columns=boxplot_columns,
            class_column=class_column,
        )
        if path:
            created["descriptor_boxplots"] = path
        else:
            skipped["descriptor_boxplots"] = "boxplot columns or class column are unavailable"
    else:
        skipped["descriptor_boxplots"] = "disabled"

    manifest = {
        "input_path": input_path,
        "output_dir": output_dir,
        "task_type": task_type,
        "target_column": target_column,
        "class_column": class_column,
        "smiles_column": smiles_column,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "created": created,
        "skipped": skipped,
    }
    manifest_path = os.path.join(output_dir, "eda_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Generic EDA outputs written to %s", output_dir)
    return manifest
