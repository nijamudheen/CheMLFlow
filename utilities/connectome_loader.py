"""
Connectome loader for the Adaptive NVAR pipeline.

Reads a WormWiring-style hermaphrodite connectome workbook, selects a subgraph
by degree or random sampling, optionally binarizes / randomizes / normalizes
the adjacency matrix. The runtime config supplies the workbook path and
options; nothing is hard-coded so the same loader can serve other connectomes
that share the WormWiring layout.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectomeBundle:
    """Output of `build_connectome` — adjacency plus provenance."""

    adjacency: np.ndarray  # [n, n] float32
    node_names: List[str]
    node_indices: np.ndarray  # indices into the original (full) adjacency
    sheet_name: str

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])


# ---------------------------------------------------------------------------
# Workbook parsing
# ---------------------------------------------------------------------------


def load_connectome_xlsx(
    xlsx_path: str, sheet_name: Optional[str] = None
) -> tuple[np.ndarray, List[str], str]:
    """Load a WormWiring-layout adjacency from an .xlsx workbook.

    Returns (square_adjacency, node_names, resolved_sheet_name). The function
    is a near-verbatim port of the loader in the user notebooks; it picks the
    "hermaphrodite chemical" sheet by default and tolerates ragged label rows.
    """
    import pandas as pd

    # pandas reads .xlsx via openpyxl, which is an optional dep CheMLFlow does
    # not install by default. Detect it up front so the user sees something
    # actionable instead of a buried ImportError from inside pandas.
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading the connectome workbook requires the optional `openpyxl` "
            "package, which is not installed in this environment. Install with "
            "`pip install openpyxl` (or `conda install -c conda-forge openpyxl`) "
            "and re-run. This is only needed for `dl_connectome_nvar`; the "
            "rest of CheMLFlow does not depend on it."
        ) from exc

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Could not find connectome workbook at '{xlsx_path}'. "
            "Provide the path via train.model.params.connectome_xlsx."
        )

    excel_file = pd.ExcelFile(xlsx_path)
    available_sheets = [str(s).strip() for s in excel_file.sheet_names]

    if sheet_name is None:
        candidates = [
            s
            for s in available_sheets
            if ("herm" in s.lower() or "hermaph" in s.lower()) and "chem" in s.lower()
        ]
        if not candidates:
            raise ValueError(
                "Could not auto-pick a sheet (looked for hermaphrodite chemical). "
                f"Available sheets: {available_sheets}. Set sheet_name explicitly."
            )
        sheet_name = candidates[0]
    elif sheet_name not in available_sheets:
        lower_map = {s.lower(): s for s in available_sheets}
        if str(sheet_name).strip().lower() in lower_map:
            sheet_name = lower_map[str(sheet_name).strip().lower()]
        else:
            raise ValueError(
                f"Sheet '{sheet_name}' not found in {xlsx_path}. "
                f"Available: {available_sheets}"
            )

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # WormWiring layout:
    #   row labels start at row index 3, column index 2
    #   col labels start at row index 2, column index 3
    #   numeric block starts at (row=3, col=3)
    row_labels_raw = df.iloc[3:, 2]
    col_labels_raw = df.iloc[2, 3:]
    row_labels = row_labels_raw.where(row_labels_raw.notna(), "").astype(str).str.strip()
    col_labels = col_labels_raw.where(col_labels_raw.notna(), "").astype(str).str.strip()
    valid_row = row_labels != ""
    valid_col = col_labels != ""
    row_labels = row_labels[valid_row].reset_index(drop=True)
    col_labels = col_labels[valid_col].reset_index(drop=True)

    A_full = (
        df.iloc[3:, 3:]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    A = A_full[np.ix_(valid_row.to_numpy(), valid_col.to_numpy())]

    common = sorted(set(row_labels.tolist()) & set(col_labels.tolist()))
    if len(common) < 2:
        raise ValueError(
            f"Sheet '{sheet_name}' has too few common row/col labels to form a square adjacency."
        )

    row_idx = [i for i, x in enumerate(row_labels) if x in common]
    col_idx = [j for j, x in enumerate(col_labels) if x in common]
    A_sq = A[np.ix_(row_idx, col_idx)]
    row_common = row_labels.iloc[row_idx].tolist()
    col_common = col_labels.iloc[col_idx].tolist()

    col_pos = {name: j for j, name in enumerate(col_common)}
    perm = [col_pos[name] for name in row_common]
    A_sq = A_sq[:, perm]

    return A_sq.astype(np.float32), row_common, sheet_name


# ---------------------------------------------------------------------------
# Subgraph selection / randomization / normalization
# ---------------------------------------------------------------------------


def select_subgraph(
    A: np.ndarray,
    names: List[str],
    n_select: int,
    *,
    seed: int = 2025,
    mode: str = "top_degree",
) -> tuple[np.ndarray, List[str], np.ndarray]:
    n_total = int(A.shape[0])
    if n_select >= n_total:
        idx = np.arange(n_total)
        return A.astype(np.float32, copy=True), [names[i] for i in idx], idx

    A_bin = (A != 0).astype(np.float32)
    degree = A_bin.sum(axis=0) + A_bin.sum(axis=1)

    if mode == "top_degree":
        idx = np.argsort(-degree, kind="stable")[:n_select]
    elif mode == "random":
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n_total, size=n_select, replace=False))
    else:
        raise ValueError("mode must be 'top_degree' or 'random'.")

    A_sub = A[np.ix_(idx, idx)].astype(np.float32)
    names_sub = [names[i] for i in idx]
    return A_sub, names_sub, np.asarray(idx)


def randomize_directed_adjacency(
    A: np.ndarray, *, seed: int = 2025, swap_factor: int = 10
) -> np.ndarray:
    """Degree-preserving randomization with an ER fallback if swap fails."""
    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "networkx is required for connectome randomization. "
            "Install with `pip install networkx`."
        ) from exc

    A_bin = (A != 0).astype(np.int8)
    np.fill_diagonal(A_bin, 0)

    G = nx.from_numpy_array(A_bin, create_using=nx.DiGraph)
    if G.number_of_edges() < 2:
        return A_bin.astype(np.float32)

    G_rand = G.copy()
    nswap = max(1, swap_factor * G.number_of_edges())
    max_tries = max(1000, 20 * nswap)

    try:
        nx.algorithms.swap.directed_edge_swap(
            G_rand, nswap=nswap, max_tries=max_tries, seed=seed
        )
        A_rand = nx.to_numpy_array(G_rand, dtype=np.float32)
        np.fill_diagonal(A_rand, 0.0)
        return A_rand.astype(np.float32)
    except Exception as exc:
        LOGGER.warning(
            "directed_edge_swap failed (%s); using density-matched ER fallback.", exc
        )
        rng = np.random.RandomState(seed)
        n = A_bin.shape[0]
        p = A_bin.sum() / max(1, n * (n - 1))
        A_rand = (rng.rand(n, n) < p).astype(np.float32)
        np.fill_diagonal(A_rand, 0.0)
        return A_rand


def normalize_adjacency(A: np.ndarray, mode: str = "maxabs") -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    if mode in (None, "", "none"):
        return A.astype(np.float32)
    if mode == "maxabs":
        scale = float(np.max(np.abs(A)))
        if scale <= 0:
            return A.astype(np.float32)
        return (A / scale).astype(np.float32)
    if mode == "spectral":
        eigvals = np.linalg.eigvals(A.astype(np.float64))
        radius = float(np.max(np.abs(eigvals)))
        if radius <= 0:
            return A.astype(np.float32)
        return (A / radius).astype(np.float32)
    raise ValueError(f"normalize_adjacency mode={mode!r}; expected 'none', 'maxabs', 'spectral'.")


# ---------------------------------------------------------------------------
# High-level entry point used by the timeseries trainer
# ---------------------------------------------------------------------------


def build_connectome(
    *,
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    n_select: int,
    selection_mode: str = "top_degree",
    selection_seed: int = 2025,
    binarize: bool = False,
    randomize: bool = False,
    randomize_swap_factor: int = 10,
    randomize_seed: int = 2025,
    normalization: str = "maxabs",
) -> ConnectomeBundle:
    """End-to-end: load workbook -> select subgraph -> (optional bin/rand/norm)."""
    full_adj, full_names, sheet = load_connectome_xlsx(xlsx_path, sheet_name)
    n_select_capped = min(int(n_select), int(full_adj.shape[0]))

    sub_adj, sub_names, sub_idx = select_subgraph(
        full_adj,
        full_names,
        n_select=n_select_capped,
        seed=selection_seed,
        mode=selection_mode,
    )
    if binarize:
        sub_adj = (sub_adj != 0).astype(np.float32)
    if randomize:
        sub_adj = randomize_directed_adjacency(
            sub_adj,
            seed=randomize_seed,
            swap_factor=randomize_swap_factor,
        )
    sub_adj = normalize_adjacency(sub_adj, mode=normalization)
    return ConnectomeBundle(
        adjacency=sub_adj,
        node_names=sub_names,
        node_indices=sub_idx,
        sheet_name=sheet,
    )
