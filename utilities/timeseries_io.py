"""
Time-series I/O helpers for CheMLFlow's `timeseries` pipeline_type.

Supports two on-disk formats:

  * .npy  — 2D float array. The orientation flag `time_axis` selects which
            axis is time. Defaults to the longer axis when set to "auto".
  * .csv  — One row per timestep, optional first-column time index, all
            remaining columns are state variables.

The pipeline contract is:
  raw_path  -> npz or csv on disk produced by `get_data`
  loader    -> returns ndarray of shape [T, d], dtype float32

Standard split semantics (warmup, train, val, test) are documented under
`slice_time_series`. We deliberately keep the slicing model-agnostic so the
same helpers can serve future autoregressive models beyond Adaptive NVAR.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# On-disk formats and exchange representation
# ---------------------------------------------------------------------------

# We store raw time-series after `get_data` as a single .npz with keys:
#   data: float32 array of shape [T, d]
#   meta: serialized JSON of provenance fields (source, original_shape, etc.)
# A .npz is used (instead of .npy) so we can ship metadata alongside the array
# without inventing a sidecar file. Existing CheMLFlow nodes for the
# `timeseries` pipeline_type read this file directly; tabular nodes never
# touch it.

RAW_TS_KEY = "data"
RAW_TS_META_KEY = "meta"


@dataclass(frozen=True)
class TimeSeriesSplitConfig:
    """Plain dataclass holding the four contiguous segment lengths."""

    warmup_len: int
    train_len: int
    val_len: int
    test_len: int

    def total(self) -> int:
        return self.warmup_len + self.train_len + self.val_len + self.test_len


@dataclass(frozen=True)
class TimeSeriesSplit:
    """Materialized contiguous splits over a [T, d] array."""

    warmup: np.ndarray  # [warmup_len, d]
    train: np.ndarray   # [train_len, d]
    val: np.ndarray     # [val_len, d]
    test: np.ndarray    # [test_len, d]

    @property
    def d(self) -> int:
        return int(self.train.shape[1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _orient_to_time_first(array: np.ndarray, time_axis: str) -> np.ndarray:
    """Return [T, d] given a 1-D or 2-D array and a time_axis hint."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(
            f"Time-series array must be 1-D or 2-D, got shape {arr.shape}."
        )

    axis = str(time_axis or "auto").strip().lower()
    if axis == "rows":
        oriented = arr
    elif axis == "cols":
        oriented = arr.T
    elif axis == "auto":
        # Longer axis is treated as time. Equal lengths -> assume rows.
        oriented = arr if arr.shape[0] >= arr.shape[1] else arr.T
        LOGGER.info(
            "time_axis=auto resolved to %s for shape %s",
            "rows" if oriented is arr else "cols",
            arr.shape,
        )
    else:
        raise ValueError(
            f"Unsupported time_axis={time_axis!r}; expected 'rows', 'cols', or 'auto'."
        )

    return oriented.astype(np.float32, copy=False)


def load_npy_timeseries(path: str, time_axis: str = "auto") -> np.ndarray:
    """Load a .npy file and return a [T, d] float32 array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Time-series .npy not found: {path}")
    arr = np.load(path, allow_pickle=False)
    return _orient_to_time_first(arr, time_axis)


def load_csv_timeseries(
    path: str,
    *,
    has_header: bool = True,
    time_column: Optional[str | int] = None,
) -> np.ndarray:
    """Load a CSV time-series and return a [T, d] float32 array.

    `time_column` is dropped if provided. We do not interpret it (the rollout
    is index-based), but we accept it so users can keep timestamped CSVs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Time-series CSV not found: {path}")

    # Use pandas only for parsing; the rest of the pipeline stays on numpy.
    import pandas as pd

    df = pd.read_csv(path, header=0 if has_header else None)
    if time_column is not None:
        if isinstance(time_column, int):
            df = df.drop(df.columns[time_column], axis=1)
        else:
            if time_column not in df.columns:
                raise ValueError(
                    f"time_column={time_column!r} not in CSV columns: {list(df.columns)}"
                )
            df = df.drop(columns=[time_column])

    arr = df.to_numpy(dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def save_raw_timeseries(
    output_path: str,
    array: np.ndarray,
    *,
    source_meta: Optional[dict] = None,
) -> None:
    """Write the canonical .npz that downstream timeseries nodes read.

    We write through an open file handle to bypass numpy's automatic
    `.npz` extension appending — the caller may want to keep the existing
    `.csv` filename used by other CheMLFlow paths.
    """
    import json

    if array.ndim != 2:
        raise ValueError(f"Expected 2-D [T, d] array, got shape {array.shape}.")
    meta_str = json.dumps(source_meta or {}, sort_keys=True)
    with open(output_path, "wb") as fh:
        np.savez(
            fh,
            **{
                RAW_TS_KEY: array.astype(np.float32, copy=False),
                RAW_TS_META_KEY: np.array(meta_str),
            },
        )


def load_raw_timeseries(path: str) -> Tuple[np.ndarray, dict]:
    """Read the canonical raw .npz back. Returns (data[T,d], meta).

    Tolerates a `.npz`-suffixed sibling in case an older save path produced
    one (numpy auto-appends `.npz` when given a string output without that
    extension; the current writer avoids it, but legacy artifacts may exist).
    """
    import json

    candidate = path
    if not os.path.exists(candidate):
        if os.path.exists(candidate + ".npz"):
            candidate = candidate + ".npz"
        else:
            raise FileNotFoundError(f"Raw time-series file not found: {path}")
    with np.load(candidate, allow_pickle=False) as bundle:
        if RAW_TS_KEY not in bundle.files:
            raise ValueError(
                f"{candidate} is not a CheMLFlow time-series file; missing key {RAW_TS_KEY!r}."
            )
        data = np.asarray(bundle[RAW_TS_KEY], dtype=np.float32)
        meta_str = str(bundle[RAW_TS_META_KEY]) if RAW_TS_META_KEY in bundle.files else "{}"
    try:
        meta = json.loads(meta_str)
    except json.JSONDecodeError:
        meta = {}
    return data, meta


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def slice_time_series(
    data: np.ndarray, split_cfg: TimeSeriesSplitConfig
) -> TimeSeriesSplit:
    """Slice a [T, d] series into (warmup, train, val, test) contiguous spans.

    No shuffling — order is preserved. Caller is responsible for ensuring the
    underlying array is the *clean* signal; noise injection happens inside
    the trainer so it can be configured per trial.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected [T, d] array, got shape {data.shape}.")
    T = int(data.shape[0])
    needed = split_cfg.total()
    if T < needed:
        raise ValueError(
            f"Time series has {T} steps but split requires {needed} "
            f"(warmup={split_cfg.warmup_len}, train={split_cfg.train_len}, "
            f"val={split_cfg.val_len}, test={split_cfg.test_len})."
        )

    a = split_cfg.warmup_len
    b = a + split_cfg.train_len
    c = b + split_cfg.val_len
    d_end = c + split_cfg.test_len
    return TimeSeriesSplit(
        warmup=np.asarray(data[:a], dtype=np.float32),
        train=np.asarray(data[a:b], dtype=np.float32),
        val=np.asarray(data[b:c], dtype=np.float32),
        test=np.asarray(data[c:d_end], dtype=np.float32),
    )


def parse_split_config(raw: Optional[dict]) -> TimeSeriesSplitConfig:
    """Read a `split` block in YAML and return a TimeSeriesSplitConfig.

    Keys: warmup_len, train_len, val_len, test_len (all required, non-negative).
    """
    raw = raw or {}
    required = ("warmup_len", "train_len", "val_len", "test_len")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"timeseries split is missing required keys: {missing}. "
            f"Got: {sorted(raw.keys())}"
        )
    values = {}
    for key in required:
        try:
            values[key] = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeseries split.{key} must be an int") from exc
        if values[key] < 0:
            raise ValueError(f"timeseries split.{key} must be >= 0")
    if values["train_len"] == 0:
        raise ValueError("timeseries split.train_len must be > 0")
    if values["val_len"] == 0 and values["test_len"] == 0:
        raise ValueError(
            "timeseries split must have val_len > 0 or test_len > 0; "
            "without an evaluation segment there is nothing to score."
        )
    return TimeSeriesSplitConfig(**values)
