import json
import logging
import hashlib
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    MurckoScaffold = None


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _scaffold_smiles(smiles: str) -> Optional[str]:
    if Chem is None or MurckoScaffold is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _find_smiles_column(columns: Iterable[str]) -> Optional[str]:
    candidates = ["canonical_smiles", "smiles", "SMILES", "Smiles", "Drug", "drug"]
    for name in candidates:
        if name in columns:
            return name
    return None


def _map_split_indices(
    split_smiles: Iterable[str],
    canonical_to_indices: Dict[str, List[int]],
) -> List[int]:
    indices: List[int] = []
    missing = 0
    for smi in split_smiles:
        canonical = _canonicalize_smiles(str(smi))
        if canonical is None:
            missing += 1
            continue
        bucket = canonical_to_indices.get(canonical)
        if not bucket:
            missing += 1
            continue
        indices.append(bucket.pop(0))
    if missing:
        logging.warning("Split mapping skipped %s rows that were not found in curated data.", missing)
    return indices


def _build_canonical_index(smiles_list: Iterable[str]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_list):
        canonical = _canonicalize_smiles(str(smi))
        if canonical is None:
            continue
        index.setdefault(canonical, []).append(i)
    return index


def _json_dump(payload: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_index_list(indices: Iterable[int]) -> List[int]:
    out = sorted({int(i) for i in indices})
    return out


def _validate_kfold_args(n_samples: int, n_splits: int, repeats: int) -> None:
    if n_samples <= 1:
        raise ValueError("k-fold requires at least 2 samples.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits!r}")
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits={n_splits} cannot exceed n_samples={n_samples}."
        )
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1; got {repeats!r}")


def compute_dataset_fingerprint(
    curated_df,
    target_column: Optional[str] = None,
) -> str:
    """Create a deterministic dataset fingerprint for split plan provenance."""
    hasher = hashlib.sha256()

    if "canonical_smiles" in curated_df.columns:
        smiles_series = curated_df["canonical_smiles"].fillna("").astype(str)
    else:
        smiles_series = curated_df.index.astype(str)

    if target_column and target_column in curated_df.columns:
        target_series = curated_df[target_column].fillna("").astype(str)
    else:
        target_series = ["" for _ in range(len(curated_df))]

    for smi, tgt in zip(smiles_series.tolist(), list(target_series)):
        canonical = _canonicalize_smiles(str(smi)) or str(smi)
        hasher.update(canonical.encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(str(tgt).encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def save_split_plan(plan: Dict[str, object], output_path: str) -> None:
    # Atomic write to avoid truncated/corrupt JSON under concurrent Slurm jobs.
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_split_plan_",
        suffix=".json",
        dir=output_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_split_plan(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def random_kfold_plan(
    n_samples: int,
    n_splits: int,
    repeats: int,
    random_state: int,
    stratify: Optional[np.ndarray] = None,
    dataset_fingerprint: Optional[str] = None,
) -> Dict[str, object]:
    _validate_kfold_args(n_samples=n_samples, n_splits=n_splits, repeats=repeats)
    indices = np.arange(n_samples)
    repeats_payload: List[Dict[str, object]] = []

    for repeat_index in range(repeats):
        seed = int(random_state) + int(repeat_index)
        if stratify is not None:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(indices, stratify)
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(indices)

        folds: List[Dict[str, object]] = []
        for fold_index, (_, test_pos) in enumerate(split_iter):
            folds.append(
                {
                    "fold_index": int(fold_index),
                    "test_indices": _normalize_index_list(indices[test_pos].tolist()),
                }
            )
        repeats_payload.append(
            {
                "repeat_index": int(repeat_index),
                "random_state": int(seed),
                "folds": folds,
            }
        )

    return {
        "version": 1,
        "plan_type": "kfold",
        "strategy": "random",
        "n_rows": int(n_samples),
        "n_splits": int(n_splits),
        "repeats": int(repeats),
        "random_state": int(random_state),
        "dataset_fingerprint": dataset_fingerprint,
        "repeat_plans": repeats_payload,
    }


def scaffold_kfold_plan(
    smiles_list: Iterable[str],
    n_splits: int,
    repeats: int,
    random_state: int,
    dataset_fingerprint: Optional[str] = None,
) -> Dict[str, object]:
    if Chem is None or MurckoScaffold is None:
        raise RuntimeError("RDKit is required for scaffold k-fold splitting.")

    smiles = list(smiles_list)
    _validate_kfold_args(n_samples=len(smiles), n_splits=n_splits, repeats=repeats)
    scaffold_groups: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        scaffold = _scaffold_smiles(str(smi))
        scaffold_groups.setdefault(scaffold or "", []).append(idx)
    n_groups = len(scaffold_groups)
    if n_splits > n_groups:
        raise ValueError(
            f"scaffold_kfold requires n_splits <= unique scaffolds; got n_splits={n_splits} "
            f"but unique_scaffolds={n_groups}."
        )
    group_items = list(scaffold_groups.items())
    repeats_payload: List[Dict[str, object]] = []

    for repeat_index in range(repeats):
        seed = int(random_state) + int(repeat_index)
        rng = np.random.RandomState(seed)
        shuffled = list(group_items)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda item: len(item[1]), reverse=True)

        fold_bins: List[List[int]] = [[] for _ in range(n_splits)]
        fold_sizes = [0 for _ in range(n_splits)]
        for _, grp_indices in shuffled:
            smallest = int(np.argmin(fold_sizes))
            fold_bins[smallest].extend(grp_indices)
            fold_sizes[smallest] += len(grp_indices)

        folds: List[Dict[str, object]] = []
        for fold_index, test_indices in enumerate(fold_bins):
            folds.append(
                {
                    "fold_index": int(fold_index),
                    "test_indices": _normalize_index_list(test_indices),
                }
            )
        repeats_payload.append(
            {
                "repeat_index": int(repeat_index),
                "random_state": int(seed),
                "folds": folds,
            }
        )

    return {
        "version": 1,
        "plan_type": "kfold",
        "strategy": "scaffold",
        "n_rows": int(len(smiles)),
        "n_splits": int(n_splits),
        "repeats": int(repeats),
        "random_state": int(random_state),
        "dataset_fingerprint": dataset_fingerprint,
        "repeat_plans": repeats_payload,
    }


def random_split_indices(
    n_samples: int,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, List[int]]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}")
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1); got {val_size!r}")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    indices = np.arange(n_samples)
    if stratify is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_pos, test_pos = next(splitter.split(indices, stratify))
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_pos, test_pos = next(splitter.split(indices))
    train_idx = indices[train_pos]
    test_idx = indices[test_pos]
    if val_size > 0:
        val_fraction = val_size / (1 - test_size)
        if not 0 < val_fraction < 1:
            raise ValueError(
                f"Derived val_fraction={val_fraction!r} from val_size/test_size is invalid."
            )
        if stratify is not None:
            strat_train = stratify[train_idx]
            splitter_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            train_pos2, val_pos = next(splitter_val.split(train_idx, strat_train))
        else:
            splitter_val = ShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            train_pos2, val_pos = next(splitter_val.split(train_idx))
        val_idx = train_idx[val_pos]
        train_idx = train_idx[train_pos2]
    else:
        val_idx = np.array([], dtype=int)
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def scaffold_split_indices(
    smiles_list: Iterable[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, List[int]]:
    if Chem is None or MurckoScaffold is None:
        raise RuntimeError("RDKit is required for scaffold splitting.")

    smiles = list(smiles_list)
    scaffold_groups: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        scaffold = _scaffold_smiles(str(smi))
        if scaffold is None:
            scaffold = ""
        scaffold_groups.setdefault(scaffold, []).append(idx)

    rng = np.random.RandomState(random_state)
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_samples = len(smiles)
    if n_samples <= 1:
        raise ValueError("Scaffold split requires at least 2 samples.")
    if test_size < 0 or val_size < 0:
        raise ValueError("test_size and val_size must be non-negative.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")
    n_test = int(round(test_size * n_samples))
    n_val = int(round(val_size * n_samples))
    if test_size > 0 and n_test == 0:
        n_test = 1
    if val_size > 0 and n_val == 0:
        n_val = 1
    if n_test + n_val >= n_samples:
        raise ValueError(
            f"Requested split sizes leave no training data for n_samples={n_samples}: "
            f"n_test={n_test}, n_val={n_val}."
        )

    train_groups: List[List[int]] = []
    val_groups: List[List[int]] = []
    test_groups: List[List[int]] = []
    train_count = 0
    val_count = 0
    test_count = 0

    for group in groups:
        if test_count + len(group) <= n_test:
            test_groups.append(group)
            test_count += len(group)
        elif val_count + len(group) <= n_val:
            val_groups.append(group)
            val_count += len(group)
        else:
            train_groups.append(group)
            train_count += len(group)

    train_idx = [idx for group in train_groups for idx in group]
    val_idx = [idx for group in val_groups for idx in group]
    test_idx = [idx for group in test_groups for idx in group]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def tdc_split_indices(
    group: str,
    name: str,
    strategy: str,
    curated_smiles: Iterable[str],
) -> Dict[str, List[int]]:
    if group.upper() != "ADME":
        raise ValueError(f"Unsupported TDC group: {group}")
    from tdc.single_pred import ADME

    desired = "scaffold" if strategy.endswith("scaffold") else "random"
    data = ADME(name=name)
    split = data.get_split(method=desired)
    if not isinstance(split, dict):
        raise ValueError(f"TDC get_split(method={desired!r}) returned an invalid payload.")

    split_keys_lower = {str(k).lower() for k in split.keys()}
    if {"train", "test"}.issubset(split_keys_lower):
        selected = split
    else:
        strategy_key = None
        for key in split.keys():
            if desired in str(key).lower():
                strategy_key = key
                break
        if strategy_key is None:
            raise ValueError(
                f"TDC split does not expose a '{desired}' key. Available: {list(split.keys())}"
            )
        selected = split[strategy_key]
        if not isinstance(selected, dict):
            raise ValueError(
                f"TDC split key {strategy_key!r} does not contain split tables."
            )

    canonical_map = _build_canonical_index(curated_smiles)
    split_dict: Dict[str, List[int]] = {}
    for split_name, df in selected.items():
        smiles_col = _find_smiles_column(df.columns)
        if smiles_col is None:
            raise ValueError("TDC split frame missing SMILES column.")
        split_dict[split_name] = _map_split_indices(df[smiles_col].astype(str).tolist(), canonical_map)
    return split_dict


def save_split_indices(split_indices: Dict[str, List[int]], output_path: str) -> None:
    _json_dump(split_indices, output_path)


def build_split_indices(
    strategy: str,
    curated_df,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_column: Optional[str] = None,
    tdc_group: Optional[str] = None,
    tdc_name: Optional[str] = None,
) -> Dict[str, List[int]]:
    if strategy.startswith("tdc"):
        if not tdc_group or not tdc_name:
            raise ValueError("TDC split requires tdc_group and tdc_name.")
        if "canonical_smiles" not in curated_df.columns:
            raise ValueError("Curated data must include canonical_smiles for TDC split mapping.")
        return tdc_split_indices(tdc_group, tdc_name, strategy, curated_df["canonical_smiles"])

    stratify = None
    if stratify_column and stratify_column in curated_df.columns:
        stratify = curated_df[stratify_column].values

    if strategy == "scaffold":
        if "canonical_smiles" not in curated_df.columns:
            raise ValueError("Curated data must include canonical_smiles for scaffold splitting.")
        return scaffold_split_indices(
            curated_df["canonical_smiles"].astype(str).tolist(),
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    return random_split_indices(
        n_samples=len(curated_df),
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
