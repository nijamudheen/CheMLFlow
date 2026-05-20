from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utilities.config_validation import validate_config_strict


CLASSIFICATION_ONLY_MODELS = {"catboost_classifier"}
DL_PREFIX = "dl_"
_DEDUPE_STRATEGY_ALIASES = {
    "keep_first": "first",
    "keep_last": "last",
}
_VALID_DEDUPE_STRATEGIES = {"first", "last", "drop_conflicts", "majority"}
_SMILES_COLUMN_CANDIDATES = (
    "canonical_smiles",
    "smiles",
    "SMILES",
    "Smiles",
    "Drug",
    "drug",
)
_DEFAULT_MAX_EXPANDED_CASES = 10000
_DOE_SPEC_NAMESPACE_LEN = 8
_FEATURE_INPUT_ALIASES = {
    "use.curated_features": "featurize.none",
}
_CHEMPROP_LIKE_MODELS = {"chemprop", "chemeleon"}
_CHEMPROP_LIKE_FEATURE_INPUTS = {"smiles_native"}
_EXECUTION_ONLY_AXES = {
    "split.cv.fold_index",
    "split.cv.repeat_index",
    "split.inner.fold_index",
    "split.inner.repeat_index",
}
_LABEL_IC50_REQUIRED_COLUMNS = ("standard_value",)
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DOEIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    task_type: str
    train_node: str
    default_source: str
    allowed_feature_inputs: tuple[str, ...]
    allowed_models: tuple[str, ...]
    supports_label_normalize: bool
    supports_label_ic50: bool
    supports_split: bool
    supports_analyze: bool
    supports_explain: bool

    def allows_model(self, model_type: str) -> bool:
        allowed = set(self.allowed_models)
        if model_type.startswith(DL_PREFIX):
            # Two ways to allow a dl_* model:
            #   1. The profile lists it by exact name (e.g. "dl_adaptive_nvar"
            #      for ts_forecast), restricting which DL models are valid.
            #   2. The profile uses the wildcard "dl_*", accepting any DL model
            #      (the tabular regression/classification profiles do this).
            return model_type in allowed or f"{DL_PREFIX}*" in allowed
        return model_type in allowed

    def allows_feature_input(self, feature_input: str) -> bool:
        return feature_input in set(self.allowed_feature_inputs)


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "reg_local_csv": ProfileSpec(
        name="reg_local_csv",
        task_type="regression",
        train_node="train",
        default_source="local_csv",
        allowed_feature_inputs=("none", "smiles_native", "featurize.none", "featurize.rdkit", "featurize.morgan"),
        allowed_models=("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "chemprop", "chemeleon", "dl_*"),
        supports_label_normalize=False,
        supports_label_ic50=False,
        supports_split=True,
        supports_analyze=False,
        supports_explain=True,
    ),
    "reg_local_csv_ic50": ProfileSpec(
        name="reg_local_csv_ic50",
        task_type="regression",
        train_node="train",
        default_source="local_csv",
        allowed_feature_inputs=("none", "smiles_native", "featurize.none", "featurize.rdkit", "featurize.morgan"),
        allowed_models=("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "chemprop", "chemeleon", "dl_*"),
        supports_label_normalize=False,
        supports_label_ic50=True,
        supports_split=True,
        supports_analyze=False,
        supports_explain=True,
    ),
    "reg_chembl_ic50": ProfileSpec(
        name="reg_chembl_ic50",
        task_type="regression",
        train_node="train",
        default_source="chembl",
        allowed_feature_inputs=("featurize.rdkit",),
        allowed_models=("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "dl_*"),
        supports_label_normalize=False,
        supports_label_ic50=True,
        supports_split=True,
        supports_analyze=True,
        supports_explain=True,
    ),
    "clf_local_csv": ProfileSpec(
        name="clf_local_csv",
        task_type="classification",
        train_node="train",
        default_source="local_csv",
        allowed_feature_inputs=(
            "none",
            "smiles_native",
            "featurize.none",
            "featurize.rdkit",
            "featurize.morgan",
        ),
        allowed_models=(
            "random_forest",
            "decision_tree",
            "xgboost",
            "svm",
            "ensemble",
            "catboost_classifier",
            "chemprop",
            "chemeleon",
            "dl_*",
        ),
        supports_label_normalize=True,
        supports_label_ic50=False,
        supports_split=True,
        supports_analyze=False,
        supports_explain=True,
    ),
    "clf_tdc_benchmark": ProfileSpec(
        name="clf_tdc_benchmark",
        task_type="classification",
        train_node="train.tdc",
        default_source="tdc",
        allowed_feature_inputs=("none",),
        allowed_models=("catboost_classifier",),
        supports_label_normalize=False,
        supports_label_ic50=False,
        supports_split=False,
        supports_analyze=False,
        supports_explain=False,
    ),
    "ts_forecast": ProfileSpec(
        name="ts_forecast",
        task_type="regression",
        train_node="train.timeseries",
        default_source="local_npy",
        # Time-series pipelines bypass tabular feature_input entirely; the
        # profile pins it to "none" so DOE generation does not attempt to
        # vary featurize.* axes that do not exist on this branch.
        allowed_feature_inputs=("none",),
        allowed_models=("dl_adaptive_nvar", "dl_connectome_nvar"),
        supports_label_normalize=False,
        supports_label_ic50=False,
        supports_split=False,
        supports_analyze=False,
        supports_explain=False,
    ),
}


class DOEGenerationError(ValueError):
    pass


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_task_type(task_type: Any) -> str:
    return str(task_type or "").strip().lower()


def _flatten_dict(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, dotted))
        else:
            out[dotted] = value
    return out


def _validate_search_space_axes(search_space_flat: dict[str, Any]) -> None:
    invalid_axes = sorted(set(search_space_flat) & _EXECUTION_ONLY_AXES)
    if not invalid_axes:
        return
    raise DOEGenerationError(
        "Execution-only split axes must not be placed in search_space. "
        "Use defaults for an explicit retry/debug slice, or omit them so DOE can expand "
        f"the folds/repeats automatically. Invalid axes: {', '.join(invalid_axes)}"
    )


def _set_dotted(container: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        return
    current = container
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _get_dotted(container: dict[str, Any], dotted: str, default: Any = None) -> Any:
    parts = [part for part in dotted.split(".") if part]
    current: Any = container
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _pop_dotted(container: dict[str, Any], dotted: str) -> None:
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        return
    parents: list[tuple[dict[str, Any], str]] = []
    current: Any = container
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        parents.append((current, part))
        current = current[part]
    if not isinstance(current, dict):
        return
    current.pop(parts[-1], None)
    while parents:
        parent, key = parents.pop()
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break


def _extract_prefixed(flat_map: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat_map.items():
        if key.startswith(prefix):
            rel = key[len(prefix) :]
            if rel:
                _set_dotted(out, rel, value)
    return out


def _sanitize_token(value: Any) -> str:
    raw = str(value)
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def _normalize_feature_input(value: str) -> str:
    normalized = str(value or "").strip()
    return _FEATURE_INPUT_ALIASES.get(normalized, normalized)


def _is_binary_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        observed_numeric = {float(v) for v in numeric.unique().tolist()}
        if observed_numeric and observed_numeric.issubset({0.0, 1.0}):
            return True

    observed = {
        str(value).strip().lower()
        for value in non_null.tolist()
        if str(value).strip() != ""
    }
    if not observed:
        return False
    allowed = {
        "0",
        "1",
        "false",
        "true",
        "inactive",
        "active",
        "no",
        "yes",
        "n",
        "y",
    }
    return observed.issubset(allowed)


def _stable_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _hashable_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    global_cfg = payload.get("global")
    if isinstance(global_cfg, dict):
        global_cfg.pop("base_dir", None)
        global_cfg.pop("run_dir", None)
        global_cfg.pop("artifact_retention", None)
        runs_cfg = global_cfg.get("runs")
        if isinstance(runs_cfg, dict):
            runs_cfg.pop("id", None)
    return payload


def _scientific_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = _hashable_config_payload(config)
    split_mode = str(_get_dotted(payload, "split.mode", "")).strip().lower()
    if split_mode == "cv":
        _pop_dotted(payload, "split.cv.fold_index")
        _pop_dotted(payload, "split.cv.repeat_index")
    elif split_mode == "nested_holdout_cv":
        _pop_dotted(payload, "split.inner.fold_index")
        _pop_dotted(payload, "split.inner.repeat_index")
    return payload


def _case_scoped_path(path_template: str, case_id: str, namespace: str | None = None) -> str:
    template = str(path_template).strip()
    if not template:
        return os.path.join(namespace, case_id) if namespace else case_id

    rendered = template
    if namespace and "{doe_spec_hash}" in rendered:
        rendered = rendered.replace("{doe_spec_hash}", namespace)
    if "{case_id}" in rendered:
        scoped_case = os.path.join(namespace, case_id) if namespace and "{doe_spec_hash}" not in template else case_id
        return rendered.replace("{case_id}", scoped_case)

    if namespace:
        return os.path.join(rendered, namespace, case_id)
    return os.path.join(rendered, case_id)


def _apply_case_isolation(
    config: dict[str, Any],
    case_id: str,
    output_dir: str,
    spec_namespace: str | None = None,
) -> None:
    global_cfg = config.setdefault("global", {})
    if not isinstance(global_cfg, dict):
        return

    base_root = str(global_cfg.get("base_dir", os.path.join("data", "doe"))).strip() or os.path.join(
        "data", "doe"
    )
    global_cfg["base_dir"] = _case_scoped_path(base_root, case_id, namespace=spec_namespace)

    run_root = str(global_cfg.get("run_dir", os.path.join(output_dir, "runs"))).strip() or os.path.join(
        output_dir, "runs"
    )
    global_cfg["run_dir"] = _case_scoped_path(run_root, case_id, namespace=spec_namespace)

    runs_cfg = global_cfg.get("runs")
    if not isinstance(runs_cfg, dict):
        runs_cfg = {"enabled": _as_bool(runs_cfg)}
        global_cfg["runs"] = runs_cfg
    else:
        runs_cfg["enabled"] = _as_bool(runs_cfg.get("enabled", True))

    configured_id = str(runs_cfg.get("id", "")).strip()
    runs_cfg["id"] = configured_id.replace("{case_id}", case_id) if configured_id else case_id


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        stdout = result.stdout.rstrip("\n")
        return stdout or None
    except Exception:
        return None


def _git_sha() -> str | None:
    return _run_git_command(["rev-parse", "--short", "HEAD"])


def _capture_git_provenance(output_dir: str) -> dict[str, Any]:
    git_sha = _git_sha()
    status_short = _run_git_command(["status", "--short"]) or ""
    dirty = bool(status_short.strip())
    diff_snapshot_path: str | None = None
    diff_hash: str | None = None

    if dirty:
        tracked_diff = _run_git_command(["diff", "--no-ext-diff", "--binary", "HEAD"]) or ""
        snapshot_parts = ["# git status --short", status_short.rstrip()]
        if tracked_diff.strip():
            snapshot_parts.extend(
                [
                    "",
                    "# git diff --no-ext-diff --binary HEAD",
                    tracked_diff.rstrip(),
                ]
            )
        snapshot_text = "\n".join(snapshot_parts).rstrip() + "\n"
        diff_snapshot_path = os.path.join(output_dir, "git_worktree.patch")
        with open(diff_snapshot_path, "w", encoding="utf-8") as fh:
            fh.write(snapshot_text)
        diff_hash = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()

    return {
        "git_sha": git_sha,
        "git_dirty": dirty,
        "git_status_short": status_short.splitlines(),
        "git_diff_snapshot_path": diff_snapshot_path,
        "git_diff_hash": diff_hash,
    }


def load_doe_spec(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict):
        raise DOEGenerationError("DOE file must parse to a mapping/object.")
    version = int(spec.get("version", 1))
    if version != 1:
        raise DOEGenerationError(f"Unsupported DOE version={version}. Only version: 1 is supported.")
    if "dataset" not in spec:
        raise DOEGenerationError("DOE spec requires a top-level 'dataset' block.")
    if "search_space" not in spec:
        raise DOEGenerationError("DOE spec requires a top-level 'search_space' block.")
    if "output" not in spec:
        raise DOEGenerationError("DOE spec requires a top-level 'output' block.")
    return spec


def probe_dataset(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    source_cfg = dataset_cfg.get("source", {}) if isinstance(dataset_cfg, dict) else {}
    source_type = str(source_cfg.get("type", "local_csv")).strip().lower()
    target_column = dataset_cfg.get("target_column")
    probe: dict[str, Any] = {
        "source_type": source_type,
        "target_column_present": None,
        "target_unique": None,
        "binary_target": None,
        "columns": [],
        "n_rows": None,
        "resolved_smiles_column": None,
    }
    if source_type != "local_csv":
        return probe

    source_path = source_cfg.get("path")
    if not source_path:
        raise DOEGenerationError("dataset.source.path is required for source.type=local_csv.")
    path = Path(str(source_path))
    if not path.exists():
        raise DOEGenerationError(f"dataset.source.path does not exist: {source_path}")
    columns = [str(c) for c in pd.read_csv(path, nrows=0).columns]
    probe["columns"] = columns
    configured_smiles = str(dataset_cfg.get("smiles_column", "")).strip()
    if configured_smiles:
        probe["resolved_smiles_column"] = configured_smiles if configured_smiles in columns else None
    else:
        probe["resolved_smiles_column"] = next((c for c in _SMILES_COLUMN_CANDIDATES if c in columns), None)

    target_name = str(target_column) if target_column is not None else ""
    target_present = bool(target_name and target_name in columns)
    if target_column:
        probe["target_column_present"] = target_present

    if target_present:
        observed_text: set[str] = set()
        observed_numeric: set[float] = set()
        unique_values: set[str] = set()
        unique_overflow = False
        max_unique_tracked = 20000
        saw_non_null = False
        numeric_possible = True
        row_count = 0

        for chunk in pd.read_csv(path, usecols=[target_name], chunksize=50000):
            series = chunk[target_name]
            row_count += len(series)
            non_null = series.dropna()
            if non_null.empty:
                continue
            saw_non_null = True

            if not unique_overflow:
                for value in non_null.tolist():
                    unique_values.add(str(value))
                    if len(unique_values) > max_unique_tracked:
                        unique_overflow = True
                        break

            normalized = non_null.astype(str).str.strip().str.lower()
            observed_text.update(v for v in normalized.tolist() if v)

            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.isna().any():
                numeric_possible = False
            else:
                observed_numeric.update(float(v) for v in numeric.unique().tolist())

        probe["n_rows"] = int(row_count)
        if not unique_overflow:
            probe["target_unique"] = len(unique_values)
        else:
            probe["target_unique"] = None

        if saw_non_null:
            if numeric_possible and observed_numeric and observed_numeric.issubset({0.0, 1.0}):
                probe["binary_target"] = True
            else:
                allowed_text = {
                    "0",
                    "1",
                    "false",
                    "true",
                    "inactive",
                    "active",
                    "no",
                    "yes",
                    "n",
                    "y",
                }
                probe["binary_target"] = bool(observed_text) and observed_text.issubset(allowed_text)
        else:
            probe["binary_target"] = False
    else:
        if columns:
            count_col = columns[0]
            row_count = 0
            for chunk in pd.read_csv(path, usecols=[count_col], chunksize=50000):
                row_count += len(chunk)
            probe["n_rows"] = int(row_count)
        else:
            probe["n_rows"] = 0
    return probe


def _add_issue(issues: list[DOEIssue], code: str, path: str, message: str) -> None:
    issues.append(DOEIssue(code=code, path=path, message=message))


def _write_doe_snapshot(output_dir: str, spec: dict[str, Any], doe_path: str | None) -> tuple[str, str]:
    snapshot_path = os.path.join(output_dir, "doe_spec.input.yaml")
    snapshot_text = ""
    if doe_path:
        candidate = Path(doe_path)
        if candidate.exists():
            snapshot_text = candidate.read_text(encoding="utf-8")
    if not snapshot_text:
        snapshot_text = yaml.safe_dump(spec, sort_keys=False)
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        fh.write(snapshot_text)
    snapshot_hash = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
    return snapshot_path, snapshot_hash


def _resolve_task_type(dataset_cfg: dict[str, Any], probe: dict[str, Any]) -> str:
    requested = _normalize_task_type(dataset_cfg.get("task_type"))
    if requested in {"classification", "regression"}:
        return requested
    if requested != "auto":
        raise DOEGenerationError(
            "dataset.task_type must be one of: regression, classification, auto."
        )
    if not _as_bool(dataset_cfg.get("auto_confirmed", False)):
        raise DOEGenerationError(
            "dataset.task_type=auto requires dataset.auto_confirmed=true. "
            "Prefer explicit task_type for reproducible DOE definitions."
        )
    binary = probe.get("binary_target")
    if binary is None:
        raise DOEGenerationError(
            "Unable to infer task_type from dataset; provide explicit dataset.task_type."
        )
    return "classification" if bool(binary) else "regression"


def resolve_profile(dataset_cfg: dict[str, Any], probe: dict[str, Any]) -> ProfileSpec:
    explicit = dataset_cfg.get("profile")
    if explicit:
        key = str(explicit).strip()
        if key not in PROFILE_SPECS:
            raise DOEGenerationError(
                f"Unknown dataset.profile={key!r}. Supported: {sorted(PROFILE_SPECS)}"
            )
        profile = PROFILE_SPECS[key]
        task = _resolve_task_type(dataset_cfg, probe)
        if profile.task_type != task:
            raise DOEGenerationError(
                f"dataset.profile={profile.name!r} expects task_type={profile.task_type!r}, "
                f"but dataset.task_type resolved to {task!r}."
            )
        return profile

    task = _resolve_task_type(dataset_cfg, probe)
    source_type = str(probe.get("source_type", "local_csv")).strip().lower()
    if source_type in {"local_npy", "local_ts_csv"}:
        return PROFILE_SPECS["ts_forecast"]
    if source_type == "tdc":
        if task != "classification":
            raise DOEGenerationError("source.type=tdc currently supports classification benchmark profile only.")
        return PROFILE_SPECS["clf_tdc_benchmark"]
    if task == "regression" and source_type == "chembl":
        return PROFILE_SPECS["reg_chembl_ic50"]
    if task == "regression":
        return PROFILE_SPECS["reg_local_csv"]
    return PROFILE_SPECS["clf_local_csv"]


def _expand_search_space(search_space: dict[str, Any], max_cases: int | None) -> list[dict[str, Any]]:
    keys = sorted(search_space.keys())
    value_lists = []
    axis_sizes: list[int] = []
    for key in keys:
        values = _as_list(search_space.get(key))
        if not values:
            raise DOEGenerationError(f"search_space.{key} must contain at least one value.")
        value_lists.append(values)
        axis_sizes.append(len(values))

    estimated = int(math.prod(axis_sizes)) if axis_sizes else 0
    if max_cases is None and estimated > _DEFAULT_MAX_EXPANDED_CASES:
        raise DOEGenerationError(
            "Expanded DOE is too large without constraints.max_cases. "
            f"Estimated combinations={estimated:,} exceeds default safety limit={_DEFAULT_MAX_EXPANDED_CASES:,}. "
            "Set constraints.max_cases explicitly."
        )

    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        combos.append({key: value for key, value in zip(keys, values)})
        if max_cases is not None and len(combos) > max_cases:
            raise DOEGenerationError(
                f"Expanded DOE produced more than constraints.max_cases={max_cases} combinations."
            )
    return combos


def _execution_axis_values(axis_key: str, merged: dict[str, Any], declared_axes: set[str], upper_bound: int) -> list[int]:
    if axis_key in declared_axes:
        return [int(merged.get(axis_key, 0))]
    return list(range(max(int(upper_bound), 1)))


def _expand_execution_axes(merged: dict[str, Any], declared_axes: set[str]) -> list[dict[str, Any]]:
    split_mode = str(merged.get("split.mode", "holdout")).strip().lower() or "holdout"
    if split_mode == "cv":
        n_splits = int(merged.get("split.cv.n_splits", 5))
        repeats = int(merged.get("split.cv.repeats", 1))
        fold_values = _execution_axis_values("split.cv.fold_index", merged, declared_axes, upper_bound=n_splits)
        repeat_values = _execution_axis_values("split.cv.repeat_index", merged, declared_axes, upper_bound=repeats)
        return [
            {
                "split.cv.fold_index": int(fold_index),
                "split.cv.repeat_index": int(repeat_index),
            }
            for repeat_index, fold_index in itertools.product(repeat_values, fold_values)
        ]

    if split_mode == "nested_holdout_cv":
        stage = str(merged.get("split.stage", "inner")).strip().lower() or "inner"
        if stage != "inner":
            return [{}]
        n_splits = int(merged.get("split.inner.n_splits", 5))
        repeats = int(merged.get("split.inner.repeats", 1))
        fold_values = _execution_axis_values(
            "split.inner.fold_index",
            merged,
            declared_axes,
            upper_bound=n_splits,
        )
        repeat_values = _execution_axis_values(
            "split.inner.repeat_index",
            merged,
            declared_axes,
            upper_bound=repeats,
        )
        return [
            {
                "split.inner.fold_index": int(fold_index),
                "split.inner.repeat_index": int(repeat_index),
            }
            for repeat_index, fold_index in itertools.product(repeat_values, fold_values)
        ]

    return [{}]


def _execution_label(config: dict[str, Any]) -> str:
    split_mode = str(_get_dotted(config, "split.mode", "holdout")).strip().lower() or "holdout"
    if split_mode == "cv":
        repeat_index = int(_get_dotted(config, "split.cv.repeat_index", 0))
        fold_index = int(_get_dotted(config, "split.cv.fold_index", 0))
        return f"rep{repeat_index}_fold{fold_index}"
    if split_mode == "nested_holdout_cv":
        stage = str(_get_dotted(config, "split.stage", "inner")).strip().lower() or "inner"
        if stage == "inner":
            repeat_index = int(_get_dotted(config, "split.inner.repeat_index", 0))
            fold_index = int(_get_dotted(config, "split.inner.fold_index", 0))
            return f"{stage}_rep{repeat_index}_fold{fold_index}"
        return stage
    return split_mode


def _case_execution_tokens(config: dict[str, Any]) -> list[str]:
    split_mode = str(_get_dotted(config, "split.mode", "")).strip().lower()
    if split_mode == "cv":
        repeat_index = int(_get_dotted(config, "split.cv.repeat_index", 0))
        fold_index = int(_get_dotted(config, "split.cv.fold_index", 0))
        return [f"rep{repeat_index}", f"fold{fold_index}"]
    if split_mode == "nested_holdout_cv":
        stage = str(_get_dotted(config, "split.stage", "inner")).strip().lower() or "inner"
        if stage == "inner":
            repeat_index = int(_get_dotted(config, "split.inner.repeat_index", 0))
            fold_index = int(_get_dotted(config, "split.inner.fold_index", 0))
            return [stage, f"rep{repeat_index}", f"fold{fold_index}"]
        return [stage]
    return []


def _build_label_block(
    dataset_cfg: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    label_cfg = dataset_cfg.get("label", {}) if isinstance(dataset_cfg.get("label"), dict) else {}
    source_column = label_cfg.get("source_column") or dataset_cfg.get("label_source_column") or target_column
    label_map = dataset_cfg.get("label_map", {}) if isinstance(dataset_cfg.get("label_map"), dict) else {}
    positive = label_cfg.get("positive", label_map.get("positive"))
    negative = label_cfg.get("negative", label_map.get("negative"))
    drop_unmapped = label_cfg.get("drop_unmapped", True)
    return {
        "source_column": source_column,
        "target_column": target_column,
        "positive": _as_list(positive),
        "negative": _as_list(negative),
        "drop_unmapped": _as_bool(drop_unmapped),
    }


def _profile_default_feature_input(profile: ProfileSpec, model_type: str) -> str:
    if model_type in _CHEMPROP_LIKE_MODELS:
        if "smiles_native" in set(profile.allowed_feature_inputs):
            return "smiles_native"
        return "none"
    allowed_inputs = set(profile.allowed_feature_inputs)
    preferred = ("featurize.morgan", "featurize.rdkit", "featurize.none")
    for candidate in preferred:
        if candidate in allowed_inputs:
            return candidate
    for candidate in profile.allowed_feature_inputs:
        if candidate != "none":
            return candidate
    if profile.allowed_feature_inputs:
        return profile.allowed_feature_inputs[0]
    return "none"


def _build_pipeline_nodes(
    profile: ProfileSpec,
    feature_input: str,
    preprocess_enabled: bool,
    select_enabled: bool,
    explain_enabled: bool,
    label_normalize_enabled: bool,
    analyze_stats_enabled: bool,
    analyze_eda_enabled: bool,
) -> list[str]:
    if profile.train_node == "train.tdc":
        return ["train.tdc"]
    if profile.train_node == "train.timeseries":
        # Time-series pipelines bypass curate / featurize / split / preprocess.
        return ["get_data", "train.timeseries"]

    nodes: list[str] = ["get_data", "curate"]
    if profile.supports_label_ic50:
        nodes.append("label.ic50")
    elif label_normalize_enabled:
        nodes.append("label.normalize")

    if feature_input not in {"none", "smiles_native"}:
        nodes.append(feature_input)
    nodes.append("split")

    if analyze_stats_enabled:
        nodes.append("analyze.stats")
    if analyze_eda_enabled:
        nodes.append("analyze.eda")
    if preprocess_enabled:
        nodes.append("preprocess.features")
    if select_enabled:
        nodes.append("select.features")
    nodes.append("train")
    if explain_enabled:
        nodes.append("explain")
    return nodes


def _build_case_config(
    profile: ProfileSpec,
    dataset_cfg: dict[str, Any],
    merged: dict[str, Any],
    resolved_task: str,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resolved_task != profile.task_type:
        raise DOEGenerationError(
            f"Internal mismatch: profile={profile.name} task={profile.task_type} resolved_task={resolved_task}."
        )

    model_type = str(
        merged.get(
            "train.model.type",
            merged.get("train_tdc.model.type", profile.allowed_models[0]),
        )
    ).strip()
    if not model_type:
        raise DOEGenerationError("Each DOE case must resolve train.model.type (or train_tdc.model.type).")

    feature_input = _normalize_feature_input(
        str(merged.get("pipeline.feature_input", _profile_default_feature_input(profile, model_type))).strip()
    )
    if not feature_input:
        feature_input = "none"

    preprocess_enabled = _as_bool(merged.get("pipeline.preprocess", False))
    select_enabled = _as_bool(merged.get("pipeline.select", False))
    explain_enabled = _as_bool(merged.get("pipeline.explain", profile.supports_explain))
    label_normalize_enabled = _as_bool(
        merged.get(
            "pipeline.label_normalize",
            bool(dataset_cfg.get("label_map")) or bool(_get_dotted(dataset_cfg, "label.positive")),
        )
    )
    analyze_stats_enabled = _as_bool(merged.get("pipeline.analyze_stats", profile.supports_analyze))
    analyze_eda_enabled = _as_bool(merged.get("pipeline.analyze_eda", profile.supports_analyze))

    nodes = _build_pipeline_nodes(
        profile=profile,
        feature_input=feature_input,
        preprocess_enabled=preprocess_enabled,
        select_enabled=select_enabled,
        explain_enabled=explain_enabled,
        label_normalize_enabled=label_normalize_enabled and profile.supports_label_normalize,
        analyze_stats_enabled=analyze_stats_enabled and profile.supports_analyze,
        analyze_eda_enabled=analyze_eda_enabled and profile.supports_analyze,
    )

    pipeline_type = str(
        merged.get("global.pipeline_type", dataset_cfg.get("name", profile.name))
    ).strip()
    if not pipeline_type:
        pipeline_type = profile.name
    default_target_column = (
        "pIC50"
        if profile.supports_label_ic50
        else ("label" if resolved_task == "classification" else "target")
    )
    target_column = str(
        dataset_cfg.get("target_column", merged.get("global.target_column", default_target_column))
    ).strip()
    random_state = int(merged.get("global.random_state", 42))

    global_cfg: dict[str, Any] = {
        "pipeline_type": pipeline_type,
        "task_type": resolved_task,
        "base_dir": str(merged.get("global.base_dir", f"data/{pipeline_type}")),
        "target_column": target_column,
        "random_state": random_state,
        "thresholds": {
            "active": int(merged.get("global.thresholds.active", 1000)),
            "inactive": int(merged.get("global.thresholds.inactive", 10000)),
        },
        "runs": {"enabled": _as_bool(merged.get("global.runs.enabled", True))},
    }
    if "global.runs.id" in merged:
        global_cfg["runs"]["id"] = str(merged["global.runs.id"])
    if "global.run_dir" in merged:
        global_cfg["run_dir"] = str(merged["global.run_dir"])
    if "global.artifact_retention" in merged:
        artifact_retention = str(merged["global.artifact_retention"]).strip()
        if artifact_retention:
            global_cfg["artifact_retention"] = artifact_retention

    config: dict[str, Any] = {
        "global": global_cfg,
        "pipeline": {
            "nodes": nodes,
            "feature_input": feature_input,
        },
    }

    if "get_data" in nodes:
        source_cfg = dataset_cfg.get("source", {}) if isinstance(dataset_cfg.get("source"), dict) else {}
        source_type = str(source_cfg.get("type", profile.default_source)).strip().lower()
        get_data_cfg: dict[str, Any] = {"data_source": source_type, "source": {}}
        if source_type == "local_csv":
            get_data_cfg["source"]["path"] = str(source_cfg.get("path", ""))
        elif source_type == "chembl":
            get_data_cfg["source"]["target_name"] = str(source_cfg.get("target_name", ""))
            if source_cfg.get("target_chembl_id"):
                get_data_cfg["source"]["target_chembl_id"] = str(source_cfg.get("target_chembl_id"))
        elif source_type == "http_csv":
            get_data_cfg["source"]["url"] = str(source_cfg.get("url", ""))
        elif source_type == "tdc":
            get_data_cfg["source"]["group"] = str(source_cfg.get("group", "ADME"))
            if source_cfg.get("name"):
                get_data_cfg["source"]["name"] = str(source_cfg.get("name"))
        elif source_type == "local_npy":
            get_data_cfg["source"]["path"] = str(source_cfg.get("path", ""))
            if source_cfg.get("time_axis"):
                get_data_cfg["source"]["time_axis"] = str(source_cfg.get("time_axis"))
        elif source_type == "local_ts_csv":
            get_data_cfg["source"]["path"] = str(source_cfg.get("path", ""))
            if "has_header" in source_cfg:
                get_data_cfg["source"]["has_header"] = bool(source_cfg.get("has_header"))
            if source_cfg.get("time_column") is not None:
                get_data_cfg["source"]["time_column"] = source_cfg.get("time_column")
        if "get_data.max_rows" in merged:
            get_data_cfg["max_rows"] = int(merged["get_data.max_rows"])
        config["get_data"] = get_data_cfg

    if "curate" in nodes:
        curate_cfg: dict[str, Any] = {}
        dataset_curate = dataset_cfg.get("curate", {}) if isinstance(dataset_cfg.get("curate"), dict) else {}
        resolved_smiles_column = ""
        if isinstance(probe, dict):
            resolved_smiles = probe.get("resolved_smiles_column")
            if resolved_smiles is not None:
                resolved_smiles_column = str(resolved_smiles).strip()
        default_properties: Any = dataset_curate.get("properties")
        if default_properties is None:
            if profile.supports_label_ic50:
                default_properties = "standard_value"
            elif resolved_task == "classification":
                default_properties = str(
                    dataset_cfg.get("label_source_column", target_column)
                )
            else:
                default_properties = target_column
        curate_cfg["properties"] = merged.get("curate.properties", default_properties)
        smiles_column = merged.get(
            "curate.smiles_column",
            dataset_curate.get("smiles_column", dataset_cfg.get("smiles_column", resolved_smiles_column)),
        )
        if smiles_column:
            curate_cfg["smiles_column"] = str(smiles_column)
        dedupe = merged.get("curate.dedupe_strategy", dataset_curate.get("dedupe_strategy"))
        if dedupe:
            curate_cfg["dedupe_strategy"] = str(dedupe)
        label_column = merged.get(
            "curate.label_column",
            dataset_curate.get("label_column", dataset_cfg.get("label_source_column")),
        )
        if label_column:
            curate_cfg["label_column"] = str(label_column)
        if "curate.require_neutral_charge" in merged:
            curate_cfg["require_neutral_charge"] = _as_bool(merged["curate.require_neutral_charge"])
        if "curate.prefer_largest_fragment" in merged:
            curate_cfg["prefer_largest_fragment"] = _as_bool(merged["curate.prefer_largest_fragment"])
        if "curate.keep_all_columns" in merged:
            curate_cfg["keep_all_columns"] = _as_bool(merged["curate.keep_all_columns"])
        elif "keep_all_columns" in dataset_curate:
            curate_cfg["keep_all_columns"] = _as_bool(dataset_curate.get("keep_all_columns"))
        elif feature_input == "featurize.none":
            curate_cfg["keep_all_columns"] = True
        elif profile.supports_label_ic50:
            curate_cfg["keep_all_columns"] = True
        for bool_key in ("drop_missing_smiles", "drop_invalid_smiles", "drop_missing_target"):
            merged_key = f"curate.{bool_key}"
            if merged_key in merged:
                curate_cfg[bool_key] = _as_bool(merged[merged_key])
            elif bool_key in dataset_curate:
                curate_cfg[bool_key] = _as_bool(dataset_curate.get(bool_key))
        raw_required_non_null = merged.get(
            "curate.required_non_null_columns",
            dataset_curate.get("required_non_null_columns"),
        )
        required_non_null = _as_str_list(raw_required_non_null)
        if required_non_null:
            curate_cfg["required_non_null_columns"] = required_non_null
        row_filters = merged.get("curate.row_filters", dataset_curate.get("row_filters"))
        if isinstance(row_filters, dict) and row_filters:
            curate_cfg["row_filters"] = row_filters
        config["curate"] = curate_cfg

    if "label.normalize" in nodes:
        config["label"] = _build_label_block(dataset_cfg, target_column=target_column)

    if "split" in nodes:
        split_mode = str(merged.get("split.mode", "holdout")).strip().lower() or "holdout"
        split_strategy = str(merged.get("split.strategy", "random")).strip().lower() or "random"
        split_cfg: dict[str, Any] = {
            "mode": split_mode,
            "strategy": split_strategy,
            "random_state": int(merged.get("split.random_state", random_state)),
            "stratify": _as_bool(
                merged.get("split.stratify", resolved_task == "classification")
            ),
        }
        if "split.stratify_column" in merged:
            split_cfg["stratify_column"] = str(merged["split.stratify_column"])
        elif split_cfg["stratify"]:
            split_cfg["stratify_column"] = target_column
        if "split.require_disjoint" in merged:
            split_cfg["require_disjoint"] = _as_bool(merged["split.require_disjoint"])
        if "split.allow_missing_val" in merged:
            split_cfg["allow_missing_val"] = _as_bool(merged["split.allow_missing_val"])
        for coverage_key in (
            "split.min_coverage",
            "split.require_full_test_coverage",
            "split.min_test_coverage",
            "split.min_train_coverage",
            "split.min_val_coverage",
        ):
            if coverage_key in merged:
                _set_dotted(split_cfg, coverage_key.removeprefix("split."), merged[coverage_key])

        if split_mode == "holdout":
            split_cfg["test_size"] = float(merged.get("split.test_size", 0.2))
            split_cfg["val_size"] = float(merged.get("split.val_size", 0.1))
        elif split_mode == "cv":
            split_cfg["cv"] = {
                "n_splits": int(merged.get("split.cv.n_splits", 5)),
                "repeats": int(merged.get("split.cv.repeats", 1)),
                "fold_index": int(merged.get("split.cv.fold_index", 0)),
                "repeat_index": int(merged.get("split.cv.repeat_index", 0)),
                "random_state": int(merged.get("split.cv.random_state", split_cfg["random_state"])),
            }
            split_cfg["val_from_train"] = {
                "val_size": float(merged.get("split.val_from_train.val_size", merged.get("split.val_size", 0.1))),
                "stratify": _as_bool(
                    merged.get("split.val_from_train.stratify", split_cfg["stratify"])
                ),
            }
            if "split.val_from_train.random_state" in merged:
                split_cfg["val_from_train"]["random_state"] = int(merged["split.val_from_train.random_state"])
        elif split_mode == "nested_holdout_cv":
            split_cfg["stage"] = str(merged.get("split.stage", "inner")).strip().lower() or "inner"
            split_cfg["outer"] = {
                "test_size": float(merged.get("split.outer.test_size", merged.get("split.test_size", 0.2))),
                "random_state": int(merged.get("split.outer.random_state", split_cfg["random_state"])),
            }
            split_cfg["inner"] = {
                "n_splits": int(merged.get("split.inner.n_splits", 5)),
                "repeats": int(merged.get("split.inner.repeats", 1)),
                "fold_index": int(merged.get("split.inner.fold_index", 0)),
                "repeat_index": int(merged.get("split.inner.repeat_index", 0)),
                "random_state": int(merged.get("split.inner.random_state", split_cfg["random_state"])),
            }
            split_cfg["val_from_train"] = {
                "val_size": float(merged.get("split.val_from_train.val_size", merged.get("split.val_size", 0.1))),
                "stratify": _as_bool(
                    merged.get("split.val_from_train.stratify", split_cfg["stratify"])
                ),
            }
            if "split.val_from_train.random_state" in merged:
                split_cfg["val_from_train"]["random_state"] = int(merged["split.val_from_train.random_state"])
        config["split"] = split_cfg

    has_configurable_featurizer = any(
        node in {"featurize.morgan", "featurize.rdkit", "featurize.rdkit_labeled", "featurize.lipinski"}
        for node in nodes
    )
    if has_configurable_featurizer:
        featurize_cfg = _extract_prefixed(merged, "featurize.")
        if "radius" not in featurize_cfg and "featurize.morgan" in nodes:
            featurize_cfg["radius"] = 2
        if "n_bits" not in featurize_cfg and "featurize.morgan" in nodes:
            featurize_cfg["n_bits"] = 2048
        config["featurize"] = featurize_cfg

    if any(node in {"preprocess.features", "select.features"} for node in nodes):
        preprocess_cfg = _extract_prefixed(merged, "preprocess.")
        if preprocess_cfg:
            config["preprocess"] = preprocess_cfg

    if "train" in nodes or "train.timeseries" in nodes:
        model_cfg: dict[str, Any] = {"type": model_type}
        model_params = _extract_prefixed(merged, "train.model.params.")
        if model_params:
            model_cfg["params"] = model_params
        for passthrough in (
            "train.model.n_jobs",
            "train.model.foundation",
            "train.model.foundation_checkpoint",
            "train.model.freeze_encoder",
            "train.model.smiles_column",
            "train.model.allow_legacy_split_positions",
        ):
            if passthrough in merged:
                key = passthrough.removeprefix("train.model.")
                model_cfg[key] = merged[passthrough]
        if model_type == "chemeleon":
            model_cfg["foundation"] = "chemeleon"
        train_cfg: dict[str, Any] = {"model": model_cfg}
        if "train.random_state" in merged:
            train_cfg["random_state"] = int(merged["train.random_state"])
        tuning_cfg = _extract_prefixed(merged, "train.tuning.")
        if "method" not in tuning_cfg:
            tuning_cfg["method"] = "fixed"
        train_cfg["tuning"] = tuning_cfg
        reporting_cfg = _extract_prefixed(merged, "train.reporting.")
        if reporting_cfg:
            train_cfg["reporting"] = reporting_cfg
        # train.timeseries doesn't use early_stopping/features in the tabular sense,
        # so we only emit those blocks for the standard `train` node.
        if "train" in nodes:
            early_cfg = _extract_prefixed(merged, "train.early_stopping.")
            if "patience" not in early_cfg:
                early_cfg["patience"] = int(merged.get("train.early_stopping.patience", 20))
            train_cfg["early_stopping"] = early_cfg
            features_cfg = _extract_prefixed(merged, "train.features.")
            if (
                "label.normalize" in nodes
                and feature_input == "featurize.none"
                and isinstance(config.get("label"), dict)
            ):
                source_column = str(config["label"].get("source_column", "")).strip()
                if source_column:
                    exclude_columns = _as_str_list(features_cfg.get("exclude_columns"))
                    if source_column not in exclude_columns:
                        exclude_columns.append(source_column)
                    features_cfg["exclude_columns"] = exclude_columns
            if features_cfg:
                train_cfg["features"] = features_cfg
        config["train"] = train_cfg

    # train.timeseries uses a top-level `split:` block to specify warmup/train/
    # val/test segment lengths. These are *time-series* split lengths, not the
    # molecule-index splits that the standard `split` node produces.
    if "train.timeseries" in nodes:
        split_cfg: dict[str, Any] = {}
        for key in ("warmup_len", "train_len", "val_len", "test_len"):
            merged_key = f"split.{key}"
            if merged_key in merged:
                split_cfg[key] = int(merged[merged_key])
        # Pull defaults from the dataset block so a DOE doesn't have to repeat them.
        dataset_split = (
            dataset_cfg.get("split", {}) if isinstance(dataset_cfg.get("split"), dict) else {}
        )
        for key in ("warmup_len", "train_len", "val_len", "test_len"):
            if key not in split_cfg and key in dataset_split:
                split_cfg[key] = int(dataset_split[key])
        if split_cfg:
            config["split"] = split_cfg

    if "train.tdc" in nodes:
        train_tdc_cfg: dict[str, Any] = {
            "group": str(merged.get("train_tdc.group", _get_dotted(dataset_cfg, "source.group", "ADMET_Group"))),
            "benchmarks": _as_list(
                merged.get("train_tdc.benchmarks", [_get_dotted(dataset_cfg, "source.name", "Pgp_Broccatelli")])
            ),
            "split_type": str(merged.get("train_tdc.split_type", "default")),
            "seeds": [int(s) for s in _as_list(merged.get("train_tdc.seeds", [1, 2, 3, 4, 5]))],
            "model": {"type": str(merged.get("train_tdc.model.type", model_type or "catboost_classifier"))},
        }
        train_tdc_params = _extract_prefixed(merged, "train_tdc.model.params.")
        if train_tdc_params:
            train_tdc_cfg["model"]["params"] = train_tdc_params
        train_tdc_cfg["tuning"] = _extract_prefixed(merged, "train_tdc.tuning.")
        if "method" not in train_tdc_cfg["tuning"]:
            train_tdc_cfg["tuning"]["method"] = "fixed"
        train_tdc_cfg["early_stopping"] = _extract_prefixed(merged, "train_tdc.early_stopping.")
        if "patience" not in train_tdc_cfg["early_stopping"]:
            train_tdc_cfg["early_stopping"]["patience"] = int(
                merged.get("train_tdc.early_stopping.patience", 20)
            )
        train_tdc_cfg["featurize"] = _extract_prefixed(merged, "train_tdc.featurize.")
        if "radius" not in train_tdc_cfg["featurize"]:
            train_tdc_cfg["featurize"]["radius"] = int(merged.get("featurize.radius", 2))
        if "n_bits" not in train_tdc_cfg["featurize"]:
            train_tdc_cfg["featurize"]["n_bits"] = int(merged.get("featurize.n_bits", 2048))
        config["train_tdc"] = train_tdc_cfg

    return config


def _validate_case(
    profile: ProfileSpec,
    config: dict[str, Any],
    probe: dict[str, Any],
) -> list[DOEIssue]:
    issues: list[DOEIssue] = []
    nodes = list((config.get("pipeline") or {}).get("nodes") or [])
    task_type = str((config.get("global") or {}).get("task_type", profile.task_type)).strip().lower()

    has_train = "train" in nodes
    has_train_tdc = "train.tdc" in nodes
    has_train_ts = "train.timeseries" in nodes
    has_preprocess = "preprocess.features" in nodes
    has_select = "select.features" in nodes
    source_type = str(_get_dotted(config, "get_data.data_source", profile.default_source)).strip().lower()
    train_node_count = int(has_train) + int(has_train_tdc) + int(has_train_ts)
    if train_node_count > 1:
        _add_issue(
            issues,
            code="DOE_TRAIN_NODE_CONFLICT",
            path="pipeline.nodes",
            message="Use only one of train, train.tdc, or train.timeseries.",
        )
    if "explain" in nodes and not (has_train or has_train_tdc):
        _add_issue(
            issues,
            code="DOE_EXPLAIN_REQUIRES_TRAIN",
            path="pipeline.nodes",
            message="explain requires train or train.tdc.",
        )

    if "get_data" in nodes:
        if source_type and source_type != profile.default_source:
            _add_issue(
                issues,
                code="DOE_SOURCE_NOT_SUPPORTED_FOR_PROFILE",
                path="get_data.data_source",
                message=(
                    f"Profile {profile.name!r} expects get_data.data_source={profile.default_source!r}, "
                    f"but got {source_type!r}."
                ),
            )
        if source_type == "local_csv" and not str(_get_dotted(config, "get_data.source.path", "")).strip():
            _add_issue(
                issues,
                code="DOE_DATA_SOURCE_CONFIG_INVALID",
                path="get_data.source.path",
                message="local_csv source requires get_data.source.path.",
            )
        if source_type == "chembl" and not (
            str(_get_dotted(config, "get_data.source.target_name", "")).strip()
            or str(_get_dotted(config, "get_data.source.target_chembl_id", "")).strip()
        ):
            _add_issue(
                issues,
                code="DOE_DATA_SOURCE_CONFIG_INVALID",
                path="get_data.source",
                message="chembl source requires get_data.source.target_name or get_data.source.target_chembl_id.",
            )

    if "curate" in nodes:
        dedupe_strategy = str(_get_dotted(config, "curate.dedupe_strategy", "")).strip().lower()
        if dedupe_strategy:
            normalized = _DEDUPE_STRATEGY_ALIASES.get(dedupe_strategy, dedupe_strategy)
            if normalized not in _VALID_DEDUPE_STRATEGIES:
                allowed = ", ".join(
                    sorted(_VALID_DEDUPE_STRATEGIES | set(_DEDUPE_STRATEGY_ALIASES.keys()))
                )
                _add_issue(
                    issues,
                    code="DOE_CURATE_DEDUPE_INVALID",
                    path="curate.dedupe_strategy",
                    message=(
                        f"Unsupported curate.dedupe_strategy={dedupe_strategy!r}. "
                        f"Allowed values: {allowed}."
                    ),
                )

    model_type = ""
    if has_train or has_train_ts:
        model_type = str(_get_dotted(config, "train.model.type", "")).strip()
    elif has_train_tdc:
        model_type = str(_get_dotted(config, "train_tdc.model.type", "")).strip()
    if not model_type:
        _add_issue(
            issues,
            code="DOE_MISSING_MODEL_TYPE",
            path="train.model.type",
            message="Model type is required for DOE cases.",
        )
        return issues

    if not profile.allows_model(model_type):
        _add_issue(
            issues,
            code="DOE_MODEL_NOT_SUPPORTED_FOR_PROFILE",
            path="train.model.type",
            message=f"Model {model_type!r} is not supported for profile {profile.name!r}.",
        )

    if task_type == "regression" and model_type in CLASSIFICATION_ONLY_MODELS:
        _add_issue(
            issues,
            code="DOE_MODEL_TASK_MISMATCH",
            path="train.model.type",
            message=f"Model {model_type!r} is classification-only but task_type is regression.",
        )

    feature_nodes = {
        "use.curated_features",
        "featurize.none",
        "featurize.rdkit",
        "featurize.rdkit_labeled",
        "featurize.morgan",
    }
    preprocess_scaler = str(_get_dotted(config, "preprocess.scaler", "robust")).strip().lower() or "robust"
    selected_feature_input = _normalize_feature_input(
        str(_get_dotted(config, "pipeline.feature_input", "none")).strip()
    ) or "none"
    if "featurize.none" in nodes or "use.curated_features" in nodes:
        selected_feature_input = "featurize.none"
    elif "featurize.rdkit" in nodes or "featurize.rdkit_labeled" in nodes:
        selected_feature_input = "featurize.rdkit"
    elif "featurize.morgan" in nodes:
        selected_feature_input = "featurize.morgan"
    chemprop_like_passthrough_preprocess = (
        model_type in _CHEMPROP_LIKE_MODELS
        and has_preprocess
        and not has_select
        and preprocess_scaler == "none"
        and selected_feature_input == "smiles_native"
        and not any(node in feature_nodes for node in nodes)
    )
    if has_train and not profile.allows_feature_input(selected_feature_input):
        _add_issue(
            issues,
            code="DOE_FEATURE_INPUT_NOT_SUPPORTED",
            path="pipeline.nodes",
            message=f"Feature input {selected_feature_input!r} is not supported for profile {profile.name!r}.",
        )
    if has_train and model_type not in _CHEMPROP_LIKE_MODELS:
        if selected_feature_input == "smiles_native":
            _add_issue(
                issues,
                code="DOE_SMILES_NATIVE_MODEL_UNSUPPORTED",
                path="pipeline.feature_input",
                message=(
                    "smiles_native is reserved for SMILES-native models (chemprop/chemeleon). "
                    "Use featurize.rdkit, featurize.morgan, or featurize.none for tabular models."
                ),
            )
        elif not any(node in feature_nodes for node in nodes):
            _add_issue(
                issues,
                code="DOE_FEATURE_INPUT_REQUIRED",
                path="pipeline.nodes",
                message="Non-chemprop training requires one feature input node (featurize.none or featurize.*).",
            )
    if (
        (has_preprocess or has_select)
        and not any(node in feature_nodes for node in nodes)
        and not chemprop_like_passthrough_preprocess
        and selected_feature_input != "smiles_native"
    ):
        _add_issue(
            issues,
            code="DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS",
            path="pipeline.nodes",
            message="preprocess.features/select.features require an explicit feature input node.",
        )
    if model_type in _CHEMPROP_LIKE_MODELS and selected_feature_input not in _CHEMPROP_LIKE_FEATURE_INPUTS:
        _add_issue(
            issues,
                code="DOE_CHEMPROP_FEATURE_INPUT_UNSUPPORTED",
                path="pipeline.feature_input",
                message=(
                "chemprop and chemeleon are SMILES-native; set pipeline.feature_input to "
                "'smiles_native' for these rows."
                ),
            )
    if model_type in _CHEMPROP_LIKE_MODELS and (
        has_select or (has_preprocess and not chemprop_like_passthrough_preprocess)
    ):
        _add_issue(
            issues,
                code="DOE_CHEMPROP_PREPROCESS_UNSUPPORTED",
                path="pipeline.nodes",
                message=(
                "chemprop/chemeleon do not use tabular preprocess/select nodes; only the no-op combination "
                "(pipeline.feature_input=smiles_native and preprocess.scaler=none) is supported."
                ),
            )
    foundation_mode = str(_get_dotted(config, "train.model.foundation", "none")).strip().lower() or "none"
    if model_type == "chemeleon":
        foundation_mode = "chemeleon"
    if foundation_mode == "chemeleon":
        checkpoint = str(_get_dotted(config, "train.model.foundation_checkpoint", "")).strip()
        if not checkpoint:
            _add_issue(
                issues,
                code="DOE_CHEMELEON_CHECKPOINT_REQUIRED",
                path="train.model.foundation_checkpoint",
                message="CheMeleon runs require train.model.foundation_checkpoint to point to the .pt checkpoint.",
            )
    if profile.name == "reg_chembl_ic50" and "featurize.rdkit" not in nodes:
        _add_issue(
            issues,
            code="DOE_FEATURE_INPUT_NOT_SUPPORTED",
            path="pipeline.nodes",
            message="reg_chembl_ic50 profile currently requires featurize.rdkit for model input.",
        )

    if has_select and not has_preprocess:
        _add_issue(
            issues,
            code="DOE_SELECT_REQUIRES_PREPROCESS",
            path="pipeline.nodes",
            message="select.features requires preprocess.features in this pipeline.",
        )

    if "split" in nodes:
        split_mode = str(_get_dotted(config, "split.mode", "holdout")).strip().lower()
        split_strategy = str(_get_dotted(config, "split.strategy", "random")).strip().lower()
        if split_mode in {"cv", "nested_holdout_cv"} and split_strategy not in {"random", "scaffold"}:
            _add_issue(
                issues,
                code="DOE_SPLIT_STRATEGY_MODE_INVALID",
                path="split.strategy",
                message=f"split.mode={split_mode!r} only supports strategy=random|scaffold.",
            )
        if split_mode == "holdout" and split_strategy.startswith("tdc"):
            _add_issue(
                issues,
                code="DOE_SPLIT_STRATEGY_MODE_INVALID",
                path="split.strategy",
                message="TDC split strategies are not supported in DOE holdout mode for this profile.",
            )
        if split_mode == "holdout":
            test_size = float(_get_dotted(config, "split.test_size", 0.2) or 0.0)
            val_size = float(_get_dotted(config, "split.val_size", 0.1) or 0.0)
            if not (0.0 < test_size < 1.0):
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.test_size",
                    message="split.test_size must be > 0 and < 1 for holdout mode.",
                )
            if not (0.0 <= val_size < 1.0):
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.val_size",
                    message="split.val_size must be >= 0 and < 1 for holdout mode.",
                )
            if test_size + val_size >= 1.0:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split",
                    message="split.test_size + split.val_size must be < 1.",
                )
        if split_mode == "cv":
            n_splits = int(_get_dotted(config, "split.cv.n_splits", 5))
            repeats = int(_get_dotted(config, "split.cv.repeats", 1))
            fold_index = int(_get_dotted(config, "split.cv.fold_index", 0))
            repeat_index = int(_get_dotted(config, "split.cv.repeat_index", 0))
            if n_splits < 2:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.cv.n_splits",
                    message="split.cv.n_splits must be >= 2.",
                )
            if repeats < 1:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.cv.repeats",
                    message="split.cv.repeats must be >= 1.",
                )
            if fold_index < 0 or fold_index >= n_splits:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.cv.fold_index",
                    message="split.cv.fold_index must satisfy 0 <= fold_index < n_splits.",
                )
            if repeat_index < 0 or repeat_index >= repeats:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.cv.repeat_index",
                    message="split.cv.repeat_index must satisfy 0 <= repeat_index < repeats.",
                )
        if split_mode == "nested_holdout_cv":
            stage = str(_get_dotted(config, "split.stage", "inner")).strip().lower()
            if stage not in {"inner", "outer"}:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.stage",
                    message="split.stage must be either 'inner' or 'outer' for nested_holdout_cv mode.",
                )
            outer_test_size = float(_get_dotted(config, "split.outer.test_size", 0.2) or 0.0)
            if not (0.0 < outer_test_size < 1.0):
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.outer.test_size",
                    message="split.outer.test_size must be > 0 and < 1 for nested_holdout_cv mode.",
                )
            inner_splits = int(_get_dotted(config, "split.inner.n_splits", 5))
            inner_repeats = int(_get_dotted(config, "split.inner.repeats", 1))
            inner_fold = int(_get_dotted(config, "split.inner.fold_index", 0))
            inner_repeat = int(_get_dotted(config, "split.inner.repeat_index", 0))
            if inner_splits < 2:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.inner.n_splits",
                    message="split.inner.n_splits must be >= 2.",
                )
            if inner_repeats < 1:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.inner.repeats",
                    message="split.inner.repeats must be >= 1.",
                )
            if inner_fold < 0 or inner_fold >= inner_splits:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.inner.fold_index",
                    message="split.inner.fold_index must satisfy 0 <= fold_index < n_splits.",
                )
            if inner_repeat < 0 or inner_repeat >= inner_repeats:
                _add_issue(
                    issues,
                    code="DOE_SPLIT_PARAM_INVALID",
                    path="split.inner.repeat_index",
                    message="split.inner.repeat_index must satisfy 0 <= repeat_index < repeats.",
                )
        if model_type in _CHEMPROP_LIKE_MODELS or model_type.startswith(DL_PREFIX):
            if split_mode == "holdout":
                val_size = float(_get_dotted(config, "split.val_size", 0.0) or 0.0)
            else:
                val_size = float(_get_dotted(config, "split.val_from_train.val_size", 0.0) or 0.0)
            if val_size <= 0:
                _add_issue(
                    issues,
                    code="DOE_VALIDATION_SPLIT_REQUIRED",
                    path="split",
                    message=f"Model {model_type!r} requires an explicit validation split (val_size > 0).",
                )

    columns = set(str(col) for col in probe.get("columns", []))
    if columns:
        resolved_smiles_probe = probe.get("resolved_smiles_column")
        resolved_smiles_column = str(resolved_smiles_probe).strip() if resolved_smiles_probe else ""
        if task_type == "regression" and profile.default_source == "local_csv" and profile.supports_label_ic50:
            missing_label_source_columns = [
                col for col in _LABEL_IC50_REQUIRED_COLUMNS if col not in columns
            ]
            if missing_label_source_columns:
                _add_issue(
                    issues,
                    code="DOE_LABEL_IC50_SOURCE_COLUMNS_MISSING",
                    path="dataset.source.path",
                    message=(
                        "Regression local_csv_ic50 DOE requires raw IC50 source columns to exist in the dataset: "
                        + ", ".join(missing_label_source_columns)
                    ),
                )
            else:
                keep_all_columns = _as_bool(_get_dotted(config, "curate.keep_all_columns", False))
                curate_properties = set(_as_str_list(_get_dotted(config, "curate.properties", [])))
                preserves_label_source = keep_all_columns or all(
                    col in curate_properties for col in _LABEL_IC50_REQUIRED_COLUMNS
                )
                if not preserves_label_source:
                    _add_issue(
                        issues,
                        code="DOE_CURATE_LABEL_IC50_SOURCE_DROPPED",
                        path="curate.properties",
                        message=(
                            "Regression local_csv_ic50 DOE requires label.ic50 input columns to survive curate. "
                            "Configure curate.keep_all_columns=true or include "
                            + ", ".join(repr(col) for col in _LABEL_IC50_REQUIRED_COLUMNS)
                            + " in curate.properties."
                        ),
                    )
        if task_type == "regression" and profile.default_source == "local_csv" and not profile.supports_label_ic50:
            target_column = str(_get_dotted(config, "global.target_column", "")).strip()
            if not target_column or target_column not in columns:
                _add_issue(
                    issues,
                    code="DOE_TARGET_COLUMN_MISSING",
                    path="global.target_column",
                    message=(
                        f"Regression local_csv DOE requires global.target_column to exist in the dataset; "
                        f"got {target_column!r}."
                    ),
                )
            else:
                keep_all_columns = _as_bool(_get_dotted(config, "curate.keep_all_columns", False))
                uses_curated_features = "featurize.none" in nodes or "use.curated_features" in nodes
                label_column = str(_get_dotted(config, "curate.label_column", "")).strip()
                curate_properties = set(_as_str_list(_get_dotted(config, "curate.properties", [])))
                preserves_target = (
                    keep_all_columns
                    or uses_curated_features
                    or (label_column and target_column == label_column)
                    or target_column in curate_properties
                )
                if not preserves_target:
                    _add_issue(
                        issues,
                        code="DOE_CURATE_TARGET_DROPPED",
                        path="curate.properties",
                        message=(
                            "Regression DOE requires the target column to survive curate. "
                            f"Configure curate.keep_all_columns=true, curate.label_column={target_column!r}, "
                            f"or include {target_column!r} in curate.properties."
                        ),
                )
        smiles_column = str(_get_dotted(config, "curate.smiles_column", "")).strip()
        if smiles_column and smiles_column not in columns:
            _add_issue(
                issues,
                code="DOE_DATASET_COLUMN_MISSING",
                path="curate.smiles_column",
                message=f"Configured smiles column {smiles_column!r} was not found in the dataset.",
            )
        elif source_type == "local_csv" and not smiles_column:
            if not resolved_smiles_column:
                _add_issue(
                    issues,
                    code="DOE_SMILES_COLUMN_MISSING",
                    path="curate.smiles_column",
                    message=(
                        "No SMILES column could be resolved for local_csv data. "
                        "Set dataset.smiles_column/curate.smiles_column explicitly."
                    ),
                )
        required_non_null = _as_str_list(_get_dotted(config, "curate.required_non_null_columns", []))
        if required_non_null:
            smiles_for_normalization = smiles_column or resolved_smiles_column
            normalized_required: list[str] = []
            for col in required_non_null:
                normalized = str(col).strip()
                if not normalized:
                    continue
                if (
                    smiles_for_normalization
                    and smiles_for_normalization != "canonical_smiles"
                    and normalized == smiles_for_normalization
                ):
                    normalized = "canonical_smiles"
                if normalized not in normalized_required:
                    normalized_required.append(normalized)

            missing_required: list[str] = []
            for col in normalized_required:
                if col == "canonical_smiles":
                    if col in columns:
                        continue
                    if smiles_for_normalization and smiles_for_normalization in columns:
                        continue
                elif col in columns:
                    continue
                missing_required.append(col)
            if missing_required:
                _add_issue(
                    issues,
                    code="DOE_CURATE_REQUIRED_COLUMNS_MISSING",
                    path="curate.required_non_null_columns",
                    message=(
                        "curate.required_non_null_columns contains columns missing from dataset: "
                        + ", ".join(missing_required)
                    ),
                )
        row_filters = _as_dict(_get_dotted(config, "curate.row_filters", {}))
        if row_filters:
            missing_filter_columns: list[str] = []
            for raw_col in row_filters.keys():
                col = str(raw_col).strip()
                if not col:
                    continue
                normalized = col
                if (
                    smiles_column
                    and smiles_column != "canonical_smiles"
                    and normalized == smiles_column
                ):
                    normalized = "canonical_smiles"
                elif (
                    resolved_smiles_column
                    and resolved_smiles_column != "canonical_smiles"
                    and normalized == resolved_smiles_column
                ):
                    normalized = "canonical_smiles"
                if normalized not in columns and not (
                    normalized == "canonical_smiles"
                    and (
                        "canonical_smiles" in columns
                        or (smiles_column and smiles_column in columns)
                        or (resolved_smiles_column and resolved_smiles_column in columns)
                    )
                ):
                    missing_filter_columns.append(normalized)
            if missing_filter_columns:
                _add_issue(
                    issues,
                    code="DOE_CURATE_ROW_FILTER_COLUMNS_MISSING",
                    path="curate.row_filters",
                    message=(
                        "curate.row_filters contains columns missing from dataset: "
                        + ", ".join(sorted(set(missing_filter_columns)))
                    ),
                )
        if task_type == "classification" and "label.normalize" in nodes:
            source_column = str(_get_dotted(config, "label.source_column", "")).strip()
            if source_column and source_column not in columns:
                _add_issue(
                    issues,
                    code="DOE_DATASET_COLUMN_MISSING",
                    path="label.source_column",
                    message=f"Configured label source column {source_column!r} was not found in the dataset.",
                )

    if task_type == "classification":
        has_label_node = "label.normalize" in nodes
        if has_label_node:
            positive = _as_list(_get_dotted(config, "label.positive", []))
            negative = _as_list(_get_dotted(config, "label.negative", []))
            source_column = _get_dotted(config, "label.source_column")
            if not source_column or not positive or not negative:
                _add_issue(
                    issues,
                    code="DOE_LABEL_MAPPING_REQUIRED",
                    path="label",
                    message="label.normalize requires source_column plus positive/negative mappings.",
                )
        else:
            if not bool(probe.get("binary_target", False)):
                _add_issue(
                    issues,
                    code="DOE_LABEL_MAPPING_REQUIRED",
                    path="dataset",
                    message=(
                        "Classification DOE without label.normalize requires already-binary target labels. "
                        "Provide dataset.label_map or enable pipeline.label_normalize."
                    ),
                )

    if has_train_tdc:
        train_tdc_model = str(_get_dotted(config, "train_tdc.model.type", "")).strip()
        if train_tdc_model != "catboost_classifier":
            _add_issue(
                issues,
                code="DOE_TDC_MODEL_UNSUPPORTED",
                path="train_tdc.model.type",
                message="train.tdc currently supports only model.type=catboost_classifier.",
            )

    if not issues:
        try:
            from main import validate_pipeline_nodes

            validate_pipeline_nodes(nodes)
            validate_config_strict(config, nodes)
        except Exception as exc:  # pragma: no cover - runtime path exercised in tests
            _add_issue(
                issues,
                code="DOE_RUNTIME_SCHEMA_INVALID",
                path="pipeline/config",
                message=str(exc),
            )
    return issues


def generate_doe(spec: dict[str, Any], doe_path: str | None = None) -> dict[str, Any]:
    dataset_cfg = spec.get("dataset", {}) if isinstance(spec.get("dataset"), dict) else {}
    search_space = spec.get("search_space", {}) if isinstance(spec.get("search_space"), dict) else {}
    output_cfg = spec.get("output", {}) if isinstance(spec.get("output"), dict) else {}
    defaults_cfg = spec.get("defaults", {}) if isinstance(spec.get("defaults"), dict) else {}
    constraints_cfg = spec.get("constraints", {}) if isinstance(spec.get("constraints"), dict) else {}
    selection_cfg = spec.get("selection", {}) if isinstance(spec.get("selection"), dict) else {}

    output_dir = str(output_cfg.get("dir", "")).strip()
    if not output_dir:
        raise DOEGenerationError("output.dir is required.")
    os.makedirs(output_dir, exist_ok=True)
    doe_spec_snapshot_path, doe_spec_hash = _write_doe_snapshot(
        output_dir=output_dir,
        spec=spec,
        doe_path=doe_path,
    )
    spec_namespace = doe_spec_hash[:_DOE_SPEC_NAMESPACE_LEN]

    probe = probe_dataset(dataset_cfg)
    resolved_task = _resolve_task_type(dataset_cfg, probe)
    profile = resolve_profile(dataset_cfg, probe)
    if probe.get("source_type") == "local_csv" and probe.get("target_column_present") is False:
        label_cfg = dataset_cfg.get("label", {}) if isinstance(dataset_cfg.get("label"), dict) else {}
        label_map_cfg = dataset_cfg.get("label_map", {}) if isinstance(dataset_cfg.get("label_map"), dict) else {}
        has_label_mapping = bool(
            label_cfg.get("positive")
            and label_cfg.get("negative")
            or label_map_cfg.get("positive")
            and label_map_cfg.get("negative")
        )
        if (resolved_task != "classification" or not has_label_mapping) and not profile.supports_label_ic50:
            target_column = dataset_cfg.get("target_column")
            raise DOEGenerationError(
                f"dataset.target_column={target_column!r} was not found in dataset.source.path."
            )

    defaults_flat = _flatten_dict(defaults_cfg)
    search_space_flat = _flatten_dict(search_space)
    _validate_search_space_axes(search_space_flat)
    declared_axes = set(defaults_flat) | set(search_space_flat)
    max_cases = constraints_cfg.get("max_cases")
    if max_cases is not None:
        max_cases = int(max_cases)
    isolate_case_artifacts = _as_bool(constraints_cfg.get("isolate_case_artifacts", True))
    expanded = _expand_search_space(search_space_flat, max_cases=max_cases)

    expanded_case_total = 0
    for factors in expanded:
        merged_preview = dict(defaults_flat)
        merged_preview.update(factors)
        expanded_case_total += len(_expand_execution_axes(merged_preview, declared_axes))
        limit = max_cases if max_cases is not None else _DEFAULT_MAX_EXPANDED_CASES
        if expanded_case_total > limit:
            limit_name = "constraints.max_cases" if max_cases is not None else "default safety limit"
            raise DOEGenerationError(
                "Expanded DOE is too large after CV/nested execution-axis expansion. "
                f"Expanded cases={expanded_case_total:,} exceeds {limit_name}={limit:,}."
            )

    valid_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    seen_config_hashes: set[str] = set()
    seen_parent_scientific_ids: set[str] = set()
    case_index = 0
    parent_index = 0
    for factors in expanded:
        merged_base = dict(defaults_flat)
        merged_base.update(factors)
        execution_axes = _expand_execution_axes(merged_base, declared_axes)
        scientific_config = _build_case_config(
            profile=profile,
            dataset_cfg=dataset_cfg,
            merged=merged_base,
            resolved_task=resolved_task,
            probe=probe,
        )
        scientific_config_id = _stable_hash(_scientific_config_payload(scientific_config))
        if scientific_config_id in seen_parent_scientific_ids:
            continue
        seen_parent_scientific_ids.add(scientific_config_id)
        parent_index += 1
        parent_case_id = f"parent_{parent_index:04d}"
        parent_model_type = _get_dotted(
            scientific_config,
            "train.model.type",
            _get_dotted(scientific_config, "train_tdc.model.type", "model"),
        )
        parent_split_mode = _get_dotted(scientific_config, "split.mode", "nosplit")
        parent_split_strategy = _get_dotted(scientific_config, "split.strategy", "na")
        parent_record: dict[str, Any] = {
            "record_type": "parent",
            "case_id": parent_case_id,
            "parent_case_id": parent_case_id,
            "scientific_config_id": scientific_config_id,
            "profile": profile.name,
            "task_type": resolved_task,
            "status": "valid",
            "model_type": parent_model_type,
            "split_mode": parent_split_mode,
            "split_strategy": parent_split_strategy,
            "factors": factors,
            "issues": [],
            "execution_count": len(execution_axes),
            "valid_execution_cases": 0,
            "skipped_execution_cases": 0,
            "execution_case_ids": [],
            "valid_execution_case_ids": [],
            "execution_labels": [],
        }
        parent_issue_keys: set[str] = set()

        for execution_index, execution_factors in enumerate(execution_axes, start=1):
            case_index += 1
            case_id = f"case_{case_index:04d}"
            merged = dict(merged_base)
            merged.update(execution_factors)
            base_config = _build_case_config(
                profile=profile,
                dataset_cfg=dataset_cfg,
                merged=merged,
                resolved_task=resolved_task,
                probe=probe,
            )
            issues = _validate_case(profile=profile, config=base_config, probe=probe)

            record: dict[str, Any] = {
                "record_type": "execution_child",
                "case_id": case_id,
                "parent_case_id": parent_case_id,
                "scientific_config_id": scientific_config_id,
                "execution_index": execution_index,
                "execution_count": len(execution_axes),
                "profile": profile.name,
                "task_type": resolved_task,
                "status": "valid",
                "factors": factors,
                "execution_factors": execution_factors,
                "issues": [],
            }
            execution_label = _execution_label(base_config)
            record["execution_label"] = execution_label
            parent_record["execution_case_ids"].append(case_id)
            parent_record["execution_labels"].append(execution_label)

            if issues:
                record["status"] = "skipped"
                record["issues"] = [issue.__dict__ for issue in issues]
                parent_record["skipped_execution_cases"] += 1
                for issue in record["issues"]:
                    issue_key = json.dumps(issue, sort_keys=True)
                    if issue_key in parent_issue_keys:
                        continue
                    parent_issue_keys.add(issue_key)
                    parent_record["issues"].append(issue)
                all_records.append(record)
                continue

            config_fingerprint = _stable_hash(_hashable_config_payload(base_config))
            if config_fingerprint in seen_config_hashes:
                record["status"] = "skipped"
                record["issues"] = [
                    {
                        "code": "DOE_DUPLICATE_CONFIG",
                        "path": "search_space",
                        "message": "Different factor combinations rendered the same runtime config.",
                    }
                ]
                parent_record["skipped_execution_cases"] += 1
                issue_key = json.dumps(record["issues"][0], sort_keys=True)
                if issue_key not in parent_issue_keys:
                    parent_issue_keys.add(issue_key)
                    parent_record["issues"].append(record["issues"][0])
                all_records.append(record)
                continue
            seen_config_hashes.add(config_fingerprint)

            config = json.loads(json.dumps(base_config))
            if isolate_case_artifacts:
                _apply_case_isolation(
                    config,
                    case_id=case_id,
                    output_dir=output_dir,
                    spec_namespace=spec_namespace,
                )
            config_hash = _stable_hash(config)

            model_type = _get_dotted(config, "train.model.type", _get_dotted(config, "train_tdc.model.type", "model"))
            split_mode = _get_dotted(config, "split.mode", "nosplit")
            split_strategy = _get_dotted(config, "split.strategy", "na")
            execution_tokens = _case_execution_tokens(config)
            filename = (
                f"{case_id}__{_sanitize_token(profile.name)}__{_sanitize_token(model_type)}__"
                f"{_sanitize_token(split_mode)}__{_sanitize_token(split_strategy)}"
            )
            if execution_tokens:
                filename += "__" + "__".join(_sanitize_token(token) for token in execution_tokens)
            filename += ".yaml"
            config_path = os.path.join(output_dir, filename)
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, sort_keys=False)

            record["config_path"] = config_path
            record["config_hash"] = config_hash
            record["config_fingerprint"] = config_fingerprint
            parent_record["valid_execution_cases"] += 1
            parent_record["valid_execution_case_ids"].append(case_id)
            valid_records.append(record)
            all_records.append(record)

        if parent_record["valid_execution_cases"] == parent_record["execution_count"] and parent_record["execution_count"] > 0:
            parent_record["status"] = "valid"
        elif parent_record["valid_execution_cases"] > 0:
            parent_record["status"] = "partial"
        else:
            parent_record["status"] = "skipped"
        parent_records.append(parent_record)

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    parent_manifest_path = os.path.join(output_dir, "parent_manifest.jsonl")
    with open(parent_manifest_path, "w", encoding="utf-8") as fh:
        for record in parent_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    counts_by_issue: dict[str, int] = {}
    for record in all_records:
        for issue in record.get("issues", []):
            code = str(issue.get("code", "UNKNOWN"))
            counts_by_issue[code] = counts_by_issue.get(code, 0) + 1

    primary_metric = str(
        selection_cfg.get("primary_metric", "auc" if resolved_task == "classification" else "r2")
    ).strip().lower()
    if resolved_task == "classification" and primary_metric not in {"auc", "auprc", "accuracy", "f1"}:
        primary_metric = "auc"
    if resolved_task == "regression" and primary_metric not in {"r2", "mae"}:
        primary_metric = "r2"

    git_provenance = _capture_git_provenance(output_dir)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "doe_path": doe_path,
        "profile": profile.name,
        "task_type": resolved_task,
        "dataset_probe": probe,
        "total_cases": len(all_records),
        "valid_cases": len(valid_records),
        "skipped_cases": len(all_records) - len(valid_records),
        "total_parent_cases": len(parent_records),
        "valid_parent_cases": sum(1 for record in parent_records if str(record.get("status", "")).lower() == "valid"),
        "skipped_parent_cases": sum(1 for record in parent_records if str(record.get("status", "")).lower() == "skipped"),
        "partial_parent_cases": sum(1 for record in parent_records if str(record.get("status", "")).lower() == "partial"),
        "total_execution_cases": len(all_records),
        "valid_execution_cases": len(valid_records),
        "skipped_execution_cases": len(all_records) - len(valid_records),
        "issue_counts": counts_by_issue,
        "manifest_path": manifest_path,
        "parent_manifest_path": parent_manifest_path,
        "doe_spec_hash": doe_spec_hash,
        "doe_spec_snapshot_path": doe_spec_snapshot_path,
        **git_provenance,
        "selection": {
            "primary_metric": primary_metric,
            "optimize": str(selection_cfg.get("optimize", "max")).strip().lower() or "max",
        },
        "constraints": {
            "isolate_case_artifacts": isolate_case_artifacts,
        },
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return {
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "parent_manifest_path": parent_manifest_path,
        "valid_cases": valid_records,
        "all_cases": all_records,
        "parent_cases": parent_records,
        "summary": summary,
    }
