from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utilities.config_validation import validate_config_strict


REGRESSION_MODELS = {"random_forest", "svm", "decision_tree", "xgboost", "ensemble"}
CLASSIFICATION_MODELS = {"catboost_classifier", "chemprop"}
DL_PREFIX = "dl_"


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
        if model_type.startswith(DL_PREFIX):
            return any(entry == f"{DL_PREFIX}*" for entry in self.allowed_models)
        return model_type in set(self.allowed_models)

    def allows_feature_input(self, feature_input: str) -> bool:
        return feature_input in set(self.allowed_feature_inputs)


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "reg_local_csv": ProfileSpec(
        name="reg_local_csv",
        task_type="regression",
        train_node="train",
        default_source="local_csv",
        allowed_feature_inputs=("use.curated_features", "featurize.rdkit", "featurize.morgan"),
        allowed_models=("random_forest", "svm", "decision_tree", "xgboost", "ensemble", "dl_*"),
        supports_label_normalize=False,
        supports_label_ic50=False,
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
            "use.curated_features",
            "featurize.rdkit",
            "featurize.morgan",
        ),
        allowed_models=("catboost_classifier", "chemprop", "dl_*"),
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


def _is_binary_target(series: pd.Series) -> bool:
    observed = {
        str(value).strip().lower()
        for value in series.dropna().tolist()
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
        runs_cfg = global_cfg.get("runs")
        if isinstance(runs_cfg, dict):
            runs_cfg.pop("id", None)
    return payload


def _case_scoped_path(path_template: str, case_id: str) -> str:
    template = str(path_template).strip()
    if not template:
        return case_id
    if "{case_id}" in template:
        return template.replace("{case_id}", case_id)
    return os.path.join(template, case_id)


def _apply_case_isolation(config: dict[str, Any], case_id: str, output_dir: str) -> None:
    global_cfg = config.setdefault("global", {})
    if not isinstance(global_cfg, dict):
        return

    base_root = str(global_cfg.get("base_dir", os.path.join("data", "doe"))).strip() or os.path.join(
        "data", "doe"
    )
    global_cfg["base_dir"] = _case_scoped_path(base_root, case_id)

    run_root = str(global_cfg.get("run_dir", os.path.join(output_dir, "runs"))).strip() or os.path.join(
        output_dir, "runs"
    )
    global_cfg["run_dir"] = _case_scoped_path(run_root, case_id)

    runs_cfg = global_cfg.get("runs")
    if not isinstance(runs_cfg, dict):
        runs_cfg = {"enabled": _as_bool(runs_cfg)}
        global_cfg["runs"] = runs_cfg
    else:
        runs_cfg["enabled"] = _as_bool(runs_cfg.get("enabled", True))

    configured_id = str(runs_cfg.get("id", "")).strip()
    runs_cfg["id"] = configured_id.replace("{case_id}", case_id) if configured_id else case_id


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


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
    }
    if source_type != "local_csv":
        return probe

    source_path = source_cfg.get("path")
    if not source_path:
        raise DOEGenerationError("dataset.source.path is required for source.type=local_csv.")
    path = Path(str(source_path))
    if not path.exists():
        raise DOEGenerationError(f"dataset.source.path does not exist: {source_path}")
    df = pd.read_csv(path)
    probe["n_rows"] = int(len(df))
    probe["columns"] = [str(c) for c in df.columns]
    if target_column:
        present = str(target_column) in df.columns
        probe["target_column_present"] = bool(present)
        if present:
            target_series = df[str(target_column)]
            probe["target_unique"] = int(target_series.dropna().nunique())
            probe["binary_target"] = bool(_is_binary_target(target_series))
    return probe


def _add_issue(issues: list[DOEIssue], code: str, path: str, message: str) -> None:
    issues.append(DOEIssue(code=code, path=path, message=message))


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
    for key in keys:
        values = _as_list(search_space.get(key))
        if not values:
            raise DOEGenerationError(f"search_space.{key} must contain at least one value.")
        value_lists.append(values)
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        combos.append({key: value for key, value in zip(keys, values)})
        if max_cases is not None and len(combos) > max_cases:
            raise DOEGenerationError(
                f"Expanded DOE produced more than constraints.max_cases={max_cases} combinations."
            )
    return combos


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
    if profile.name == "clf_local_csv" and model_type == "chemprop":
        return "none"
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

    nodes: list[str] = ["get_data", "curate"]
    if profile.supports_label_ic50:
        nodes.extend(["featurize.lipinski", "label.ic50"])
    elif label_normalize_enabled:
        nodes.append("label.normalize")
    nodes.append("split")

    if analyze_stats_enabled:
        nodes.append("analyze.stats")
    if analyze_eda_enabled:
        nodes.append("analyze.eda")

    if feature_input != "none":
        nodes.append(feature_input)
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

    feature_input = str(
        merged.get("pipeline.feature_input", _profile_default_feature_input(profile, model_type))
    ).strip()
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
    target_column = str(
        dataset_cfg.get("target_column", merged.get("global.target_column", "label" if resolved_task == "classification" else "target"))
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

    config: dict[str, Any] = {
        "global": global_cfg,
        "pipeline": {"nodes": nodes},
    }

    if "get_data" in nodes:
        source_cfg = dataset_cfg.get("source", {}) if isinstance(dataset_cfg.get("source"), dict) else {}
        source_type = str(source_cfg.get("type", profile.default_source)).strip().lower()
        get_data_cfg: dict[str, Any] = {"data_source": source_type, "source": {}}
        if source_type == "local_csv":
            get_data_cfg["source"]["path"] = str(source_cfg.get("path", ""))
        elif source_type == "chembl":
            get_data_cfg["source"]["target_name"] = str(source_cfg.get("target_name", ""))
        elif source_type == "http_csv":
            get_data_cfg["source"]["url"] = str(source_cfg.get("url", ""))
        elif source_type == "tdc":
            get_data_cfg["source"]["group"] = str(source_cfg.get("group", "ADME"))
            if source_cfg.get("name"):
                get_data_cfg["source"]["name"] = str(source_cfg.get("name"))
        if "get_data.max_rows" in merged:
            get_data_cfg["max_rows"] = int(merged["get_data.max_rows"])
        config["get_data"] = get_data_cfg

    if "curate" in nodes:
        curate_cfg: dict[str, Any] = {}
        dataset_curate = dataset_cfg.get("curate", {}) if isinstance(dataset_cfg.get("curate"), dict) else {}
        default_properties: Any = dataset_curate.get("properties")
        if default_properties is None:
            if profile.name == "reg_chembl_ic50":
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
            dataset_curate.get("smiles_column", dataset_cfg.get("smiles_column")),
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
        elif profile.name == "reg_chembl_ic50":
            curate_cfg["keep_all_columns"] = True
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

    if any(node.startswith("featurize.") for node in nodes):
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

    if "train" in nodes:
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
        ):
            if passthrough in merged:
                key = passthrough.removeprefix("train.model.")
                model_cfg[key] = merged[passthrough]
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
        early_cfg = _extract_prefixed(merged, "train.early_stopping.")
        if "patience" not in early_cfg:
            early_cfg["patience"] = int(merged.get("train.early_stopping.patience", 20))
        train_cfg["early_stopping"] = early_cfg
        features_cfg = _extract_prefixed(merged, "train.features.")
        if (
            "label.normalize" in nodes
            and feature_input == "use.curated_features"
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
    has_preprocess = "preprocess.features" in nodes
    has_select = "select.features" in nodes
    if has_train and has_train_tdc:
        _add_issue(
            issues,
            code="DOE_TRAIN_NODE_CONFLICT",
            path="pipeline.nodes",
            message="Use either train or train.tdc, not both.",
        )
    if "explain" in nodes and not (has_train or has_train_tdc):
        _add_issue(
            issues,
            code="DOE_EXPLAIN_REQUIRES_TRAIN",
            path="pipeline.nodes",
            message="explain requires train or train.tdc.",
        )

    if "get_data" in nodes:
        source_type = str(_get_dotted(config, "get_data.data_source", "")).strip().lower()
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
        if source_type == "chembl" and not str(_get_dotted(config, "get_data.source.target_name", "")).strip():
            _add_issue(
                issues,
                code="DOE_DATA_SOURCE_CONFIG_INVALID",
                path="get_data.source.target_name",
                message="chembl source requires get_data.source.target_name.",
            )

    model_type = ""
    if has_train:
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

    if task_type == "classification" and model_type in REGRESSION_MODELS:
        _add_issue(
            issues,
            code="DOE_MODEL_TASK_MISMATCH",
            path="train.model.type",
            message=f"Model {model_type!r} is regression-only but task_type is classification.",
        )
    if task_type == "regression" and model_type in CLASSIFICATION_MODELS:
        _add_issue(
            issues,
            code="DOE_MODEL_TASK_MISMATCH",
            path="train.model.type",
            message=f"Model {model_type!r} is classification-only but task_type is regression.",
        )

    feature_nodes = {
        "use.curated_features",
        "featurize.rdkit",
        "featurize.rdkit_labeled",
        "featurize.morgan",
    }
    selected_feature_input = "none"
    for candidate in ("use.curated_features", "featurize.rdkit", "featurize.morgan"):
        if candidate in nodes:
            selected_feature_input = candidate
            break
    if has_train and not profile.allows_feature_input(selected_feature_input):
        _add_issue(
            issues,
            code="DOE_FEATURE_INPUT_NOT_SUPPORTED",
            path="pipeline.nodes",
            message=f"Feature input {selected_feature_input!r} is not supported for profile {profile.name!r}.",
        )
    if has_train and model_type != "chemprop":
        if not any(node in feature_nodes for node in nodes):
            _add_issue(
                issues,
                code="DOE_FEATURE_INPUT_REQUIRED",
                path="pipeline.nodes",
                message="Non-chemprop training requires one feature input node (use.curated_features or featurize.*).",
            )
    if (has_preprocess or has_select) and not any(node in feature_nodes for node in nodes):
        _add_issue(
            issues,
            code="DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS",
            path="pipeline.nodes",
            message="preprocess.features/select.features require an explicit feature input node.",
        )
    if model_type == "chemprop" and (has_preprocess or has_select):
        _add_issue(
            issues,
            code="DOE_CHEMPROP_PREPROCESS_UNSUPPORTED",
            path="pipeline.nodes",
            message="chemprop does not use tabular preprocess/select nodes; remove preprocess.features/select.features.",
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
        if model_type == "chemprop" or model_type.startswith(DL_PREFIX):
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
        smiles_column = str(_get_dotted(config, "curate.smiles_column", "")).strip()
        if smiles_column and smiles_column not in columns:
            _add_issue(
                issues,
                code="DOE_DATASET_COLUMN_MISSING",
                path="curate.smiles_column",
                message=f"Configured smiles column {smiles_column!r} was not found in the dataset.",
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

    probe = probe_dataset(dataset_cfg)
    resolved_task = _resolve_task_type(dataset_cfg, probe)
    if probe.get("source_type") == "local_csv" and probe.get("target_column_present") is False:
        label_cfg = dataset_cfg.get("label", {}) if isinstance(dataset_cfg.get("label"), dict) else {}
        label_map_cfg = dataset_cfg.get("label_map", {}) if isinstance(dataset_cfg.get("label_map"), dict) else {}
        has_label_mapping = bool(
            label_cfg.get("positive")
            and label_cfg.get("negative")
            or label_map_cfg.get("positive")
            and label_map_cfg.get("negative")
        )
        if resolved_task != "classification" or not has_label_mapping:
            target_column = dataset_cfg.get("target_column")
            raise DOEGenerationError(
                f"dataset.target_column={target_column!r} was not found in dataset.source.path."
            )
    profile = resolve_profile(dataset_cfg, probe)

    defaults_flat = _flatten_dict(defaults_cfg)
    search_space_flat = _flatten_dict(search_space)
    max_cases = constraints_cfg.get("max_cases")
    if max_cases is not None:
        max_cases = int(max_cases)
    isolate_case_artifacts = _as_bool(constraints_cfg.get("isolate_case_artifacts", True))
    expanded = _expand_search_space(search_space_flat, max_cases=max_cases)

    valid_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    seen_config_hashes: set[str] = set()

    for index, factors in enumerate(expanded, start=1):
        case_id = f"case_{index:04d}"
        merged = dict(defaults_flat)
        merged.update(factors)
        base_config = _build_case_config(
            profile=profile,
            dataset_cfg=dataset_cfg,
            merged=merged,
            resolved_task=resolved_task,
        )
        issues = _validate_case(profile=profile, config=base_config, probe=probe)

        record: dict[str, Any] = {
            "case_id": case_id,
            "profile": profile.name,
            "task_type": resolved_task,
            "status": "valid",
            "factors": factors,
            "issues": [],
        }

        if issues:
            record["status"] = "skipped"
            record["issues"] = [issue.__dict__ for issue in issues]
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
            all_records.append(record)
            continue
        seen_config_hashes.add(config_fingerprint)

        config = json.loads(json.dumps(base_config))
        if isolate_case_artifacts:
            _apply_case_isolation(config, case_id=case_id, output_dir=output_dir)
        config_hash = _stable_hash(config)

        model_type = _get_dotted(config, "train.model.type", _get_dotted(config, "train_tdc.model.type", "model"))
        split_mode = _get_dotted(config, "split.mode", "nosplit")
        split_strategy = _get_dotted(config, "split.strategy", "na")
        filename = (
            f"{case_id}__{_sanitize_token(profile.name)}__{_sanitize_token(model_type)}__"
            f"{_sanitize_token(split_mode)}__{_sanitize_token(split_strategy)}.yaml"
        )
        config_path = os.path.join(output_dir, filename)
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)

        record["config_path"] = config_path
        record["config_hash"] = config_hash
        record["config_fingerprint"] = config_fingerprint
        valid_records.append(record)
        all_records.append(record)

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for record in all_records:
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

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "doe_path": doe_path,
        "profile": profile.name,
        "task_type": resolved_task,
        "dataset_probe": probe,
        "total_cases": len(all_records),
        "valid_cases": len(valid_records),
        "skipped_cases": len(all_records) - len(valid_records),
        "issue_counts": counts_by_issue,
        "manifest_path": manifest_path,
        "git_sha": _git_sha(),
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
        "valid_cases": valid_records,
        "all_cases": all_records,
        "summary": summary,
    }
