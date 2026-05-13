# an example workflow that uses ChemBL bioactivty data
import csv
import glob
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import yaml
import pandas as pd
import joblib
import numpy as np

from contracts import (
    ANALYZE_EDA_INPUT_2CLASS_CONTRACT,
    ANALYZE_EDA_INPUT_3CLASS_CONTRACT,
    ANALYZE_EDA_GENERIC_INPUT_CONTRACT,
    ANALYZE_EDA_OUTPUT_CONTRACT,
    ANALYZE_STATS_INPUT_CONTRACT,
    ANALYZE_STATS_OUTPUT_CONTRACT,
    CURATE_INPUT_CONTRACT,
    CURATE_OUTPUT_CONTRACT,
    FEATURIZE_RDKIT_INPUT_CONTRACT,
    FEATURIZE_RDKIT_OUTPUT_CONTRACT,
    FEATURIZE_RDKIT_LABELED_INPUT_CONTRACT,
    FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT,
    FEATURIZE_MORGAN_INPUT_CONTRACT,
    FEATURIZE_MORGAN_OUTPUT_CONTRACT,
    PREPROCESS_FEATURES_INPUT_CONTRACT,
    PREPROCESS_FEATURES_OUTPUT_CONTRACT,
    PREPROCESS_ARTIFACTS_CONTRACT,
    SELECT_FEATURES_INPUT_FEATURES_CONTRACT,
    SELECT_FEATURES_OUTPUT_CONTRACT,
    SELECT_FEATURES_LIST_CONTRACT,
    EXPLAIN_INPUT_MODEL_CONTRACT,
    EXPLAIN_OUTPUT_CONTRACT,
    GET_DATA_INPUT_CONTRACT,
    GET_DATA_OUTPUT_CONTRACT,
    DESCRIPTORS_CONTRACT,
    IC50_INPUT_CONTRACT,
    LABEL_IC50_INPUT_CONTRACT,
    LABEL_IC50_OUTPUT_2CLASS_CONTRACT,
    LABEL_IC50_OUTPUT_3CLASS_CONTRACT,
    MODEL_LABELS_CONTRACT,
    PIC50_2CLASS_CONTRACT,
    PIC50_3CLASS_CONTRACT,
    PREPROCESSED_CONTRACT,
    TRAIN_INPUT_FEATURES_CONTRACT,
    TRAIN_INPUT_LABELS_CONTRACT,
    TRAIN_OUTPUT_CONTRACT,
    ContractSpec,
    bind_output_path,
    make_target_column_contract,
    validate_contract,
)
from MLModels import data_preprocessing
from MLModels import train_models
from utilities import splitters
from utilities.config_validation import validate_config_strict

_EXPLICIT_FEATURE_INPUT_NODES = {
    "featurize.none",
    "use.curated_features",
    "featurize.rdkit",
    "featurize.rdkit_labeled",
    "featurize.morgan",
}


def build_paths(base_dir: str) -> dict[str, str]:
    return {
        "raw": os.path.join(base_dir, "raw.csv"),
        "raw_sample": os.path.join(base_dir, "raw_sample.csv"),
        "preprocessed": os.path.join(base_dir, "preprocessed.csv"),
        "curated": os.path.join(base_dir, "curated.csv"),
        "curated_labeled": os.path.join(base_dir, "curated_labeled.csv"),
        "curated_smiles": os.path.join(base_dir, "curated_smiles.csv"),
        "pic50_3class": os.path.join(base_dir, "bioactivity_3class_pIC50.csv"),
        "pic50_2class": os.path.join(base_dir, "bioactivity_2class_pIC50.csv"),
        "rdkit_descriptors": os.path.join(base_dir, "rdkit_descriptors.csv"),
        "rdkit_labeled": os.path.join(base_dir, "rdkit_descriptors_labeled.csv"),
        "morgan_fingerprints": os.path.join(base_dir, "morgan_fingerprints.csv"),
        "morgan_labeled": os.path.join(base_dir, "morgan_fingerprints_labeled.csv"),
        "morgan_meta": os.path.join(base_dir, "morgan_meta.json"),
        "preprocessed_features": os.path.join(base_dir, "preprocessed_features.csv"),
        "preprocessed_labels": os.path.join(base_dir, "preprocessed_labels.csv"),
        "selected_features": os.path.join(base_dir, "selected_features.csv"),
        "selected_features_list": os.path.join(base_dir, "selected_features.txt"),
        "preprocess_artifacts": os.path.join(base_dir, "preprocess_artifacts.joblib"),
        "split_dir": os.path.join(base_dir, "splits"),
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def sample_csv(input_path: str, output_path: str, max_rows: int) -> None:
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for idx, row in enumerate(reader):
            writer.writerow(row)
            if idx >= max_rows:
                break


def resolve_run_dir(config: dict) -> str:
    global_config = config.get("global", {})
    run_dir = global_config.get("run_dir")
    if run_dir:
        return run_dir
    runs = global_config.get("runs", {})
    if runs.get("enabled"):
        run_id = runs.get("id")
        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join("runs", run_id)
    return "results"

def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_log_level(global_config: dict) -> int:
    configured = global_config.get("log_level")
    if configured is not None:
        if isinstance(configured, int):
            return configured
        if isinstance(configured, str):
            level = getattr(logging, configured.strip().upper(), None)
            if isinstance(level, int):
                return level
    if _as_bool(global_config.get("debug", False)):
        return logging.DEBUG
    return logging.INFO


def _resolve_seed(global_seed: int, override: object | None = None) -> int:
    if override is None:
        return int(global_seed)
    return int(override)


def _resolve_artifact_retention(global_config: dict) -> str:
    retention = str(
        global_config.get("artifact_retention", _ARTIFACT_RETENTION_FULL)
    ).strip().lower()
    if not retention:
        return _ARTIFACT_RETENTION_FULL
    if retention not in _SUPPORTED_ARTIFACT_RETENTION:
        supported = ", ".join(sorted(_SUPPORTED_ARTIFACT_RETENTION))
        raise ValueError(f"global.artifact_retention must be one of: {supported}")
    return retention


def _track_heavy_artifact(context: dict, path: str) -> None:
    if not isinstance(path, str):
        return
    clean = path.strip()
    if not clean:
        return
    heavy = context.setdefault("_heavy_artifacts", set())
    if not isinstance(heavy, set):
        return
    heavy.add(os.path.abspath(clean))


def _track_heavy_artifact_glob(context: dict, pattern: str) -> None:
    if not isinstance(pattern, str):
        return
    clean = pattern.strip()
    if not clean:
        return
    globs = context.setdefault("_heavy_artifact_globs", set())
    if not isinstance(globs, set):
        return
    globs.add(os.path.abspath(clean))


def _path_is_within(path: str, roots: list[str]) -> bool:
    abs_path = os.path.abspath(path)
    for root in roots:
        try:
            if os.path.commonpath([abs_path, root]) == root:
                return True
        except ValueError:
            continue
    return False


def _apply_artifact_retention(context: dict, *, run_status: str) -> None:
    retention = str(
        context.get("artifact_retention", _ARTIFACT_RETENTION_FULL)
    ).strip().lower()
    summary = {
        "retention": retention,
        "run_status": run_status,
        "candidate_count": 0,
        "deleted_count": 0,
        "skipped_preexisting_count": 0,
        "failed_delete_count": 0,
        "failed_deletes": [],
    }
    if retention != _ARTIFACT_RETENTION_AUDIT_LIGHT:
        context["artifact_retention_summary"] = summary
        return

    run_dir = str(context.get("run_dir", "")).strip()
    base_dir = str(context.get("base_dir", "")).strip()
    run_start_epoch: float | None
    try:
        run_start_epoch = float(context.get("run_start_epoch"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        run_start_epoch = None
    allowed_roots = [
        os.path.abspath(root)
        for root in (base_dir, run_dir)
        if isinstance(root, str) and root.strip()
    ]
    heavy_paths = context.get("_heavy_artifacts", set())
    heavy_globs = context.get("_heavy_artifact_globs", set())
    candidate_paths: set[str] = set()

    if isinstance(heavy_paths, set):
        candidate_paths.update(path for path in heavy_paths if isinstance(path, str) and path.strip())
    if isinstance(heavy_globs, set):
        for pattern in heavy_globs:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            candidate_paths.update(glob.glob(pattern, recursive=True))

    deleted_count = 0
    skipped_preexisting_count = 0
    failed_deletes: list[dict[str, str]] = []
    summary["candidate_count"] = len(candidate_paths)
    for path in sorted(candidate_paths):
        if not os.path.isfile(path):
            continue
        if allowed_roots and not _path_is_within(path, allowed_roots):
            failed_deletes.append({"path": path, "reason": "outside_allowed_roots"})
            continue
        if run_start_epoch is not None:
            try:
                if os.path.getmtime(path) + 1e-6 < run_start_epoch:
                    skipped_preexisting_count += 1
                    continue
            except OSError as exc:
                failed_deletes.append({"path": path, "reason": str(exc)})
                continue
        try:
            os.remove(path)
            deleted_count += 1
        except OSError as exc:
            failed_deletes.append({"path": path, "reason": str(exc)})

    summary["deleted_count"] = deleted_count
    summary["skipped_preexisting_count"] = skipped_preexisting_count
    summary["failed_delete_count"] = len(failed_deletes)
    summary["failed_deletes"] = failed_deletes
    logging.info(
        "Artifact retention '%s' applied after %s run: removed %d heavy artifact files.",
        retention,
        run_status,
        deleted_count,
    )
    for failure in failed_deletes:
        logging.warning(
            "artifact retention could not remove '%s': %s",
            failure["path"],
            failure["reason"],
        )
    context["artifact_retention_summary"] = summary


def _configure_logging(run_dir: str, level: int = logging.INFO) -> None:
    """Configure root logger to emit to stdout and to <run_dir>/run.log.

    This must be defined before any pipeline runner calls it.
    """
    log_path = os.path.join(run_dir, "run.log")
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _hashable_config_payload(config: dict) -> dict:
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


_PIPELINE_NODE_ALIASES = {
    "use.curated_features": "featurize.none",
}
_CHEMPROP_LIKE_MODELS = {"chemprop", "chemeleon"}
_ARTIFACT_RETENTION_FULL = "full"
_ARTIFACT_RETENTION_AUDIT_LIGHT = "audit_light"
_SUPPORTED_ARTIFACT_RETENTION = {
    _ARTIFACT_RETENTION_FULL,
    _ARTIFACT_RETENTION_AUDIT_LIGHT,
}


def _canonicalize_pipeline_nodes(nodes: list[str]) -> list[str]:
    return [_PIPELINE_NODE_ALIASES.get(str(node), str(node)) for node in nodes]


def _is_chemprop_like_model(model_type: object) -> bool:
    return str(model_type or "").strip().lower() in _CHEMPROP_LIKE_MODELS


def _normalize_runtime_config(config: dict, nodes: list[str]) -> dict:
    payload = json.loads(json.dumps(config))
    pipeline_cfg = payload.get("pipeline")
    if isinstance(pipeline_cfg, dict):
        pipeline_cfg["nodes"] = list(nodes)
    else:
        payload["pipeline"] = {"nodes": list(nodes)}
    return payload


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


def _write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(temp_path, path)


def _run_subprocess(cmd: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a child process and raise with captured output on failure."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        command = " ".join(str(part) for part in cmd)
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details: list[str] = [
            f"Command failed with exit code {result.returncode}: {command}",
        ]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n".join(details))
    return result


def run_node_get_data(context: dict) -> None:
    validate_contract(
        bind_output_path(GET_DATA_INPUT_CONTRACT, context["config_path"]),
        warn_only=False,
    )
    script_path = os.path.join("GetData", "get_data.py")
    output_file = context["paths"]["raw"]
    _run_subprocess([sys.executable, script_path, output_file, "--config", context["config_path"]])
    validate_contract(bind_output_path(GET_DATA_OUTPUT_CONTRACT, output_file), warn_only=True)


def run_node_curate(context: dict) -> None:
    validate_contract(
        bind_output_path(CURATE_INPUT_CONTRACT, context["paths"]["raw"]),
        warn_only=False,
    )
    raw_data_file = context["paths"]["raw"]
    pipeline_type = context["pipeline_type"]
    get_data_config = context.get("get_data_config", {})
    if pipeline_type == "qm9":
        max_rows = get_data_config.get("max_rows")
        if max_rows:
            sampled_path = context["paths"]["raw_sample"]
            sample_csv(raw_data_file, sampled_path, int(max_rows))
            raw_data_file = sampled_path

    curate_config = context["curate_config"]
    target_column = context["target_column"]
    properties_items = _as_str_list(curate_config.get("properties"))
    if not properties_items:
        if pipeline_type == "qm9" or context.get("task_type") == "classification":
            properties_items = _as_str_list(target_column)
        else:
            properties_items = _as_str_list("standard_value")
    properties = ",".join(properties_items)

    script_path = os.path.join("utilities", "prepareActivityData.py")
    preprocessed_file = context["paths"]["preprocessed"]
    curated_file = context["paths"]["curated"]
    curated_smiles_output = context["paths"]["curated_smiles"]
    cmd = [
        sys.executable,
        script_path,
        raw_data_file,
        preprocessed_file,
        curated_file,
        curated_smiles_output,
        "--active_threshold",
        str(context["active_threshold"]),
        "--inactive_threshold",
        str(context["inactive_threshold"]),
        "--properties",
        properties,
    ]
    smiles_column = curate_config.get("smiles_column")
    if smiles_column:
        cmd.extend(["--smiles_column", smiles_column])
    cmd.extend(["--target_column", str(target_column)])
    dedupe_strategy = curate_config.get("dedupe_strategy")
    if dedupe_strategy:
        cmd.extend(["--dedupe_strategy", dedupe_strategy])
    label_column = curate_config.get("label_column")
    if not label_column and context.get("task_type") == "classification":
        label_column = context["target_column"]
    if label_column:
        cmd.extend(["--label_column", label_column])
    if curate_config.get("require_neutral_charge"):
        cmd.append("--require_neutral_charge")
    if "prefer_largest_fragment" in curate_config:
        if curate_config.get("prefer_largest_fragment"):
            cmd.append("--prefer_largest_fragment")
        else:
            cmd.append("--no_prefer_largest_fragment")
    if context["keep_all_columns"]:
        cmd.append("--keep_all_columns")
    if "drop_missing_smiles" in curate_config:
        if _as_bool(curate_config.get("drop_missing_smiles", True)):
            cmd.append("--drop_missing_smiles")
        else:
            cmd.append("--no_drop_missing_smiles")
    if "drop_invalid_smiles" in curate_config:
        if _as_bool(curate_config.get("drop_invalid_smiles", True)):
            cmd.append("--drop_invalid_smiles")
        else:
            cmd.append("--no_drop_invalid_smiles")
    if "drop_missing_target" in curate_config:
        if _as_bool(curate_config.get("drop_missing_target", True)):
            cmd.append("--drop_missing_target")
        else:
            cmd.append("--no_drop_missing_target")
    required_non_null_columns = _as_str_list(curate_config.get("required_non_null_columns", []))
    if required_non_null_columns:
        cmd.extend(["--required_non_null_columns", ",".join(required_non_null_columns)])
    row_filters = curate_config.get("row_filters", {})
    if isinstance(row_filters, dict) and row_filters:
        cmd.extend(["--row_filters", json.dumps(row_filters, separators=(",", ":"))])
    _run_subprocess(cmd)
    validate_contract(bind_output_path(PREPROCESSED_CONTRACT, preprocessed_file), warn_only=True)
    validate_contract(bind_output_path(CURATE_OUTPUT_CONTRACT, curated_file), warn_only=True)
    # Establish the canonical dataset path for downstream nodes.
    context["curated_path"] = curated_file


def run_node_featurize_none(context: dict) -> None:
    curated_path = context.get("curated_path", context["paths"]["curated"])
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="featurize_none_input",
                target_column=context["target_column"],
                output_path=curated_path,
            ),
            curated_path,
        ),
        warn_only=False,
    )
    context["feature_matrix"] = curated_path
    context["labels_matrix"] = curated_path
    context["feature_method"] = "none"


def run_node_label_ic50(context: dict) -> None:
    input_file = context.get("curated_path", context["paths"]["curated"])
    validate_contract(
        bind_output_path(LABEL_IC50_INPUT_CONTRACT, input_file),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "IC50_pIC50.py")
    output_file_3class = context["paths"]["pic50_3class"]
    output_file_2class = context["paths"]["pic50_2class"]
    _run_subprocess(
        [sys.executable, script_path, input_file, output_file_3class, output_file_2class]
    )
    validate_contract(
        bind_output_path(LABEL_IC50_OUTPUT_3CLASS_CONTRACT, output_file_3class),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(LABEL_IC50_OUTPUT_2CLASS_CONTRACT, output_file_2class),
        warn_only=True,
    )
    context["labels_matrix"] = output_file_3class
    # Treat the labeled pIC50 output as the canonical dataset for downstream nodes.
    context["curated_path"] = output_file_3class


def run_node_analyze_stats(context: dict) -> None:
    validate_contract(
        bind_output_path(ANALYZE_STATS_INPUT_CONTRACT, context["paths"]["pic50_2class"]),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "stat_tests.py")
    input_file = context["paths"]["pic50_2class"]
    output_dir = context["base_dir"]
    descriptor = context["target_column"]
    tests = ["mannwhitney", "ttest"]
    try:
        stats_df = pd.read_csv(input_file)
        if descriptor in stats_df.columns and stats_df[descriptor].nunique(dropna=True) <= 20:
            tests.append("chi2")
        else:
            logging.info(
                "Skipping chi2 for descriptor '%s' due to high cardinality/continuous values.",
                descriptor,
            )
    except Exception as exc:
        logging.warning("Could not inspect stats input for chi2 guard (%s). Running without chi2.", exc)
    for test in tests:
        _run_subprocess([sys.executable, script_path, input_file, output_dir, test, descriptor])
    validate_contract(
        bind_output_path(ANALYZE_STATS_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_analyze_eda(context: dict) -> None:
    from utilities.generic_eda import run_generic_eda

    input_file = context.get("curated_path", context["paths"]["raw"])
    validate_contract(
        bind_output_path(ANALYZE_EDA_GENERIC_INPUT_CONTRACT, input_file),
        warn_only=False,
    )
    analyze_cfg = context.get("analyze_config", {}) or {}
    eda_cfg = analyze_cfg.get("eda", {}) if isinstance(analyze_cfg, dict) else {}
    output_dir = str((eda_cfg or {}).get("output_dir") or os.path.join(context["run_dir"], "eda"))
    run_generic_eda(
        input_path=input_file,
        output_dir=output_dir,
        task_type=context.get("task_type"),
        target_column=context.get("target_column"),
        config=eda_cfg,
    )
    validate_contract(
        bind_output_path(ANALYZE_EDA_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_featurize_rdkit(context: dict) -> None:
    validate_contract(
        bind_output_path(
            FEATURIZE_RDKIT_INPUT_CONTRACT,
            context.get("curated_path", context["paths"]["curated"]),
        ),
        warn_only=False,
    )
    # Use labeled descriptors so features and target live in a single canonical file.
    script_path = os.path.join("GenDescriptors", "RDKit_descriptors_labeled.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["rdkit_descriptors"]
    labeled_output_file = context["paths"]["rdkit_labeled"]
    _track_heavy_artifact(context, output_file)
    _track_heavy_artifact(context, labeled_output_file)
    _run_subprocess(
        [
            sys.executable,
            script_path,
            input_file,
            output_file,
            "--labeled-output-file",
            labeled_output_file,
            "--property-columns",
            context["target_column"],
        ]
    )
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT, labeled_output_file),
        warn_only=True,
    )
    context["feature_matrix"] = labeled_output_file
    context["labels_matrix"] = labeled_output_file
    context["feature_method"] = "rdkit"


def run_node_featurize_rdkit_labeled(context: dict) -> None:
    # Backward-compatible alias: use the same implementation as featurize.rdkit.
    run_node_featurize_rdkit(context)


def run_node_featurize_morgan(context: dict) -> None:
    validate_contract(
        bind_output_path(
            FEATURIZE_MORGAN_INPUT_CONTRACT,
            context.get("curated_path", context["paths"]["curated"]),
        ),
        warn_only=False,
    )
    script_path = os.path.join("GenDescriptors", "Morgan_fingerprints.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["morgan_fingerprints"]
    labeled_output = context["paths"]["morgan_labeled"]
    _track_heavy_artifact(context, output_file)
    _track_heavy_artifact(context, labeled_output)
    featurize_config = context.get("featurize_config", {})
    radius = featurize_config.get("radius", 2)
    n_bits = featurize_config.get("n_bits", 2048)
    cmd = [
        sys.executable,
        script_path,
        input_file,
        output_file,
        "--radius",
        str(radius),
        "--n_bits",
        str(n_bits),
        "--labeled-output-file",
        labeled_output,
        "--property-columns",
        context["target_column"],
    ]
    _run_subprocess(cmd)
    with open(context["paths"]["morgan_meta"], "w", encoding="utf-8") as meta_out:
        json.dump(
            {"radius": radius, "n_bits": n_bits},
            meta_out,
            indent=2,
        )
    validate_contract(
        bind_output_path(FEATURIZE_MORGAN_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )
    validate_contract(
        make_target_column_contract(
            name="featurize_morgan_labeled_output",
            target_column=context["target_column"],
            output_path=labeled_output,
        ),
        warn_only=True,
    )
    context["feature_matrix"] = labeled_output
    context["labels_matrix"] = labeled_output
    context["feature_method"] = "morgan"


def run_node_label_normalize(context: dict) -> None:
    label_config = context.get("label_config", {})
    source_column = label_config.get("source_column")
    target_column = label_config.get("target_column", context["target_column"])
    positive = label_config.get("positive")
    negative = label_config.get("negative")
    drop_unmapped = label_config.get("drop_unmapped", True)

    def _ensure_list(value):
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(value)]

    positive_list = _ensure_list(positive)
    negative_list = _ensure_list(negative)

    if not source_column or not positive_list or not negative_list:
        raise ValueError("label.normalize requires source_column, positive, and negative label lists.")

    script_path = os.path.join("utilities", "label_normalize.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["curated_labeled"]
    cmd = [
        sys.executable,
        script_path,
        input_file,
        output_file,
        "--source-column",
        source_column,
        "--target-column",
        target_column,
        "--positive",
        ",".join(positive_list),
        "--negative",
        ",".join(negative_list),
    ]
    if drop_unmapped:
        cmd.append("--drop-unmapped")
    _run_subprocess(cmd)
    context["curated_path"] = output_file
    context["target_column"] = target_column


def _has_explicit_feature_input_node(context: dict) -> bool:
    pipeline_nodes = list(context.get("pipeline_nodes") or [])
    if not pipeline_nodes:
        return False
    return any(node in _EXPLICIT_FEATURE_INPUT_NODES for node in pipeline_nodes)


def _resolve_feature_method(context: dict) -> str | None:
    feature_method = context.get("feature_method")
    if isinstance(feature_method, str) and feature_method.strip():
        return feature_method.strip().lower()

    pipeline_feature_input = str(context.get("pipeline_feature_input", "")).strip().lower()
    if pipeline_feature_input == "smiles_native":
        return "smiles_native"

    pipeline_nodes = [str(node).strip().lower() for node in (context.get("pipeline_nodes") or [])]
    for candidate in ("featurize.morgan", "featurize.rdkit_labeled", "featurize.rdkit", "featurize.none"):
        if candidate in pipeline_nodes:
            return candidate.split(".", 1)[1]

    feature_matrix = str(context.get("feature_matrix", "")).strip().lower()
    if "morgan" in feature_matrix:
        return "morgan"
    if "rdkit" in feature_matrix:
        return "rdkit"

    return None


def _write_explain_skip_marker(
    output_dir: str,
    *,
    reason: str,
    model_type: str,
    feature_method: str | None,
) -> str:
    skip_path = os.path.join(output_dir, "explain_skipped.json")
    _write_json_atomic(
        skip_path,
        {
            "status": "skipped",
            "reason": reason,
            "model_type": model_type,
            "feature_method": feature_method,
            "timestamp": _utc_now_iso(),
        },
    )
    return skip_path


def _normalize_and_validate_row_index(
    df: pd.DataFrame,
    row_index_col: str,
    *,
    stage: str,
) -> pd.DataFrame:
    if row_index_col not in df.columns:
        raise ValueError(f"{stage}: missing required row index column {row_index_col!r}.")
    raw = df[row_index_col]
    if raw.isna().any():
        raise ValueError(f"{stage}: {row_index_col} contains null values.")
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{stage}: {row_index_col} must contain integer-like values.")
    if (numeric % 1 != 0).any():
        raise ValueError(f"{stage}: {row_index_col} must contain integer-like values.")
    normalized = numeric.astype(int)
    duplicates = normalized[normalized.duplicated()].unique().tolist()
    if duplicates:
        preview = duplicates[:10]
        raise ValueError(
            f"{stage}: {row_index_col} must be unique; duplicate_count={len(duplicates)} preview={preview}."
        )
    out = df.copy()
    out[row_index_col] = normalized
    return out


def run_node_split(context: dict) -> None:
    split_config = context.get("split_config", {})
    mode = str(split_config.get("mode", "holdout")).strip().lower() or "holdout"
    strategy = str(split_config.get("strategy", "random")).strip().lower() or "random"
    test_size = float(split_config.get("test_size", 0.2))
    val_size = float(split_config.get("val_size", 0.1))
    random_state = int(split_config.get("random_state", context.get("global_random_state", 42)))
    stratify = _as_bool(split_config.get("stratify", False))
    stratify_column = split_config.get("stratify_column")
    min_coverage = split_config.get("min_coverage")
    allow_missing_val = _as_bool(split_config.get("allow_missing_val", True))
    require_disjoint = _as_bool(split_config.get("require_disjoint", False))
    if stratify and not stratify_column:
        stratify_column = context.get("target_column")

    def _safe_token(value: object) -> str:
        out = []
        for ch in str(value):
            if ch.isalnum() or ch in {"_", "-"}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    def _fmt_float(value: object) -> str:
        try:
            return str(float(value)).replace(".", "p")
        except Exception:
            return _safe_token(str(value))

    def _normalize_split_keys(raw: dict) -> dict[str, list[int]]:
        normalized: dict[str, list[int]] = {}
        for key, value in raw.items():
            lowered = str(key).lower()
            values = [int(i) for i in (value or [])]
            if lowered in {"valid", "val", "validation"}:
                normalized["val"] = values
            elif lowered == "test":
                normalized["test"] = values
            elif lowered == "train":
                normalized["train"] = values
            else:
                normalized[lowered] = values
        normalized.setdefault("train", [])
        normalized.setdefault("val", [])
        normalized.setdefault("test", [])
        return normalized

    def _plan_id(payload: dict) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]

    def _load_or_build_plan(plan_path: str, builder) -> dict:
        if os.path.exists(plan_path):
            try:
                return splitters.load_split_plan(plan_path)
            except json.JSONDecodeError as exc:
                logging.warning("Corrupt split plan at %s (%s); rebuilding.", plan_path, exc)
        plan = builder()
        splitters.save_split_plan(plan, plan_path)
        return splitters.load_split_plan(plan_path)

    def _extract_test_indices_from_plan(plan: dict, repeat_index: int, fold_index: int) -> list[int]:
        repeat_plans = list(plan.get("repeat_plans") or [])
        if repeat_index < 0 or repeat_index >= len(repeat_plans):
            raise ValueError(
                f"repeat_index={repeat_index} out of bounds for repeats={len(repeat_plans)}."
            )
        folds = list((repeat_plans[repeat_index] or {}).get("folds") or [])
        if fold_index < 0 or fold_index >= len(folds):
            raise ValueError(
                f"fold_index={fold_index} out of bounds for n_splits={len(folds)}."
            )
        return sorted(int(i) for i in (folds[fold_index] or {}).get("test_indices", []))

    def _split_train_pool(
        train_pool: list[int],
        val_cfg: dict,
        base_seed: int,
    ) -> tuple[list[int], list[int], dict]:
        if not train_pool:
            return [], [], {"requested_val_size": 0.0, "random_state": int(base_seed), "stratify": False}
        local_val_size = float(val_cfg.get("val_size", split_config.get("val_size", 0.0)))
        local_stratify = _as_bool(val_cfg.get("stratify", stratify))
        local_seed = int(val_cfg.get("random_state", base_seed))

        if local_val_size <= 0:
            return sorted(int(i) for i in train_pool), [], {
                "requested_val_size": local_val_size,
                "random_state": local_seed,
                "stratify": local_stratify,
            }
        if len(train_pool) < 2:
            raise ValueError(
                f"val_from_train requested (val_size={local_val_size}) but train_pool has only "
                f"{len(train_pool)} rows; cannot create train/val split."
            )

        stratify_values = None
        if local_stratify:
            if not stratify_column:
                raise ValueError("Stratified val_from_train requires split.stratify_column.")
            stratify_values = split_df.iloc[train_pool][stratify_column].to_numpy()

        try:
            local_split = splitters.random_split_indices(
                n_samples=len(train_pool),
                test_size=local_val_size,
                val_size=0.0,
                random_state=local_seed,
                stratify=stratify_values,
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to create val_from_train split. "
                f"train_pool_rows={len(train_pool)} val_size={local_val_size} "
                f"stratify={local_stratify} random_state={local_seed}."
            ) from exc
        local_train = [int(i) for i in local_split.get("train", [])]
        local_val = [int(i) for i in local_split.get("test", [])]
        if not local_train or not local_val:
            raise ValueError(
                "val_from_train produced an empty train or val split. "
                f"train_pool_rows={len(train_pool)} val_size={local_val_size} "
                f"-> train_rows={len(local_train)} val_rows={len(local_val)}."
            )
        train_idx = sorted(int(train_pool[i]) for i in local_train)
        val_idx = sorted(int(train_pool[i]) for i in local_val)
        return train_idx, val_idx, {
            "requested_val_size": local_val_size,
            "random_state": local_seed,
            "stratify": local_stratify,
        }

    curated_path = context.get("curated_path", context["paths"]["curated"])
    curated_df = pd.read_csv(curated_path)
    row_index_col = data_preprocessing.ROW_INDEX_COL
    if row_index_col not in curated_df.columns:
        curated_df[row_index_col] = curated_df.index.astype(int)
    curated_df = _normalize_and_validate_row_index(
        curated_df,
        row_index_col,
        stage="split curated input",
    )

    split_df = curated_df.copy()
    split_source = "curated"
    split_features_file = ""
    split_labels_file = ""
    retained_duplicate_feature_label_rows = 0
    feature_nodes = {
        "featurize.none",
        "featurize.rdkit",
        "featurize.rdkit_labeled",
        "featurize.morgan",
    }
    pipeline_nodes = list(context.get("pipeline_nodes") or [])
    explicit_feature_input_before_split = False
    if pipeline_nodes and "split" in pipeline_nodes:
        split_pos = pipeline_nodes.index("split")
        explicit_feature_input_before_split = any(
            node in feature_nodes for node in pipeline_nodes[:split_pos]
        )
    elif context.get("feature_matrix") or context.get("labels_matrix"):
        # Backward-compatible fallback for lightweight test contexts without pipeline_nodes.
        explicit_feature_input_before_split = True

    if explicit_feature_input_before_split:
        features_file, labels_file = _resolve_feature_inputs(context)
        if not (os.path.exists(features_file) and os.path.exists(labels_file)):
            raise ValueError(
                "split pre-split cleaning expected feature/label artifacts from explicit feature input, "
                f"but files were missing (features={features_file} labels={labels_file})."
            )
        X_split, y_split = data_preprocessing.load_features_labels(
            features_file,
            labels_file,
            context["target_column"],
            context.get("categorical_features"),
            exclude_columns=_exclude_feature_columns(context),
            drop_invalid_rows=True,
            drop_duplicate_rows=False,
            fail_on_duplicate_rows=False,
        )
        retained_duplicate_feature_label_rows = int(pd.concat([X_split, y_split], axis=1).duplicated().sum())
        eligible_row_ids = [int(i) for i in X_split.index.tolist()]
        if not eligible_row_ids:
            raise ValueError("Pre-split cleaning produced zero rows; cannot build split indices.")
        curated_row_id_universe = set(int(v) for v in curated_df[row_index_col].tolist())
        missing_from_curated = sorted(
            int(v) for v in set(eligible_row_ids) if int(v) not in curated_row_id_universe
        )
        if missing_from_curated:
            preview = missing_from_curated[:10]
            raise ValueError(
                "Pre-split feature cleaning produced row IDs not present in curated row-id universe. "
                f"row_index_column={row_index_col} missing_count={len(missing_from_curated)} preview={preview}. "
                "Feature artifacts likely came from a different dataset/version."
            )
        split_df = (
            split_df.set_index(row_index_col, drop=False)
            .reindex(eligible_row_ids)
            .dropna(subset=[row_index_col])
            .reset_index(drop=True)
        )
        if split_df.empty:
            raise ValueError(
                "Pre-split feature cleaning produced eligible row IDs that do not map to curated rows via "
                f"{row_index_col}. Feature artifacts likely lost {row_index_col} or came from a different dataset."
            )
        split_df = _normalize_and_validate_row_index(
            split_df,
            row_index_col,
            stage="split feature-cleaned alignment",
        )
        split_source = "feature_cleaned"
        split_features_file = features_file
        split_labels_file = labels_file
    else:
        split_source = "curated_no_explicit_feature_input"
        logging.info(
            "Pre-split feature cleaning skipped because no explicit feature input node runs before split."
        )

    if stratify:
        if not stratify_column:
            raise ValueError("split.stratify=true requires split.stratify_column to be set.")
        if stratify_column not in split_df.columns:
            raise ValueError(
                f"split.stratify_column={stratify_column!r} not found in split dataset."
            )
        unique = split_df[stratify_column].dropna().nunique()
        if unique > 20:
            raise ValueError(
                f"split.stratify_column={stratify_column!r} has {unique} unique values; "
                "stratification requires a low-cardinality label column (e.g., binary class)."
            )

    os.makedirs(context["paths"]["split_dir"], exist_ok=True)
    source_curated_count = len(curated_df)
    curated_count = len(split_df)
    all_indices_ordered = list(range(curated_count))
    dataset_fingerprint = splitters.compute_dataset_fingerprint(
        split_df,
        target_column=context.get("target_column"),
    )
    tdc_group = context.get("source", {}).get("group")
    tdc_name = context.get("source", {}).get("name")

    split_indices_raw: dict
    split_meta: dict[str, object] = {
        "mode": mode,
        "strategy": strategy,
        "source_curated_rows": int(source_curated_count),
        "dataset_rows": int(curated_count),
        "dataset_fingerprint": dataset_fingerprint,
        "row_index_column": row_index_col,
        "split_source": split_source,
        "split_features_file": split_features_file,
        "split_labels_file": split_labels_file,
        "retained_duplicate_feature_label_rows": int(retained_duplicate_feature_label_rows),
        "stratify": bool(stratify),
        "stratify_column": stratify_column,
        "random_state": int(random_state),
        "source_tdc_group": tdc_group,
        "source_tdc_name": tdc_name,
    }
    split_filename_parts: list[str] = []

    if mode == "holdout":
        split_indices_raw = splitters.build_split_indices(
            strategy=strategy,
            curated_df=split_df,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify_column=stratify_column if stratify else None,
            tdc_group=tdc_group,
            tdc_name=tdc_name,
        )
        split_meta.update(
            {
                "stage": "holdout",
                "test_size": float(test_size),
                "val_size": float(val_size),
                "repeat_index": None,
                "fold_index": None,
                "plan_id": None,
                "plan_path": None,
            }
        )
        split_filename_parts = [
            _safe_token(strategy),
            f"test{_fmt_float(test_size)}",
            f"val{_fmt_float(val_size)}",
            f"seed{_safe_token(random_state)}",
        ]
        if stratify and stratify_column:
            split_filename_parts.append(f"strat_{_safe_token(stratify_column)}")
        if strategy.startswith("tdc") and tdc_group and tdc_name:
            split_filename_parts.append(f"tdc_{_safe_token(tdc_group)}_{_safe_token(tdc_name)}")

    elif mode == "cv":
        if strategy not in {"random", "scaffold"}:
            raise ValueError(
                f"split.mode=cv currently supports strategy=random|scaffold; got {strategy!r}."
            )
        if strategy == "scaffold" and stratify:
            logging.warning(
                "split.mode=cv with strategy=scaffold ignores split.stratify; scaffold grouping defines folds."
            )
        cv_cfg = split_config.get("cv", {}) or {}
        n_splits = int(cv_cfg.get("n_splits", 5))
        repeats = int(cv_cfg.get("repeats", 1))
        fold_index = int(cv_cfg.get("fold_index", 0))
        repeat_index = int(cv_cfg.get("repeat_index", 0))
        cv_seed = int(cv_cfg.get("random_state", random_state))

        plan_spec = {
            "mode": mode,
            "strategy": strategy,
            "dataset_rows": int(curated_count),
            "dataset_fingerprint": dataset_fingerprint,
            "n_splits": n_splits,
            "repeats": repeats,
            "random_state": cv_seed,
            "stratify": bool(stratify),
            "stratify_column": stratify_column,
        }
        plan_id = _plan_id(plan_spec)
        plan_path = os.path.join(
            context["paths"]["split_dir"],
            f"{_safe_token(strategy)}_cv_k{n_splits}_r{repeats}_seed{cv_seed}_{plan_id}.plan.json",
        )

        def _build_cv_plan():
            stratify_values = None
            if stratify and stratify_column:
                stratify_values = split_df[stratify_column].to_numpy()
            if strategy == "scaffold":
                if "canonical_smiles" not in split_df.columns:
                    raise ValueError("Curated data must include canonical_smiles for scaffold CV splitting.")
                return splitters.scaffold_kfold_plan(
                    smiles_list=split_df["canonical_smiles"].astype(str).tolist(),
                    n_splits=n_splits,
                    repeats=repeats,
                    random_state=cv_seed,
                    dataset_fingerprint=dataset_fingerprint,
                )
            return splitters.random_kfold_plan(
                n_samples=curated_count,
                n_splits=n_splits,
                repeats=repeats,
                random_state=cv_seed,
                stratify=stratify_values,
                dataset_fingerprint=dataset_fingerprint,
            )

        plan = _load_or_build_plan(plan_path, _build_cv_plan)
        if int(plan.get("n_rows", -1)) != curated_count:
            raise ValueError(
                f"CV plan row-count mismatch: plan={plan.get('n_rows')} current={curated_count}. "
                "Delete stale plan file or update split parameters."
            )
        if str(plan.get("dataset_fingerprint")) != dataset_fingerprint:
            raise ValueError(
                "CV plan dataset fingerprint mismatch. Delete stale plan file or keep dataset fixed."
            )

        test_idx = _extract_test_indices_from_plan(plan, repeat_index=repeat_index, fold_index=fold_index)
        test_set = set(test_idx)
        train_pool = [i for i in all_indices_ordered if i not in test_set]
        train_pool.sort()
        train_idx, val_idx, val_meta = _split_train_pool(
            train_pool=train_pool,
            val_cfg=split_config.get("val_from_train", {}) or {},
            base_seed=cv_seed + repeat_index * 1000 + fold_index,
        )
        split_indices_raw = {"train": train_idx, "val": val_idx, "test": test_idx}
        split_meta.update(
            {
                "stage": "cv",
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "cv": {
                    "n_splits": n_splits,
                    "repeats": repeats,
                    "random_state": cv_seed,
                },
                "val_from_train": val_meta,
                "plan_id": plan_id,
                "plan_path": plan_path,
            }
        )
        split_filename_parts = [
            _safe_token(mode),
            _safe_token(strategy),
            f"k{n_splits}",
            f"r{repeats}",
            f"rep{repeat_index}",
            f"fold{fold_index}",
            f"seed{cv_seed}",
            plan_id,
        ]

    elif mode == "nested_holdout_cv":
        if strategy not in {"random", "scaffold"}:
            raise ValueError(
                f"split.mode=nested_holdout_cv currently supports strategy=random|scaffold; got {strategy!r}."
            )
        if strategy == "scaffold" and stratify:
            logging.warning(
                "split.mode=nested_holdout_cv with strategy=scaffold ignores split.stratify; "
                "scaffold grouping defines holdout/CV membership."
            )
        stage = str(split_config.get("stage", "inner")).strip().lower()
        if stage not in {"inner", "outer"}:
            raise ValueError("split.stage must be 'inner' or 'outer' for split.mode=nested_holdout_cv.")

        outer_cfg = split_config.get("outer", {}) or {}
        outer_test_size = float(outer_cfg.get("test_size", split_config.get("test_size", 0.2)))
        outer_seed = int(outer_cfg.get("random_state", random_state))
        outer_spec = {
            "mode": mode,
            "strategy": strategy,
            "stage": "outer_plan",
            "dataset_rows": int(curated_count),
            "dataset_fingerprint": dataset_fingerprint,
            "test_size": outer_test_size,
            "random_state": outer_seed,
            "stratify": bool(stratify),
            "stratify_column": stratify_column,
        }
        outer_plan_id = _plan_id(outer_spec)
        outer_plan_path = os.path.join(
            context["paths"]["split_dir"],
            f"{_safe_token(strategy)}_nested_outer_test{_fmt_float(outer_test_size)}_seed{outer_seed}_{outer_plan_id}.plan.json",
        )

        def _build_outer_plan():
            if strategy == "scaffold":
                if "canonical_smiles" not in split_df.columns:
                    raise ValueError("Curated data must include canonical_smiles for scaffold splitting.")
                split = splitters.scaffold_split_indices(
                    split_df["canonical_smiles"].astype(str).tolist(),
                    test_size=outer_test_size,
                    val_size=0.0,
                    random_state=outer_seed,
                )
            else:
                stratify_values = None
                if stratify and stratify_column:
                    stratify_values = split_df[stratify_column].to_numpy()
                split = splitters.random_split_indices(
                    n_samples=curated_count,
                    test_size=outer_test_size,
                    val_size=0.0,
                    random_state=outer_seed,
                    stratify=stratify_values,
                )
            return {
                "version": 1,
                "plan_type": "nested_outer_holdout",
                "strategy": strategy,
                "dataset_fingerprint": dataset_fingerprint,
                "n_rows": curated_count,
                "test_size": outer_test_size,
                "random_state": outer_seed,
                "test_indices": [int(i) for i in split.get("test", [])],
            }

        outer_plan = _load_or_build_plan(outer_plan_path, _build_outer_plan)
        if int(outer_plan.get("n_rows", -1)) != curated_count:
            raise ValueError(
                f"Nested outer plan row-count mismatch: plan={outer_plan.get('n_rows')} current={curated_count}. "
                "Delete stale outer plan or keep dataset fixed."
            )
        if str(outer_plan.get("dataset_fingerprint")) != dataset_fingerprint:
            raise ValueError(
                "Nested outer plan dataset fingerprint mismatch. Delete stale outer plan or keep dataset fixed."
            )
        outer_test_idx = sorted(int(i) for i in outer_plan.get("test_indices", []))
        outer_test_set = set(outer_test_idx)
        dev_idx = [i for i in all_indices_ordered if i not in outer_test_set]
        dev_idx.sort()

        split_meta.update(
            {
                "stage": stage,
                "outer": {
                    "test_size": outer_test_size,
                    "random_state": outer_seed,
                    "plan_id": outer_plan_id,
                    "plan_path": outer_plan_path,
                    "outer_test_size": len(outer_test_idx),
                    "dev_size": len(dev_idx),
                },
            }
        )

        if stage == "outer":
            train_idx, val_idx, val_meta = _split_train_pool(
                train_pool=dev_idx,
                val_cfg=split_config.get("val_from_train", {}) or {},
                base_seed=outer_seed + 999,
            )
            split_indices_raw = {"train": train_idx, "val": val_idx, "test": outer_test_idx}
            split_meta.update(
                {
                    "repeat_index": None,
                    "fold_index": None,
                    "val_from_train": val_meta,
                    "plan_id": outer_plan_id,
                    "plan_path": outer_plan_path,
                    "inner": None,
                }
            )
            split_filename_parts = [
                _safe_token(mode),
                "outer",
                _safe_token(strategy),
                f"test{_fmt_float(outer_test_size)}",
                f"seed{outer_seed}",
                outer_plan_id,
            ]
        else:
            inner_cfg = split_config.get("inner", {}) or {}
            n_splits = int(inner_cfg.get("n_splits", 5))
            repeats = int(inner_cfg.get("repeats", 1))
            fold_index = int(inner_cfg.get("fold_index", 0))
            repeat_index = int(inner_cfg.get("repeat_index", 0))
            inner_seed = int(inner_cfg.get("random_state", random_state))

            dev_df = split_df.iloc[dev_idx].reset_index(drop=True)
            inner_spec = {
                "mode": mode,
                "stage": "inner",
                "strategy": strategy,
                "dataset_rows": int(curated_count),
                "dataset_fingerprint": dataset_fingerprint,
                "outer_plan_id": outer_plan_id,
                "dev_rows": len(dev_idx),
                "n_splits": n_splits,
                "repeats": repeats,
                "random_state": inner_seed,
                "stratify": bool(stratify),
                "stratify_column": stratify_column,
            }
            inner_plan_id = _plan_id(inner_spec)
            inner_plan_path = os.path.join(
                context["paths"]["split_dir"],
                f"{_safe_token(strategy)}_nested_inner_k{n_splits}_r{repeats}_seed{inner_seed}_{inner_plan_id}.plan.json",
            )

            def _build_inner_plan():
                stratify_values = None
                if stratify and stratify_column and stratify_column in dev_df.columns:
                    stratify_values = dev_df[stratify_column].to_numpy()
                if strategy == "scaffold":
                    if "canonical_smiles" not in dev_df.columns:
                        raise ValueError("Curated data must include canonical_smiles for scaffold CV splitting.")
                    return splitters.scaffold_kfold_plan(
                        smiles_list=dev_df["canonical_smiles"].astype(str).tolist(),
                        n_splits=n_splits,
                        repeats=repeats,
                        random_state=inner_seed,
                        dataset_fingerprint=dataset_fingerprint,
                    )
                return splitters.random_kfold_plan(
                    n_samples=len(dev_df),
                    n_splits=n_splits,
                    repeats=repeats,
                    random_state=inner_seed,
                    stratify=stratify_values,
                    dataset_fingerprint=dataset_fingerprint,
                )

            inner_plan = _load_or_build_plan(inner_plan_path, _build_inner_plan)
            if int(inner_plan.get("n_rows", -1)) != len(dev_df):
                raise ValueError(
                    f"Nested inner plan row-count mismatch: plan={inner_plan.get('n_rows')} current={len(dev_df)}. "
                    "Delete stale inner plan or keep dataset fixed."
                )
            if str(inner_plan.get("dataset_fingerprint")) != dataset_fingerprint:
                raise ValueError(
                    "Nested inner plan dataset fingerprint mismatch. Delete stale inner plan or keep dataset fixed."
                )
            local_test_idx = _extract_test_indices_from_plan(
                inner_plan,
                repeat_index=repeat_index,
                fold_index=fold_index,
            )
            fold_test_idx = sorted(int(dev_idx[i]) for i in local_test_idx)
            fold_test_set = set(fold_test_idx)
            train_pool = [i for i in dev_idx if i not in fold_test_set]
            train_pool.sort()
            train_idx, val_idx, val_meta = _split_train_pool(
                train_pool=train_pool,
                val_cfg=split_config.get("val_from_train", {}) or {},
                base_seed=inner_seed + repeat_index * 1000 + fold_index,
            )
            split_indices_raw = {"train": train_idx, "val": val_idx, "test": fold_test_idx}
            split_meta.update(
                {
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "val_from_train": val_meta,
                    "plan_id": inner_plan_id,
                    "plan_path": inner_plan_path,
                    "inner": {
                        "n_splits": n_splits,
                        "repeats": repeats,
                        "random_state": inner_seed,
                        "plan_id": inner_plan_id,
                        "plan_path": inner_plan_path,
                    },
                }
            )
            split_filename_parts = [
                _safe_token(mode),
                "inner",
                _safe_token(strategy),
                f"k{n_splits}",
                f"r{repeats}",
                f"rep{repeat_index}",
                f"fold{fold_index}",
                f"seed{inner_seed}",
                outer_plan_id,
                inner_plan_id,
            ]
    else:
        raise ValueError(
            f"Unsupported split.mode={mode!r}. Expected one of: holdout, cv, nested_holdout_cv."
        )

    normalized_local = _normalize_split_keys(split_indices_raw)
    row_id_lookup = [int(v) for v in split_df[row_index_col].tolist()]
    normalized: dict[str, list[int]] = {}
    for split_name, local_ids in normalized_local.items():
        remapped: list[int] = []
        for local_idx in local_ids:
            if local_idx < 0 or local_idx >= len(row_id_lookup):
                raise ValueError(
                    f"Split index out of bounds for split={split_name}: idx={local_idx} "
                    f"with split_rows={len(row_id_lookup)}."
                )
            remapped.append(int(row_id_lookup[local_idx]))
        normalized[split_name] = remapped

    dataset_split_path = os.path.join(
        context["paths"]["split_dir"],
        "_".join(split_filename_parts) + ".json",
    )
    run_split_path = os.path.join(context["run_dir"], "split_indices.json")
    run_split_meta_path = os.path.join(context["run_dir"], "split_meta.json")

    splitters.save_split_indices(normalized, dataset_split_path)
    splitters.save_split_indices(normalized, run_split_path)

    train_set = set(normalized.get("train", []))
    val_set = set(normalized.get("val", []))
    test_set = set(normalized.get("test", []))
    all_indices = set(train_set | val_set | test_set)
    if not all_indices:
        raise ValueError("Split mapping produced zero indices.")
    valid_row_ids = set(row_id_lookup)
    invalid_ids = sorted(int(i) for i in all_indices if i not in valid_row_ids)
    if invalid_ids:
        preview = invalid_ids[:10]
        raise ValueError(
            "Split indices contain ids outside the split dataset universe. "
            f"count={len(invalid_ids)} preview={preview}."
        )
    if min_coverage is not None:
        coverage = len(all_indices) / max(1, curated_count)
        if coverage < min_coverage:
            raise ValueError(
                f"Split coverage {coverage:.2%} below min_coverage={float(min_coverage):.2%}. "
                "Check split mapping or strategy."
            )
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        message = f"Split overlap detected across train/val/test: {len(overlap)} shared indices."
        if require_disjoint:
            raise ValueError(message)
        logging.warning(message)
    if not normalized.get("train"):
        raise ValueError("Split mapping produced empty train split.")
    if not normalized.get("test"):
        raise ValueError("Split mapping produced empty test split.")

    expected_val_size = split_meta.get("val_from_train", {}).get("requested_val_size", val_size)
    if float(expected_val_size or 0) > 0 and not normalized.get("val"):
        if strategy.startswith("tdc") and allow_missing_val:
            pass
        else:
            raise ValueError("Split mapping produced empty validation split.")

    split_meta.update(
        {
            "split_indices_path": run_split_path,
            "dataset_split_path": dataset_split_path,
            "sizes": {
                "train": len(normalized.get("train", [])),
                "val": len(normalized.get("val", [])),
                "test": len(normalized.get("test", [])),
                "assigned_total": len(all_indices),
            },
            "coverage": {
                "source_curated_rows": source_curated_count,
                "curated_rows": curated_count,
                "dropped_before_split": max(0, source_curated_count - curated_count),
                "assigned_rows": len(all_indices),
                "assigned_fraction": len(all_indices) / max(1, curated_count),
            },
            "require_disjoint": require_disjoint,
        }
    )
    with open(run_split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    context["split_indices"] = normalized
    context["split_path"] = run_split_path
    context["dataset_split_path"] = dataset_split_path
    context["split_meta_path"] = run_split_meta_path
    context["split_meta"] = split_meta


def _resolve_feature_inputs(
    context: dict,
    *,
    require_explicit: bool = False,
    consumer: str = "node",
) -> tuple[str, str]:
    feature_matrix = context.get("feature_matrix")
    labels_matrix = context.get("labels_matrix")
    if feature_matrix and labels_matrix:
        return feature_matrix, labels_matrix
    if feature_matrix and not labels_matrix:
        return feature_matrix, context.get("curated_path", context["paths"]["curated"])
    if require_explicit and not _has_explicit_feature_input_node(context):
        raise ValueError(
            f"{consumer} requires an explicit feature input node in pipeline.nodes "
            "(featurize.none/featurize.rdkit/featurize.rdkit_labeled/featurize.morgan)."
        )
    pipeline_type = context.get("pipeline_type", "chembl")
    if pipeline_type == "qm9":
        return context["paths"]["rdkit_labeled"], context["paths"]["rdkit_labeled"]
    return context["paths"]["rdkit_descriptors"], context["paths"]["pic50_3class"]


def _preprocess_params(context: dict) -> tuple[float, float, tuple[float, float], str, int, int]:
    preprocess_config = context.get("preprocess_config", {})
    variance_threshold = preprocess_config.get("variance_threshold", 0.8 * (1 - 0.8))
    corr_threshold = preprocess_config.get("corr_threshold", 0.95)
    clip_range = preprocess_config.get("clip_range", (-1e10, 1e10))
    if isinstance(clip_range, list):
        clip_range = tuple(clip_range)
    scaler = str(preprocess_config.get("scaler", "robust")).strip().lower() or "robust"
    stable_k = preprocess_config.get("stable_features_k", 50)
    random_state = _resolve_seed(
        context.get("global_random_state", 42),
        preprocess_config.get("random_state"),
    )
    return variance_threshold, corr_threshold, clip_range, scaler, stable_k, random_state


def _should_skip_tabular_preprocess_for_chemprop(context: dict) -> bool:
    if not _is_chemprop_like_model(context.get("model_type")):
        return False
    _, _, _, scaler, _, _ = _preprocess_params(context)
    if scaler != "none":
        return False
    if _has_explicit_feature_input_node(context):
        return False
    pipeline_nodes = set(context.get("pipeline_nodes") or [])
    return "select.features" not in pipeline_nodes


def _exclude_feature_columns(context: dict) -> list[str]:
    train_config = context.get("train_config", {}) or {}
    train_features = train_config.get("features", {}) if isinstance(train_config, dict) else {}
    raw = train_features.get("exclude_columns", [])
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple, set, str)):
        raise ValueError(
            "train.features.exclude_columns must be a list or comma-separated string of column names."
        )
    return _as_str_list(raw)


def _split_index_universe(context: dict) -> list[int] | None:
    split_indices = context.get("split_indices")
    if not split_indices:
        return None
    universe: set[int] = set()
    for split_name in ("train", "val", "test"):
        universe.update(int(i) for i in (split_indices.get(split_name, []) or []))
    return sorted(universe)


def _resolve_split_partitions(
    context: dict,
    index: pd.Index,
) -> tuple[list[int], list[int], list[int]] | None:
    """
    Resolve train/val/test indices against a cleaned feature/label index.

    If split_indices exist in the context, we filter them down to rows that are
    still present after NaN/Inf cleaning. If a split becomes
    empty after filtering, we raise instead of silently re-splitting elsewhere.
    """
    split_indices = context.get("split_indices")
    if not split_indices:
        return None

    split_config = context.get("split_config", {}) or {}
    train_idx = list(split_indices.get("train", []) or [])
    val_idx = list(split_indices.get("val", []) or [])
    test_idx = list(split_indices.get("test", []) or [])
    if not test_idx and val_idx:
        # Some sources only provide train/val; treat val as test downstream.
        test_idx, val_idx = val_idx, []

    orig_counts = (len(train_idx), len(val_idx), len(test_idx))
    available = set(index.tolist())
    original_by_split = {
        "train": list(train_idx),
        "val": list(val_idx),
        "test": list(test_idx),
    }
    train_idx = [i for i in train_idx if i in available]
    val_idx = [i for i in val_idx if i in available]
    test_idx = [i for i in test_idx if i in available]

    filtered_by_split = {
        "train": list(train_idx),
        "val": list(val_idx),
        "test": list(test_idx),
    }
    missing_by_split = {
        name: max(0, len(original_by_split[name]) - len(filtered_by_split[name]))
        for name in ("train", "val", "test")
    }

    def _coverage(split_name: str) -> float:
        original = len(original_by_split[split_name])
        if original == 0:
            return 1.0
        return len(filtered_by_split[split_name]) / original

    require_full_test_coverage = _as_bool(split_config.get("require_full_test_coverage", False))
    min_test_coverage = split_config.get("min_test_coverage")
    min_train_coverage = split_config.get("min_train_coverage")
    min_val_coverage = split_config.get("min_val_coverage")
    if require_full_test_coverage and min_test_coverage is None:
        min_test_coverage = 1.0

    for split_name, threshold in (
        ("train", min_train_coverage),
        ("val", min_val_coverage),
        ("test", min_test_coverage),
    ):
        if threshold is None:
            continue
        threshold_f = float(threshold)
        if _coverage(split_name) + 1e-12 < threshold_f:
            reason = "split_indices coverage check failed after feature/label cleaning."
            if split_name == "test" and require_full_test_coverage and float(threshold_f) >= 1.0:
                reason = (
                    "split.require_full_test_coverage=true but test membership changed after cleaning."
                )
            raise ValueError(
                f"{reason} "
                f"split={split_name} threshold={threshold_f:.2%} actual={_coverage(split_name):.2%} "
                f"missing={missing_by_split[split_name]} "
                f"(original={len(original_by_split[split_name])} filtered={len(filtered_by_split[split_name])}) "
                "Featurizer/cleaning dropped rows; this run is not directly comparable."
            )

    if not train_idx or not test_idx:
        raise ValueError(
            "split_indices did not align with cleaned features/labels (train/test empty after filtering). "
            "Do not re-split downstream: run the 'split' node after any row-filtering/cleaning, or ensure "
            f"your feature/label matrices preserve {data_preprocessing.ROW_INDEX_COL}. "
            f"(available_rows={len(available)} split_counts(train,val,test)={orig_counts} "
            f"filtered_counts(train,val,test)=({len(train_idx)},{len(val_idx)},{len(test_idx)}))"
        )
    return train_idx, val_idx, test_idx


def run_node_preprocess_features(context: dict) -> None:
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "preprocess.features requires split_indices from the split node. "
            "Add 'split' before 'preprocess.features' in pipeline.nodes."
        )

    if _should_skip_tabular_preprocess_for_chemprop(context):
        model_type = str(context.get("model_type", "chemprop")).strip() or "chemprop"
        logging.info(
            "Skipping tabular preprocess.features for %s because pipeline.feature_input is smiles-native "
            "and preprocess.scaler is none.",
            model_type,
        )
        context["preprocessed_ready"] = False
        return

    features_file, labels_file = _resolve_feature_inputs(
        context,
        require_explicit=True,
        consumer="preprocess.features",
    )
    validate_contract(
        bind_output_path(PREPROCESS_FEATURES_INPUT_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="preprocess_input_labels_dynamic",
                target_column=context["target_column"],
            ),
            labels_file,
        ),
        warn_only=False,
    )

    X_clean, y_clean = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        context["target_column"],
        context.get("categorical_features"),
        exclude_columns=_exclude_feature_columns(context),
        restrict_to_indices=_split_index_universe(context),
        drop_invalid_rows=False,
        drop_duplicate_rows=False,
        fail_on_invalid_rows=True,
        fail_on_duplicate_rows=False,
    )
    data_preprocessing.verify_data_quality(X_clean, y_clean)

    partitions = _resolve_split_partitions(context, X_clean.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, test_idx = partitions
    X_train = X_clean.loc[train_idx]
    X_test = X_clean.loc[test_idx]

    variance_threshold, corr_threshold, clip_range, scaler, _, _ = _preprocess_params(context)
    preprocessor = data_preprocessing.fit_preprocessor(
        X_train,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
        clip_range=clip_range,
        scaler=scaler,
    )
    X_preprocessed = data_preprocessing.transform_preprocessor(X_clean, preprocessor)
    X_preprocessed.index = X_clean.index
    y_aligned = y_clean

    preprocessed_features = context["paths"]["preprocessed_features"]
    preprocessed_labels = context["paths"]["preprocessed_labels"]
    _track_heavy_artifact(context, preprocessed_features)
    X_preprocessed = X_preprocessed.copy()
    X_preprocessed[data_preprocessing.ROW_INDEX_COL] = X_preprocessed.index
    X_preprocessed.to_csv(preprocessed_features, index=False)
    y_values = y_aligned.values
    if hasattr(y_values, "ndim") and y_values.ndim > 1:
        y_values = y_values[:, 0]
    pd.DataFrame(
        {
            context["target_column"]: y_values,
            data_preprocessing.ROW_INDEX_COL: y_aligned.index,
        }
    ).to_csv(
        preprocessed_labels, index=False
    )
    joblib.dump(preprocessor, context["paths"]["preprocess_artifacts"])
    data_preprocessing.check_data_leakage(X_train, X_test)

    validate_contract(
        bind_output_path(PREPROCESS_FEATURES_OUTPUT_CONTRACT, preprocessed_features),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="preprocess_labels_output_dynamic",
                target_column=context["target_column"],
            ),
            preprocessed_labels,
        ),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(PREPROCESS_ARTIFACTS_CONTRACT, context["paths"]["preprocess_artifacts"]),
        warn_only=True,
    )
    context["preprocessed_ready"] = True


def run_node_select_features(context: dict) -> None:
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "select.features requires split_indices from the split node. "
            "Add 'split' before 'select.features' in pipeline.nodes."
        )

    features_file = context["paths"]["preprocessed_features"]
    labels_file = context["paths"]["preprocessed_labels"]

    validate_contract(
        bind_output_path(SELECT_FEATURES_INPUT_FEATURES_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="select_input_labels_dynamic",
                target_column=context["target_column"],
            ),
            labels_file,
        ),
        warn_only=False,
    )

    target_column = context["target_column"]
    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
        exclude_columns=_exclude_feature_columns(context),
        restrict_to_indices=_split_index_universe(context),
        drop_invalid_rows=False,
        drop_duplicate_rows=False,
        fail_on_invalid_rows=True,
        fail_on_duplicate_rows=False,
    )

    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, test_idx = partitions
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]

    _, _, _, _, stable_k, random_state = _preprocess_params(context)
    X_selected_train = data_preprocessing.select_stable_features(
        X_train,
        y_train,
        random_state=random_state,
        k=stable_k,
        out_path=context["paths"]["selected_features_list"],
    )

    X_selected = X[X_selected_train.columns]
    data_preprocessing.check_data_leakage(X_train, X_test)

    selected_features = context["paths"]["selected_features"]
    _track_heavy_artifact(context, selected_features)
    X_selected_out = X_selected.copy()
    X_selected_out[data_preprocessing.ROW_INDEX_COL] = X_selected_out.index
    X_selected_out.to_csv(selected_features, index=False)

    validate_contract(
        bind_output_path(SELECT_FEATURES_OUTPUT_CONTRACT, selected_features),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(SELECT_FEATURES_LIST_CONTRACT, context["paths"]["selected_features_list"]),
        warn_only=True,
    )
    context["selected_ready"] = True


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, set):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


def _as_int_list(value: object, default: list[int]) -> list[int]:
    if value is None:
        raw = default
    else:
        raw = value
    items = _as_str_list(raw)
    out: list[int] = []
    for item in items:
        out.append(int(item))
    return out


def _normalize_tdc_group_name(group_name: str) -> str:
    token = str(group_name).strip().lower().replace("-", "_")
    if token in {"admet_group", "admet", "adme_group", "adme"}:
        return "ADMET_Group"
    raise ValueError(f"Unsupported TDC benchmark group: {group_name!r}")


def _resolve_train_tdc_config(context: dict) -> dict:
    train_tdc_config = context.get("train_tdc_config", {}) or {}
    source_cfg = context.get("source", {}) or {}

    raw_group = train_tdc_config.get("group") or source_cfg.get("group") or "ADMET_Group"
    group = _normalize_tdc_group_name(str(raw_group))

    benchmarks = _as_str_list(
        train_tdc_config.get("benchmarks")
        or train_tdc_config.get("benchmark")
        or source_cfg.get("name")
    )
    if not benchmarks:
        raise ValueError(
            "train_tdc requires at least one benchmark name (train_tdc.benchmarks or get_data.source.name)."
        )

    split_type = str(train_tdc_config.get("split_type", "default")).strip() or "default"
    seeds = _as_int_list(train_tdc_config.get("seeds"), default=[1, 2, 3, 4, 5])
    if not seeds:
        raise ValueError("train_tdc.seeds must contain at least one integer seed.")

    model_cfg = train_tdc_config.get("model", {}) or {}
    tuning_cfg = train_tdc_config.get("tuning", {}) or {}
    early_cfg = train_tdc_config.get("early_stopping", {}) or {}
    featurize_cfg = train_tdc_config.get("featurize", {}) or {}

    return {
        "group": group,
        "benchmarks": benchmarks,
        "split_type": split_type,
        "seeds": seeds,
        "path": str(train_tdc_config.get("path", context["base_dir"])),
        "model": model_cfg,
        "tuning": tuning_cfg,
        "early_stopping": early_cfg,
        "featurize": featurize_cfg,
    }


def _load_tdc_benchmark_group(group_name: str, path: str):
    try:
        from tdc.benchmark_group import admet_group
    except Exception as exc:
        raise ImportError(
            "TDC benchmark workflow requires pytdc benchmark APIs. Install pytdc to use 'train.tdc'."
        ) from exc

    if group_name != "ADMET_Group":
        raise ValueError(f"Unsupported TDC benchmark group: {group_name}")
    return admet_group(path=path)


def _find_tdc_smiles_column(df: pd.DataFrame) -> str:
    for candidate in ["Drug", "drug", "SMILES", "Smiles", "smiles", "canonical_smiles"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("TDC benchmark split frame is missing a SMILES column.")


def _find_tdc_label_column(df: pd.DataFrame, target_column: str) -> str:
    if target_column in df.columns:
        return target_column
    for candidate in ["Y", "label", "Label", "y"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"TDC benchmark split frame is missing label column. Expected '{target_column}' or one of Y/label."
    )


def _morgan_features_from_smiles(
    smiles_values: pd.Series,
    radius: int,
    n_bits: int,
    split_name: str,
) -> pd.DataFrame:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception as exc:
        raise ImportError(
            "RDKit is required for CatBoost TDC benchmarking with Morgan fingerprints."
        ) from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    rows: list[np.ndarray] = []
    invalid_positions: list[int] = []
    for idx, smi in enumerate(smiles_values.astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_positions.append(idx)
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((int(n_bits),), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)

    if invalid_positions:
        preview = invalid_positions[:5]
        raise ValueError(
            f"TDC {split_name} split has {len(invalid_positions)} invalid SMILES "
            f"(first row positions: {preview})."
        )

    return pd.DataFrame(rows, columns=[f"fp_{i}" for i in range(int(n_bits))])


def run_node_train_tdc(context: dict) -> None:
    cfg = _resolve_train_tdc_config(context)
    model_cfg = dict(cfg.get("model", {}) or {})
    model_type = str(model_cfg.get("type", "")).strip()
    if context.get("task_type") != "classification":
        raise ValueError("train.tdc currently supports classification benchmarks only.")
    if model_type != "catboost_classifier":
        raise ValueError(
            "train.tdc currently supports model.type=catboost_classifier. "
            "Use the regular 'train' node for other models."
        )

    featurize_config = cfg.get("featurize", {}) or {}
    radius = int(featurize_config.get("radius", 2))
    n_bits = int(featurize_config.get("n_bits", 2048))

    output_dir = os.path.join(context["run_dir"], "tdc_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    logging.warning("##### WARNING: TDC BENCHMARK TRAINING WORKFLOW ENABLED #####")
    logging.warning(
        "##### group=%s benchmarks=%s seeds=%s split_type=%s #####",
        cfg["group"],
        ",".join(cfg["benchmarks"]),
        ",".join(str(seed) for seed in cfg["seeds"]),
        cfg["split_type"],
    )
    logging.warning("##### Results will be reported via TDC evaluate_many #####")

    group = _load_tdc_benchmark_group(cfg["group"], cfg["path"])
    predictions_list: list[dict[str, list[float]]] = []
    seed_metric_rows: list[dict[str, object]] = []

    model_config = dict(model_cfg)
    model_config["_debug_logging"] = context.get("debug_logging", False)
    tuning_cfg = dict(cfg.get("tuning", {}) or {})
    early_cfg = dict(cfg.get("early_stopping", {}) or {})

    for seed in cfg["seeds"]:
        seed_predictions: dict[str, list[float]] = {}
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for benchmark_name in cfg["benchmarks"]:
            benchmark = group.get(benchmark_name)
            canonical_name = str(benchmark.get("name", benchmark_name))
            train_df, valid_df = group.get_train_valid_split(
                benchmark=canonical_name,
                split_type=cfg["split_type"],
                seed=int(seed),
            )
            test_df = benchmark["test"]
            if not isinstance(train_df, pd.DataFrame) or not isinstance(valid_df, pd.DataFrame):
                raise ValueError(f"TDC benchmark split is not tabular for {canonical_name}.")
            if not isinstance(test_df, pd.DataFrame):
                raise ValueError(f"TDC benchmark test split is not tabular for {canonical_name}.")

            smiles_col = _find_tdc_smiles_column(train_df)
            label_col = _find_tdc_label_column(train_df, context["target_column"])
            if label_col not in valid_df.columns or label_col not in test_df.columns:
                raise ValueError(
                    f"TDC benchmark {canonical_name} split is missing label column '{label_col}'."
                )
            if smiles_col not in valid_df.columns or smiles_col not in test_df.columns:
                raise ValueError(
                    f"TDC benchmark {canonical_name} split is missing smiles column '{smiles_col}'."
                )

            X_train = _morgan_features_from_smiles(train_df[smiles_col], radius, n_bits, "train")
            X_val = _morgan_features_from_smiles(valid_df[smiles_col], radius, n_bits, "valid")
            X_test = _morgan_features_from_smiles(test_df[smiles_col], radius, n_bits, "test")

            y_train = train_models._ensure_binary_labels(train_df[label_col])
            y_val = train_models._ensure_binary_labels(valid_df[label_col])
            y_test = train_models._ensure_binary_labels(test_df[label_col])

            bench_slug = canonical_name.lower().replace(" ", "_")
            bench_out_dir = os.path.join(seed_dir, bench_slug)
            os.makedirs(bench_out_dir, exist_ok=True)
            _track_heavy_artifact(
                context,
                os.path.join(bench_out_dir, f"{model_type}_best_model.cbm"),
            )

            estimator, train_result = train_models.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_type=model_type,
                output_dir=bench_out_dir,
                random_state=_resolve_seed(context.get("global_random_state", 42), int(seed)),
                cv_folds=int(tuning_cfg.get("cv_folds", 5)),
                search_iters=int(tuning_cfg.get("search_iters", 100)),
                use_hpo=_as_bool(tuning_cfg.get("use_hpo", False)),
                hpo_trials=int(tuning_cfg.get("hpo_trials", 30)),
                patience=int(early_cfg.get("patience", 20)),
                task_type="classification",
                model_config=model_config,
                X_val=X_val,
                y_val=y_val,
            )
            _track_heavy_artifact(context, train_result.model_path)
            _track_heavy_artifact_glob(
                context,
                os.path.join(bench_out_dir, "**", "*_best_model.*"),
            )

            y_pred_test = estimator.predict_proba(X_test)[:, 1].astype(float)
            seed_predictions[canonical_name] = [float(v) for v in y_pred_test.tolist()]

            pred_df = pd.DataFrame(
                {
                    "smiles": test_df[smiles_col].astype(str).tolist(),
                    "y_true": y_test.tolist(),
                    "y_pred_proba": y_pred_test.tolist(),
                }
            )
            pred_df.to_csv(os.path.join(bench_out_dir, "tdc_test_predictions.csv"), index=False)

            metrics_payload = {}
            if os.path.exists(train_result.metrics_path):
                with open(train_result.metrics_path, "r", encoding="utf-8") as f:
                    metrics_payload = json.load(f)
            seed_metric_rows.append(
                {
                    "seed": int(seed),
                    "benchmark": canonical_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "auc": metrics_payload.get("auc"),
                    "auprc": metrics_payload.get("auprc"),
                    "accuracy": metrics_payload.get("accuracy"),
                    "f1": metrics_payload.get("f1"),
                    "metrics_path": train_result.metrics_path,
                }
            )
            logging.info(
                "TDC benchmark run complete: benchmark=%s seed=%s n_train=%s n_val=%s n_test=%s",
                canonical_name,
                seed,
                len(X_train),
                len(X_val),
                len(X_test),
            )

        predictions_list.append(seed_predictions)
        with open(
            os.path.join(seed_dir, f"tdc_predictions_seed_{seed}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(seed_predictions, f, indent=2)

    results = group.evaluate_many(predictions_list)
    results_path = os.path.join(output_dir, "tdc_benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_rows: list[dict[str, object]] = []
    for benchmark_name, value in results.items():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            summary_rows.append(
                {
                    "benchmark": benchmark_name,
                    "mean": float(value[0]),
                    "std": float(value[1]),
                }
            )
        else:
            summary_rows.append(
                {
                    "benchmark": benchmark_name,
                    "mean": value,
                    "std": None,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "tdc_benchmark_summary.csv"),
        index=False,
    )
    if seed_metric_rows:
        pd.DataFrame(seed_metric_rows).to_csv(
            os.path.join(output_dir, "tdc_seed_metrics.csv"),
            index=False,
        )

    logging.warning("##### TDC benchmark workflow completed. results=%s #####", results_path)

    context["tdc_benchmark_results_path"] = results_path
    validate_contract(
        bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_train(context: dict) -> None:
    pipeline_type = context["pipeline_type"]
    output_dir = context["run_dir"]
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]
    task_type = context.get("task_type", "regression")
    train_config = context.get("train_config", {}) or {}
    model_config = dict(context.get("model_config", {}) or {})
    tuning_cfg = train_config.get("tuning", {}) if isinstance(train_config, dict) else {}
    reporting_cfg = train_config.get("reporting", {}) if isinstance(train_config, dict) else {}
    early_cfg = train_config.get("early_stopping", {}) if isinstance(train_config, dict) else {}
    train_random_state = _resolve_seed(
        context.get("global_random_state", 42),
        train_config.get("random_state") if isinstance(train_config, dict) else None,
    )
    if "plot_split_performance" in reporting_cfg and "plot_split_performance" not in model_config:
        model_config["plot_split_performance"] = _as_bool(
            reporting_cfg.get("plot_split_performance", False)
        )
    if isinstance(tuning_cfg, dict):
        model_config["tuning"] = dict(tuning_cfg)
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "train requires split_indices from the split node. "
            "Add 'split' before 'train' in pipeline.nodes."
        )

    # Chemprop-like models are SMILES-native; they do not use tabular descriptors.
    # For apples-to-apples benchmarking, we still rely on CheMLFlow's split_indices.
    if _is_chemprop_like_model(model_type):
        curated_path = context.get("curated_path", paths["curated"])
        curated_df = pd.read_csv(curated_path)
        if not split_indices.get("val"):
            raise ValueError(
                "SMILES-native training requires an explicit validation split from the split node. "
                "Set split.val_size > 0."
            )

        chemprop_model_config = dict(model_config)
        if str(model_type).strip().lower() == "chemeleon":
            chemprop_model_config["foundation"] = "chemeleon"
        _track_heavy_artifact(context, os.path.join(output_dir, "chemprop_best_model.ckpt"))
        _track_heavy_artifact_glob(context, os.path.join(output_dir, "**", "chemprop-best-*.ckpt"))
        _track_heavy_artifact_glob(context, os.path.join(output_dir, "**", "last.ckpt"))
        _, train_result = train_models.train_chemprop_model(
            curated_df=curated_df,
            target_column=target_column,
            split_indices=split_indices,
            output_dir=output_dir,
            random_state=train_random_state,
            task_type=task_type,
            model_config=chemprop_model_config,
        )
        context["trained_model_path"] = train_result.model_path
        _track_heavy_artifact(context, train_result.model_path)

        validate_contract(
            bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
            warn_only=True,
        )
        return

    use_selected = context.get("selected_ready", False)
    use_preprocessed = context.get("preprocessed_ready", False)
    skip_preprocess = use_preprocessed
    skip_feature_selection = use_selected
    skip_quality_checks = use_selected

    features_file = paths["rdkit_descriptors"]
    labels_file = paths["pic50_3class"]
    if use_selected:
        features_file = paths["selected_features"]
        if use_preprocessed and os.path.exists(paths["preprocessed_labels"]):
            labels_file = paths["preprocessed_labels"]
    elif use_preprocessed:
        features_file = paths["preprocessed_features"]
        if os.path.exists(paths["preprocessed_labels"]):
            labels_file = paths["preprocessed_labels"]
    else:
        features_file, labels_file = _resolve_feature_inputs(
            context,
            require_explicit=True,
            consumer="train",
        )

    validate_contract(
        bind_output_path(TRAIN_INPUT_FEATURES_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(TRAIN_INPUT_LABELS_CONTRACT, labels_file),
        warn_only=False,
    )
    validate_contract(
        make_target_column_contract(
            name="train_input_labels_dynamic",
            target_column=target_column,
            output_path=labels_file,
        ),
        warn_only=False,
    )

    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
        exclude_columns=_exclude_feature_columns(context),
        restrict_to_indices=_split_index_universe(context),
        drop_invalid_rows=False,
        drop_duplicate_rows=False,
        fail_on_invalid_rows=True,
        fail_on_duplicate_rows=False,
    )
    if isinstance(y, pd.DataFrame):
        y = data_preprocessing.select_target_series(y, target_column)
    if not skip_quality_checks:
        data_preprocessing.verify_data_quality(X, y)

    cv_folds = int(tuning_cfg.get("cv_folds", 5))
    search_iters = int(tuning_cfg.get("search_iters", 100)
    )
    X_val = None
    y_val = None
    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, val_idx, test_idx = partitions

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    if val_idx:
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]
    if not skip_quality_checks:
        data_preprocessing.check_data_leakage(X_train, X_test)

    train_model_config = dict(model_config)
    train_model_config["_debug_logging"] = context.get("debug_logging", False)
    if model_type == "catboost_classifier":
        expected_model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
    elif str(model_type).startswith("dl_"):
        expected_model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
    else:
        expected_model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
    _track_heavy_artifact(context, expected_model_path)

    estimator, train_result = train_models.train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        output_dir,
        random_state=train_random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        use_hpo=_as_bool(tuning_cfg.get("use_hpo", False)),
        hpo_trials=int(tuning_cfg.get("hpo_trials", 30)),
        patience=int(early_cfg.get("patience", 20)),
        task_type=task_type,
        model_config=train_model_config,
        X_val=X_val,
        y_val=y_val,
    )
    context["trained_model_path"] = train_result.model_path
    _track_heavy_artifact(context, train_result.model_path)

    validate_contract(
        bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_explain(context: dict) -> None:
    output_dir = context["run_dir"]
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]
    is_dl = model_type.startswith("dl_")
    feature_method = _resolve_feature_method(context)
    if _is_chemprop_like_model(model_type):
        skip_path = _write_explain_skip_marker(
            output_dir,
            reason="chemprop_explainability_not_supported",
            model_type=model_type,
            feature_method=feature_method,
        )
        logging.warning(
            "Explainability is not implemented for Chemprop/CheMeleon; skipping explain node and writing %s.",
            skip_path,
        )
        return
    if feature_method == "morgan":
        skip_path = _write_explain_skip_marker(
            output_dir,
            reason="morgan_explainability_disabled",
            model_type=model_type,
            feature_method=feature_method,
        )
        logging.warning(
            "Explainability is disabled for Morgan fingerprints; skipping explain node and writing %s.",
            skip_path,
        )
        return
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "explain requires split_indices from the split node. "
            "Add 'split' before 'explain' in pipeline.nodes."
        )

    model_path = context.get("trained_model_path")
    if not model_path:
        if is_dl:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        elif model_type == "catboost_classifier":
            model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        else:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")

    validate_contract(
        bind_output_path(EXPLAIN_INPUT_MODEL_CONTRACT, model_path),
        warn_only=False,
    )

    features_file = paths["rdkit_descriptors"]
    labels_file = paths["pic50_3class"]
    if context.get("selected_ready", False):
        features_file = paths["selected_features"]
        if context.get("preprocessed_ready", False) and os.path.exists(paths["preprocessed_labels"]):
            labels_file = paths["preprocessed_labels"]
    elif context.get("preprocessed_ready", False):
        features_file = paths["preprocessed_features"]
        if os.path.exists(paths["preprocessed_labels"]):
            labels_file = paths["preprocessed_labels"]
    else:
        features_file, labels_file = _resolve_feature_inputs(
            context,
            require_explicit=True,
            consumer="explain",
        )

    logging.info(
        "Explain node start: model=%s task=%s feature_method=%s features=%s labels=%s model_path=%s",
        model_type,
        context.get("task_type", "regression"),
        feature_method,
        features_file,
        labels_file,
        model_path,
    )
    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
        exclude_columns=_exclude_feature_columns(context),
        restrict_to_indices=_split_index_universe(context),
        drop_invalid_rows=False,
        drop_duplicate_rows=False,
        fail_on_invalid_rows=True,
        fail_on_duplicate_rows=False,
    )
    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, test_idx = partitions

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    logging.info(
        "Explain node data ready: model=%s X=%s X_train=%s X_test=%s y_test=%s",
        model_type,
        X.shape,
        X_train.shape,
        X_test.shape,
        y_test.shape,
    )
    logging.info("Explain node loading estimator: model=%s path=%s", model_type, model_path)
    estimator = train_models.load_model(
        model_path, 
        model_type, 
        input_dim=X_train.shape[1]
    )

    train_models.run_explainability(estimator, X_train, X_test, y_test, model_type, output_dir, task_type=context.get("task_type", "regression"))
    logging.info("Explain node complete: model=%s output_dir=%s", model_type, output_dir)

    validate_contract(
        bind_output_path(EXPLAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


NODE_REGISTRY = {
    "get_data": run_node_get_data,
    "curate": run_node_curate,
    "featurize.none": run_node_featurize_none,
    "label.normalize": run_node_label_normalize,
    "split": run_node_split,
    "label.ic50": run_node_label_ic50,
    "analyze.stats": run_node_analyze_stats,
    "analyze.eda": run_node_analyze_eda,
    "featurize.rdkit": run_node_featurize_rdkit,
    "featurize.rdkit_labeled": run_node_featurize_rdkit_labeled,
    "featurize.morgan": run_node_featurize_morgan,
    "preprocess.features": run_node_preprocess_features,
    "select.features": run_node_select_features,
    "train": run_node_train,
    "train.tdc": run_node_train_tdc,
    "explain": run_node_explain,
}

_SPLIT_REQUIRED_FOR = {
    "preprocess.features",
    "select.features",
    "train",
    "explain",
}

_SPLIT_MUST_FOLLOW = {
    "curate",
    "label.normalize",
    "label.ic50",
    "featurize.none",
    "featurize.rdkit",
    "featurize.rdkit_labeled",
    "featurize.morgan",
}


def validate_pipeline_nodes(nodes: list[str]) -> None:
    """Validate pipeline node order and required dependencies.

    This pipeline enforces a single source of truth for dataset splits: the split node.
    """
    uses_split_dependents = any(node in nodes for node in _SPLIT_REQUIRED_FOR)
    split_positions = [i for i, node in enumerate(nodes) if node == "split"]
    if len(split_positions) > 1:
        raise ValueError("Pipeline must include at most one 'split' node.")

    train_tdc_positions = [i for i, node in enumerate(nodes) if node == "train.tdc"]
    if len(train_tdc_positions) > 1:
        raise ValueError("Pipeline must include at most one 'train.tdc' node.")
    if train_tdc_positions:
        train_tdc_pos = train_tdc_positions[0]
        if "train" in nodes:
            raise ValueError("Use either 'train' or 'train.tdc' in a pipeline, not both.")
        if train_tdc_pos != len(nodes) - 1:
            raise ValueError("'train.tdc' must be the terminal node in pipeline.nodes.")

    if uses_split_dependents and not split_positions:
        raise ValueError(
            "Pipeline includes nodes that require train/val/test membership, but is missing 'split'. "
            "Add 'split' before preprocess.features/select.features/train/explain."
        )

    if not split_positions:
        return

    split_pos = split_positions[0]

    for prerequisite in _SPLIT_MUST_FOLLOW:
        if prerequisite in nodes and nodes.index(prerequisite) > split_pos:
            raise ValueError(f"'split' must come after '{prerequisite}'.")

    if uses_split_dependents:
        for dep in _SPLIT_REQUIRED_FOR:
            if dep in nodes and split_pos > nodes.index(dep):
                raise ValueError(f"'split' must come before '{dep}'.")


def run_configured_pipeline_nodes(config: dict, config_path: str) -> bool:
    pipeline = config.get("pipeline", {})
    nodes = pipeline.get("nodes")
    if not nodes:
        return False
    nodes = _canonicalize_pipeline_nodes(list(nodes))
    runtime_config = _normalize_runtime_config(config, nodes)
    validate_pipeline_nodes(nodes)
    validate_config_strict(runtime_config, nodes)

    global_config = runtime_config.get("global")
    if not global_config:
        raise ValueError("global section is required in config")
    pipeline_type = global_config.get("pipeline_type", "chembl")
    base_dir = global_config["base_dir"]
    os.makedirs(base_dir, exist_ok=True)
    run_dir = resolve_run_dir(runtime_config)
    os.makedirs(run_dir, exist_ok=True)
    run_config_path = os.path.join(run_dir, "run_config.yaml")
    run_status_path = os.path.join(run_dir, "run_status.json")
    log_level = _resolve_log_level(global_config)
    _configure_logging(run_dir, level=log_level)
    debug_logging = _as_bool(global_config.get("debug", False)) or log_level <= logging.DEBUG
    global_random_state = int(global_config.get("random_state", 42))
    artifact_retention = _resolve_artifact_retention(global_config)
    start_time = _utc_now_iso()
    config_hash = _stable_hash(runtime_config)
    config_fingerprint = _stable_hash(_hashable_config_payload(runtime_config))

    train_config = runtime_config.get("train", {}) or {}
    train_features_config = train_config.get("features", {}) if isinstance(train_config, dict) else {}
    model_config = train_config.get("model", {}) if isinstance(train_config, dict) else {}
    train_tdc_config = runtime_config.get("train_tdc", {}) or {}
    train_tdc_model_config = (
        train_tdc_config.get("model", {}) if isinstance(train_tdc_config, dict) else {}
    )

    # If the pipeline uses curated tabular descriptors directly, default to keep all columns.
    keep_all_columns = _as_bool(runtime_config.get("curate", {}).get("keep_all_columns", False))
    if "featurize.none" in nodes:
        keep_all_columns = True

    model_type = None
    if "train" in nodes:
        model_type = model_config.get("type")
    elif "train.tdc" in nodes:
        model_type = train_tdc_model_config.get("type")

    context = {
        "config_path": config_path,
        "base_dir": base_dir,
        "paths": build_paths(base_dir),
        "pipeline_type": pipeline_type,
        "task_type": global_config.get("task_type", "regression"),
        "active_threshold": global_config["thresholds"]["active"],
        "inactive_threshold": global_config["thresholds"]["inactive"],
        "target_column": global_config.get("target_column", "pIC50"),
        "model_type": model_type,
        "global_random_state": global_random_state,
        "get_data_config": runtime_config.get("get_data", {}),
        "curate_config": runtime_config.get("curate", {}),
        "preprocess_config": runtime_config.get("preprocess", {}),
        "split_config": runtime_config.get("split", {}),
        "featurize_config": runtime_config.get("featurize", {}),
        "label_config": runtime_config.get("label", {}),
        "analyze_config": runtime_config.get("analyze", {}),
        "train_config": train_config,
        "model_config": model_config,
        "train_tdc_config": train_tdc_config,
        "categorical_features": _as_str_list(train_features_config.get("categorical_features", [])),
        "keep_all_columns": keep_all_columns,
        "source": runtime_config.get("get_data", {}).get("source", {}),
        "run_dir": run_dir,
        "debug_logging": debug_logging,
        "config_hash": config_hash,
        "config_fingerprint": config_fingerprint,
        "run_start_epoch": time.time(),
        "feature_method": None,
        "pipeline_feature_input": (runtime_config.get("pipeline", {}) or {}).get("feature_input"),
        "pipeline_nodes": list(nodes),
        "artifact_retention": artifact_retention,
    }

    with open(run_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, sort_keys=False)

    _write_json_atomic(
        run_status_path,
        {
            "status": "running",
            "start_time": start_time,
            "config_path": config_path,
            "run_dir": run_dir,
            "base_dir": base_dir,
            "pipeline_nodes": list(nodes),
            "config_hash": config_hash,
            "config_fingerprint": config_fingerprint,
            "git_sha": _git_sha(),
            "artifact_retention": artifact_retention,
        },
    )

    failed_node = None
    try:
        for node_name in nodes:
            failed_node = node_name
            node_fn = NODE_REGISTRY.get(node_name)
            if not node_fn:
                raise ValueError(f"Unknown pipeline node: {node_name}")
            node_fn(context)
    except BaseException as exc:
        _apply_artifact_retention(context, run_status="failed")
        artifact_retention_summary = context.get("artifact_retention_summary")
        _write_json_atomic(
            run_status_path,
            {
                "status": "failed",
                "start_time": start_time,
                "end_time": _utc_now_iso(),
                "config_path": config_path,
                "run_dir": run_dir,
                "base_dir": base_dir,
                "pipeline_nodes": list(nodes),
                "failed_node": failed_node,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "config_hash": config_hash,
                "config_fingerprint": config_fingerprint,
                "git_sha": _git_sha(),
                "artifact_retention": artifact_retention,
                "artifact_retention_summary": artifact_retention_summary,
            },
        )
        raise

    _apply_artifact_retention(context, run_status="success")
    artifact_retention_summary = context.get("artifact_retention_summary")
    _write_json_atomic(
        run_status_path,
        {
            "status": "success",
            "start_time": start_time,
            "end_time": _utc_now_iso(),
            "config_path": config_path,
            "run_dir": run_dir,
            "base_dir": base_dir,
            "pipeline_nodes": list(nodes),
            "config_hash": config_hash,
            "config_fingerprint": config_fingerprint,
            "git_sha": _git_sha(),
            "artifact_retention": artifact_retention,
            "artifact_retention_summary": artifact_retention_summary,
        },
    )
    return True


def main() -> int:
    config_path = os.environ.get("CHEMLFLOW_CONFIG", "config/config.chembl.yaml")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config: {config_path}: {exc}", file=sys.stderr)
        return 1

    if run_configured_pipeline_nodes(config, config_path):
        # In some environments, native libraries loaded by the Chemprop stack can abort
        # during interpreter teardown after successful training/artifact writes. In pytest
        # subprocess runs, force a hard exit on success to avoid false-negative e2e failures.
        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            and _is_chemprop_like_model((config.get("train", {}) or {}).get("model", {}).get("type"))
        ):
            logging.shutdown()
            os._exit(0)
        return 0

    print(
        "Missing required pipeline definition. Add pipeline.nodes to the config and run the node-based pipeline. "
        "Splitting is performed only by the 'split' node.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
