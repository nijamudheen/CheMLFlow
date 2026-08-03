from __future__ import annotations

import pytest

from utilities.config_validation import ConfigValidationError, collect_config_issues, validate_config_strict


def _base_config(nodes: list[str]) -> dict:
    return {
        "global": {
            "pipeline_type": "qm9",
            "base_dir": "data/qm9",
            "thresholds": {"active": 1, "inactive": 2},
        },
        "pipeline": {"nodes": nodes},
    }


def test_strict_rejects_unknown_top_level_block() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["mystery"] = {"foo": "bar"}
    with pytest.raises(ConfigValidationError, match="CFG_UNKNOWN_TOP_LEVEL_BLOCK"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_invalid_global_artifact_retention() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["global"]["artifact_retention"] = "tiny"
    with pytest.raises(ConfigValidationError, match="CFG_GLOBAL_ARTIFACT_RETENTION_INVALID"):
        validate_config_strict(cfg, ["train"])


def test_strict_allows_analyze_block_for_analyze_node() -> None:
    cfg = _base_config(["get_data", "analyze.eda"])
    cfg["global"]["task_type"] = "classification"
    cfg["get_data"] = {"data_source": "local_csv", "source": {"path": "local_data/example.csv"}}
    cfg["analyze"] = {"eda": {"include": {"overview": True}}}
    validate_config_strict(cfg, ["get_data", "analyze.eda"])


def test_strict_allows_valid_get_data_sample() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "sample": {"fraction": 0.1, "seed": 42, "strategy": "random"},
    }

    validate_config_strict(cfg, ["get_data"])


def test_strict_rejects_invalid_get_data_sample_fraction() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "sample": {"fraction": 1.5, "seed": 42, "strategy": "random"},
    }

    issues = collect_config_issues(cfg, ["get_data"])

    assert any(
        issue.code == "CFG_GET_DATA_SAMPLE_INVALID"
        and issue.path == "get_data.sample.fraction"
        for issue in issues
    )


def test_strict_rejects_invalid_get_data_sample_strategy() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "sample": {"fraction": 0.1, "seed": 42, "strategy": "quantile"},
    }

    issues = collect_config_issues(cfg, ["get_data"])

    assert any(
        issue.code == "CFG_GET_DATA_SAMPLE_INVALID"
        and issue.path == "get_data.sample.strategy"
        for issue in issues
    )


def test_strict_rejects_unknown_get_data_sample_key() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "sample": {"fraction": 0.1, "seed": 42, "stratgey": "stratified"},
    }

    issues = collect_config_issues(cfg, ["get_data"])

    assert any(
        issue.code == "CFG_GET_DATA_SAMPLE_UNKNOWN_KEY"
        and issue.path == "get_data.sample"
        for issue in issues
    )


def test_strict_rejects_non_integral_get_data_sample_seed() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "sample": {"fraction": 0.1, "seed": 42.9, "strategy": "random"},
    }

    issues = collect_config_issues(cfg, ["get_data"])

    assert any(
        issue.code == "CFG_GET_DATA_SAMPLE_INVALID"
        and issue.path == "get_data.sample.seed"
        for issue in issues
    )


def test_strict_rejects_get_data_sample_with_max_rows() -> None:
    cfg = _base_config(["get_data"])
    cfg["get_data"] = {
        "data_source": "local_csv",
        "source": {"path": "local_data/example.csv"},
        "max_rows": 100,
        "sample": {"fraction": 0.1, "seed": 42, "strategy": "random"},
    }

    issues = collect_config_issues(cfg, ["get_data"])

    assert any(issue.code == "CFG_GET_DATA_SAMPLE_CONFLICT" for issue in issues)


def test_strict_rejects_block_not_in_pipeline() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["split"] = {"strategy": "random"}
    with pytest.raises(ConfigValidationError, match="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_configless_featurize_none_block() -> None:
    cfg = _base_config(["featurize.none"])
    cfg["featurize"] = {"radius": 2}
    with pytest.raises(ConfigValidationError, match="CFG_CONFIGLESS_NODE_HAS_BLOCK"):
        validate_config_strict(cfg, ["featurize.none"])


def test_configless_featurize_none_allows_shared_featurize_block_with_morgan() -> None:
    cfg = _base_config(["featurize.none", "featurize.morgan"])
    cfg["featurize"] = {"radius": 2, "n_bits": 1024}
    validate_config_strict(cfg, ["featurize.none", "featurize.morgan"])


def test_strict_allows_legacy_use_curated_features_alias_node() -> None:
    cfg = _base_config(["use.curated_features"])
    validate_config_strict(cfg, ["use.curated_features"])


def test_strict_requires_train_model_type() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {}}
    with pytest.raises(ConfigValidationError, match="CFG_MISSING_TRAIN_MODEL_TYPE"):
        validate_config_strict(cfg, ["train"])


@pytest.mark.parametrize(
    ("tuning_cfg", "path"),
    [
        ({"method": "train_cv"}, "train.tuning.method"),
        ({"use_hpo": True}, "train.tuning.use_hpo"),
    ],
)
def test_strict_rejects_child_level_train_hpo(tuning_cfg: dict, path: str) -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "random_forest"}, "tuning": tuning_cfg}

    issues = collect_config_issues(cfg, ["train"])

    assert any(
        issue.code == "CFG_CHILD_LEVEL_HPO_UNSUPPORTED" and issue.path == path
        for issue in issues
    )


def test_strict_allows_train_optuna() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {
        "model": {"type": "random_forest"},
        "tuning": {
            "method": "optuna",
            "n_trials": 2,
            "metric": "val_auc",
            "params": {
                "max_depth": {"type": "categorical", "choices": [2, 4]},
            },
        },
    }

    issues = collect_config_issues(cfg, ["train"])

    assert not any(
        issue.code == "CFG_CHILD_LEVEL_HPO_UNSUPPORTED"
        and issue.path == "train.tuning.method"
        for issue in issues
    )


def test_strict_allows_timeseries_optuna() -> None:
    cfg = _base_config(["train.timeseries"])
    cfg["train"] = {
        "model": {"type": "dl_adaptive_nvar"},
        "tuning": {"method": "optuna"},
    }
    cfg["split"] = {"warmup_len": 5, "train_len": 30, "val_len": 20, "test_len": 20}

    issues = collect_config_issues(cfg, ["train.timeseries"])

    assert not any(
        issue.code == "CFG_CHILD_LEVEL_HPO_UNSUPPORTED"
        and issue.path == "train.tuning.method"
        for issue in issues
    )


def test_strict_rejects_child_level_tdc_hpo() -> None:
    cfg = _base_config(["train.tdc"])
    cfg["train_tdc"] = {
        "model": {"type": "catboost_classifier"},
        "tuning": {"use_hpo": True},
    }

    issues = collect_config_issues(cfg, ["train.tdc"])

    assert any(
        issue.code == "CFG_CHILD_LEVEL_HPO_UNSUPPORTED"
        and issue.path == "train_tdc.tuning.use_hpo"
        for issue in issues
    )


def test_strict_rejects_legacy_preprocess_keys() -> None:
    cfg = _base_config(["preprocess.features"])
    cfg["preprocess"] = {"keep_all_columns": True, "exclude_columns": ["A"]}
    issues = collect_config_issues(cfg, ["preprocess.features"])
    codes = {issue.code for issue in issues}
    assert "CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN" in codes


def test_strict_requires_feature_input_node_for_preprocess() -> None:
    cfg = _base_config(["split", "preprocess.features"])
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {}
    issues = collect_config_issues(cfg, ["split", "preprocess.features"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" in codes


def test_strict_requires_feature_input_node_for_non_chemprop_train() -> None:
    cfg = _base_config(["split", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}
    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" in codes


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_strict_allows_train_without_feature_node_for_chemprop_like(model_type: str) -> None:
    cfg = _base_config(["split", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": model_type}}
    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" not in codes
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" not in codes


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_strict_allows_noop_preprocess_without_feature_node_for_chemprop_like(model_type: str) -> None:
    cfg = _base_config(["split", "preprocess.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": model_type}}

    issues = collect_config_issues(cfg, ["split", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" not in codes


def test_strict_rejects_invalid_preprocess_scaler() -> None:
    cfg = _base_config(["split", "featurize.rdkit", "preprocess.features", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "banana"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, ["split", "featurize.rdkit", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_PREPROCESS_SCALER_INVALID" in codes


def test_strict_allows_explicit_preprocess_clip_mapping() -> None:
    nodes = ["split", "featurize.rdkit", "preprocess.features", "train"]
    cfg = _base_config(nodes)
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {
        "scaler": "standard",
        "clip": {"min": -6, "max": 6},
    }
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, nodes)

    assert not any(issue.code == "CFG_PREPROCESS_CLIP_INVALID" for issue in issues)


@pytest.mark.parametrize(
    "clip",
    [
        [-6, 6],
        {"min": -6},
        {"min": -6, "max": 6, "unexpected": 1},
        {"min": 6, "max": -6},
        {"min": "negative", "max": 6},
        {"min": float("-inf"), "max": 6},
    ],
)
def test_strict_rejects_invalid_preprocess_clip(clip: object) -> None:
    nodes = ["split", "featurize.rdkit", "preprocess.features", "train"]
    cfg = _base_config(nodes)
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "standard", "clip": clip}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, nodes)

    assert any(issue.code == "CFG_PREPROCESS_CLIP_INVALID" for issue in issues)


def test_strict_rejects_chemprop_with_explicit_tabular_featurizer() -> None:
    cfg = _base_config(["split", "featurize.rdkit", "preprocess.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "featurize.rdkit", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_PIPELINE_FEATURE_INPUT_MISMATCH" in codes
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" in codes


def test_strict_rejects_chemprop_like_select_features_branch() -> None:
    cfg = _base_config(["split", "preprocess.features", "select.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "preprocess.features", "select.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMPROP_PREPROCESS_UNSUPPORTED" in codes


def test_strict_requires_chemeleon_checkpoint() -> None:
    cfg = _base_config(["split", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "chemeleon"}}

    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMELEON_CHECKPOINT_REQUIRED" in codes


def test_strict_requires_smiles_native_for_chemprop_like_models() -> None:
    cfg = _base_config(["split", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" in codes


def test_strict_requires_preprocess_for_select_features() -> None:
    cfg = _base_config(["split", "select.features", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, ["split", "select.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_SELECT_REQUIRES_PREPROCESS" in codes


def test_strict_rejects_classification_only_model_for_regression_task() -> None:
    cfg = _base_config(["featurize.none", "split", "train"])
    cfg["global"]["task_type"] = "regression"
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "catboost_classifier"}}

    issues = collect_config_issues(cfg, ["featurize.none", "split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_MODEL_TASK_MISMATCH" in codes


def test_strict_rejects_chembl_smiles_native_chemprop_runtime_branch() -> None:
    cfg = _base_config(["get_data", "split", "train"])
    cfg["global"]["task_type"] = "regression"
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["get_data"] = {"data_source": "chembl", "source": {"target_name": "IC50"}}
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["get_data", "split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NOT_SUPPORTED" in codes
    assert "CFG_MODEL_NOT_SUPPORTED_FOR_PROFILE" in codes


def test_strict_allows_chembl_target_pin_with_curate_row_filters() -> None:
    nodes = ["get_data", "curate", "label.ic50", "featurize.rdkit", "split", "train"]
    cfg = _base_config(nodes)
    cfg["global"]["task_type"] = "regression"
    cfg["global"]["target_column"] = "pIC50"
    cfg["get_data"] = {
        "data_source": "chembl",
        "source": {"target_chembl_id": "CHEMBL3885651"},
    }
    cfg["curate"] = {
        "row_filters": {
            "standard_type": ["IC50"],
            "standard_units": ["nM"],
            "standard_relation": ["="],
        }
    }
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    validate_config_strict(cfg, nodes)


def test_strict_rejects_non_mapping_curate_row_filters() -> None:
    nodes = ["curate", "split", "featurize.rdkit", "train"]
    cfg = _base_config(nodes)
    cfg["curate"] = {"row_filters": ["not", "a", "mapping"]}
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, nodes)
    codes = {issue.code for issue in issues}
    assert "CFG_CURATE_ROW_FILTERS_INVALID" in codes


def test_strict_allows_rdkit_labeled_as_rdkit_profile_branch() -> None:
    cfg = _base_config(["get_data", "featurize.rdkit_labeled", "split", "train"])
    cfg["global"]["task_type"] = "regression"
    cfg["get_data"] = {"data_source": "local_csv", "source": {"path": "data.csv"}}
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, ["get_data", "featurize.rdkit_labeled", "split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NOT_SUPPORTED" not in codes


def test_strict_allows_ecfp4_rdkit_as_local_csv_feature_branch() -> None:
    nodes = ["get_data", "featurize.ecfp4_rdkit", "split", "train"]
    cfg = _base_config(nodes)
    cfg["global"]["task_type"] = "classification"
    cfg["pipeline"]["feature_input"] = "featurize.ecfp4_rdkit"
    cfg["get_data"] = {"data_source": "local_csv", "source": {"path": "data.csv"}}
    cfg["featurize"] = {"radius": 2, "n_bits": 2048}
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, nodes)
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NOT_SUPPORTED" not in codes
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" not in codes
    assert "CFG_PIPELINE_FEATURE_INPUT_MISMATCH" not in codes


def test_strict_requires_chemeleon_fp_checkpoint() -> None:
    nodes = ["get_data", "curate", "featurize.chemeleon_fp", "split", "train"]
    cfg = _base_config(nodes)
    cfg["global"]["task_type"] = "classification"
    cfg["global"]["target_column"] = "label"
    cfg["pipeline"]["feature_input"] = "featurize.chemeleon_fp"
    cfg["get_data"] = {"data_source": "local_csv", "source": {"path": "data.csv"}}
    cfg["curate"] = {"smiles_column": "SMILES", "properties": "label"}
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "tabpfn"}}

    issues = collect_config_issues(cfg, nodes)

    assert any(issue.code == "CFG_CHEMELEON_FP_CHECKPOINT_REQUIRED" for issue in issues)


def test_strict_allows_tabpfn_with_chemeleon_fp_and_clipped_standardization() -> None:
    nodes = [
        "get_data",
        "curate",
        "featurize.chemeleon_fp",
        "split",
        "preprocess.features",
        "train",
    ]
    cfg = _base_config(nodes)
    cfg["global"]["task_type"] = "classification"
    cfg["global"]["target_column"] = "label"
    cfg["pipeline"]["feature_input"] = "featurize.chemeleon_fp"
    cfg["get_data"] = {"data_source": "local_csv", "source": {"path": "data.csv"}}
    cfg["curate"] = {"smiles_column": "SMILES", "properties": "label"}
    cfg["featurize"] = {"checkpoint": "models/chemeleon_mp.pt"}
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "standard", "clip": {"min": -6, "max": 6}}
    cfg["train"] = {"model": {"type": "tabpfn"}}

    issues = collect_config_issues(cfg, nodes)
    codes = {issue.code for issue in issues}

    assert "CFG_CHEMELEON_FP_CHECKPOINT_REQUIRED" not in codes
    assert "CFG_FEATURE_INPUT_NOT_SUPPORTED" not in codes
    assert "CFG_MODEL_NOT_SUPPORTED_FOR_PROFILE" not in codes
    assert "CFG_MODEL_TASK_MISMATCH" not in codes


def test_strict_rejects_tdc_split_profile_shape() -> None:
    cfg = _base_config(["get_data", "split", "train.tdc"])
    cfg["global"]["task_type"] = "classification"
    cfg["get_data"] = {"data_source": "tdc", "source": {}}
    cfg["split"] = {"strategy": "random"}
    cfg["train_tdc"] = {"model": {"type": "catboost_classifier"}}

    issues = collect_config_issues(cfg, ["get_data", "split", "train.tdc"])
    codes = {issue.code for issue in issues}
    assert "CFG_PROFILE_NODE_UNSUPPORTED" in codes


def test_strict_rejects_tdc_profile_wrong_model_type() -> None:
    cfg = _base_config(["get_data", "train.tdc"])
    cfg["global"]["task_type"] = "classification"
    cfg["get_data"] = {"data_source": "tdc", "source": {}}
    cfg["train_tdc"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, ["get_data", "train.tdc"])
    codes = {issue.code for issue in issues}
    assert "CFG_MODEL_NOT_SUPPORTED_FOR_PROFILE" in codes


# ---------------------------------------------------------------------------
# train.timeseries strict validation (requires train.model.type + split fields)
# ---------------------------------------------------------------------------


def _ts_nodes() -> list[str]:
    return ["get_data", "train.timeseries"]


def test_timeseries_missing_train_and_split_blocks_flagged() -> None:
    cfg = {
        "global": {"pipeline_type": "timeseries", "base_dir": "data/ts",
                   "thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": _ts_nodes(), "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "data/x.npy"}},
    }
    codes = {i.code for i in collect_config_issues(cfg, _ts_nodes())}
    assert "CFG_MISSING_BLOCK_FOR_NODE" in codes  # train and/or split missing


def test_timeseries_missing_model_type_and_split_field_flagged() -> None:
    cfg = {
        "global": {"pipeline_type": "timeseries", "base_dir": "data/ts",
                   "thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": _ts_nodes(), "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "data/x.npy"}},
        "train": {"model": {}},
        "split": {"warmup_len": 100, "train_len": 500, "val_len": 100},  # test_len missing
    }
    codes = {i.code for i in collect_config_issues(cfg, _ts_nodes())}
    assert "CFG_MISSING_TRAIN_MODEL_TYPE" in codes
    assert "CFG_MISSING_SPLIT_FIELD" in codes


def test_timeseries_nonpositive_split_field_flagged() -> None:
    cfg = {
        "global": {"pipeline_type": "timeseries", "base_dir": "data/ts",
                   "thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": _ts_nodes(), "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "data/x.npy"}},
        "train": {"model": {"type": "dl_adaptive_nvar"}},
        "split": {"warmup_len": 100, "train_len": 0, "val_len": 100, "test_len": 100},
    }
    codes = {i.code for i in collect_config_issues(cfg, _ts_nodes())}
    assert "CFG_INVALID_SPLIT_FIELD" in codes


def test_timeseries_valid_config_has_no_ts_issues() -> None:
    cfg = {
        "global": {"pipeline_type": "timeseries", "base_dir": "data/ts",
                   "thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": _ts_nodes(), "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "data/x.npy"}},
        "train": {"model": {"type": "dl_adaptive_nvar"}},
        "split": {"warmup_len": 100, "train_len": 500, "val_len": 100, "test_len": 100},
    }
    codes = {i.code for i in collect_config_issues(cfg, _ts_nodes())}
    for bad in ("CFG_MISSING_TRAIN_MODEL_TYPE", "CFG_MISSING_SPLIT_FIELD",
                "CFG_INVALID_SPLIT_FIELD"):
        assert bad not in codes


# ---------------------------------------------------------------------------
# train.timeseries split validation must MATCH parse_split_config semantics
# ---------------------------------------------------------------------------


def _ts_cfg(split: dict) -> dict:
    return {
        "global": {"pipeline_type": "timeseries", "base_dir": "data/ts",
                   "thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": ["get_data", "train.timeseries"], "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "data/x.npy"}},
        "train": {"model": {"type": "dl_adaptive_nvar"}},
        "split": split,
    }


def _split_codes(split: dict) -> set:
    nodes = ["get_data", "train.timeseries"]
    return {i.code for i in collect_config_issues(_ts_cfg(split), nodes) if "SPLIT" in i.code}


def test_timeseries_warmup_zero_is_accepted() -> None:
    # parse_split_config allows warmup_len == 0; validator must too.
    assert _split_codes({"warmup_len": 0, "train_len": 500, "val_len": 100, "test_len": 100}) == set()


def test_timeseries_zero_test_with_val_is_accepted() -> None:
    assert _split_codes({"warmup_len": 50, "train_len": 500, "val_len": 100, "test_len": 0}) == set()


def test_timeseries_zero_val_with_test_is_accepted() -> None:
    assert _split_codes({"warmup_len": 50, "train_len": 500, "val_len": 0, "test_len": 100}) == set()


def test_timeseries_both_eval_segments_zero_is_rejected() -> None:
    assert "CFG_INVALID_SPLIT_FIELD" in _split_codes(
        {"warmup_len": 50, "train_len": 500, "val_len": 0, "test_len": 0}
    )


def test_timeseries_train_len_zero_is_rejected() -> None:
    assert "CFG_INVALID_SPLIT_FIELD" in _split_codes(
        {"warmup_len": 50, "train_len": 0, "val_len": 100, "test_len": 100}
    )


def test_timeseries_negative_split_is_rejected() -> None:
    assert "CFG_INVALID_SPLIT_FIELD" in _split_codes(
        {"warmup_len": -1, "train_len": 500, "val_len": 100, "test_len": 100}
    )


def test_timeseries_validator_matches_parser_on_edge_cases() -> None:
    """The strict validator must accept exactly what parse_split_config accepts."""
    from utilities.timeseries_io import parse_split_config

    edge_cases = [
        {"warmup_len": 0, "train_len": 500, "val_len": 100, "test_len": 100},
        {"warmup_len": 50, "train_len": 500, "val_len": 100, "test_len": 0},
        {"warmup_len": 50, "train_len": 500, "val_len": 0, "test_len": 100},
        {"warmup_len": 50, "train_len": 500, "val_len": 0, "test_len": 0},
        {"warmup_len": 50, "train_len": 0, "val_len": 100, "test_len": 100},
    ]
    for split in edge_cases:
        validator_ok = _split_codes(split) == set()
        try:
            parse_split_config(split)
            parser_ok = True
        except ValueError:
            parser_ok = False
        assert validator_ok == parser_ok, f"validator/parser disagree on {split}"


# ---------------------------------------------------------------------------
# time-series source/model must require the train.timeseries node
# ---------------------------------------------------------------------------


def test_timeseries_source_with_tabular_train_rejected() -> None:
    cfg = {
        "global": {"thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": ["get_data", "train"], "feature_input": "none"},
        "get_data": {"data_source": "local_npy", "source": {"path": "d.npy"}},
        "train": {"model": {"type": "random_forest_regressor"}},
    }
    codes = {i.code for i in collect_config_issues(cfg, ["get_data", "train"])}
    assert "CFG_TIMESERIES_SOURCE_REQUIRES_TS_NODE" in codes


def test_timeseries_model_under_tabular_train_rejected() -> None:
    cfg = {
        "global": {"thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": ["get_data", "train"], "feature_input": "none"},
        "get_data": {"data_source": "local_csv", "source": {"path": "d.csv"}},
        "train": {"model": {"type": "dl_adaptive_nvar"}},
    }
    codes = {i.code for i in collect_config_issues(cfg, ["get_data", "train"])}
    assert "CFG_TIMESERIES_MODEL_REQUIRES_TS_NODE" in codes


def test_tabular_pipeline_has_no_timeseries_issues() -> None:
    cfg = {
        "global": {"thresholds": {"active": 1, "inactive": 2}},
        "pipeline": {"nodes": ["get_data", "train"], "feature_input": "none"},
        "get_data": {"data_source": "local_csv", "source": {"path": "d.csv"}},
        "train": {"model": {"type": "random_forest_regressor"}},
    }
    codes = {i.code for i in collect_config_issues(cfg, ["get_data", "train"])}
    assert not any("TIMESERIES" in c for c in codes)
