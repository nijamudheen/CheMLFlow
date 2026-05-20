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
