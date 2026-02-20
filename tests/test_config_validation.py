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


def test_strict_rejects_block_not_in_pipeline() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["split"] = {"strategy": "random"}
    with pytest.raises(ConfigValidationError, match="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_configless_use_block() -> None:
    cfg = _base_config(["use.curated_features"])
    cfg["use"] = {}
    with pytest.raises(ConfigValidationError, match="CFG_CONFIGLESS_NODE_HAS_BLOCK"):
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
