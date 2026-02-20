from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    hint: str | None = None


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        details = []
        for issue in issues:
            hint = f" Hint: {issue.hint}" if issue.hint else ""
            details.append(f"- [{issue.code}] {issue.path}: {issue.message}{hint}")
        msg = "Config validation failed:\n" + "\n".join(details)
        super().__init__(msg)


_NODE_TO_BLOCK = {
    "get_data": "get_data",
    "curate": "curate",
    "label.normalize": "label",
    "split": "split",
    "featurize.lipinski": "featurize",
    "featurize.rdkit": "featurize",
    "featurize.rdkit_labeled": "featurize",
    "featurize.morgan": "featurize",
    "preprocess.features": "preprocess",
    "select.features": "preprocess",
    "train": "train",
    "train.tdc": "train_tdc",
}

_CONFIGLESS_NODE_TO_BLOCK = {
    "use.curated_features": "use",
}

_ALWAYS_ALLOWED_BLOCKS = {"global", "pipeline"}
_KNOWN_TOP_LEVEL_BLOCKS = {
    *_ALWAYS_ALLOWED_BLOCKS,
    "get_data",
    "curate",
    "label",
    "split",
    "featurize",
    "preprocess",
    "train",
    "train_tdc",
    "use",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def collect_config_issues(config: dict[str, Any], nodes: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    blocks_present = set(config.keys())

    allowed_blocks = set(_ALWAYS_ALLOWED_BLOCKS)
    for node in nodes:
        block = _NODE_TO_BLOCK.get(node)
        if block:
            allowed_blocks.add(block)

    # Top-level block validity.
    for block in blocks_present:
        if block == "model":
            issues.append(
                ValidationIssue(
                    code="CFG_LEGACY_MODEL_BLOCK_FORBIDDEN",
                    path="model",
                    message="Top-level model block is no longer supported.",
                    hint="Move model settings to train.model.*",
                )
            )
            continue
        if block not in _KNOWN_TOP_LEVEL_BLOCKS:
            issues.append(
                ValidationIssue(
                    code="CFG_UNKNOWN_TOP_LEVEL_BLOCK",
                    path=block,
                    message="Unknown top-level config block.",
                )
            )
            continue
        if block not in allowed_blocks:
            if block == "use" and "use.curated_features" in nodes:
                continue
            issues.append(
                ValidationIssue(
                    code="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE",
                    path=block,
                    message=f"Block '{block}' is present but no corresponding node is in pipeline.nodes.",
                )
            )

    # Configless-node specific checks.
    if "use.curated_features" in nodes and "use" in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_CONFIGLESS_NODE_HAS_BLOCK",
                path="use",
                message="Node use.curated_features is configless and does not accept a top-level use block.",
            )
        )

    # Required blocks for specific nodes.
    if "train" in nodes and "train" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train",
                message="Pipeline contains train node but train block is missing.",
            )
        )
    if "train.tdc" in nodes and "train_tdc" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train_tdc",
                message="Pipeline contains train.tdc node but train_tdc block is missing.",
            )
        )

    if "train" in blocks_present and not isinstance(config.get("train"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train",
                message="train block must be a mapping/object.",
            )
        )
    train_cfg = _as_dict(config.get("train"))
    if "train" in nodes:
        model_value = train_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train.model",
                    message="train.model block is required for train node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train.model.type",
                        message="train.model.type is required.",
                    )
                )

    if "train_tdc" in blocks_present and not isinstance(config.get("train_tdc"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train_tdc",
                message="train_tdc block must be a mapping/object.",
            )
        )
    train_tdc_cfg = _as_dict(config.get("train_tdc"))
    if "train.tdc" in nodes:
        model_value = train_tdc_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train_tdc.model",
                    message="train_tdc.model block is required for train.tdc node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train_tdc.model.type",
                        message="train_tdc.model.type is required.",
                    )
                )

    # Legacy preprocess keys that leaked into other nodes.
    preprocess_cfg = _as_dict(config.get("preprocess"))
    if "keep_all_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.keep_all_columns",
                message="preprocess.keep_all_columns is no longer supported.",
                hint="Move to curate.keep_all_columns",
            )
        )
    if "exclude_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.exclude_columns",
                message="preprocess.exclude_columns is no longer supported.",
                hint="Move to train.features.exclude_columns",
            )
        )

    return issues


def validate_config_strict(config: dict[str, Any], nodes: list[str]) -> None:
    issues = collect_config_issues(config, nodes)
    if issues:
        raise ConfigValidationError(issues)
