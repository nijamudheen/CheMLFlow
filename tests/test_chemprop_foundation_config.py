import pandas as pd
import pytest

from MLModels.train_models import (
    _resolve_chemprop_predictor_ctor,
    _resolve_chemprop_foundation_config,
    _resolve_chemprop_split_positions,
    train_chemprop_model,
)


def test_resolve_chemprop_foundation_defaults() -> None:
    foundation, checkpoint, freeze_encoder = _resolve_chemprop_foundation_config({})
    assert foundation == "none"
    assert checkpoint is None
    assert freeze_encoder is False


def test_resolve_chemprop_foundation_invalid_mode() -> None:
    with pytest.raises(ValueError, match="model.foundation"):
        _resolve_chemprop_foundation_config({"foundation": "bad_mode"})


def test_resolve_chemprop_foundation_requires_checkpoint_for_chemeleon() -> None:
    with pytest.raises(ValueError, match="foundation_checkpoint"):
        _resolve_chemprop_foundation_config({"foundation": "chemeleon"})


def test_resolve_chemprop_foundation_requires_existing_checkpoint(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_chemprop_foundation_config(
            {"foundation": "chemeleon", "foundation_checkpoint": str(missing)}
        )


def test_resolve_chemprop_foundation_with_checkpoint_and_freeze(tmp_path) -> None:
    ckpt = tmp_path / "chemeleon_mp.pt"
    ckpt.write_bytes(b"placeholder")
    foundation, checkpoint, freeze_encoder = _resolve_chemprop_foundation_config(
        {
            "foundation": "chemeleon",
            "foundation_checkpoint": str(ckpt),
            "freeze_encoder": True,
        }
    )
    assert foundation == "chemeleon"
    assert checkpoint == str(ckpt)
    assert freeze_encoder is True


def test_resolve_chemprop_foundation_freeze_requires_chemeleon() -> None:
    with pytest.raises(ValueError, match="freeze_encoder"):
        _resolve_chemprop_foundation_config({"freeze_encoder": True})


def test_resolve_chemprop_split_positions_maps_row_ids() -> None:
    curated = pd.DataFrame(
        {
            "__row_index": [10, 11, 12],
            "canonical_smiles": ["CC", "CCC", "CCCC"],
            "label": [0, 1, 0],
        }
    )
    splits = {"train": [10, 11], "val": [12], "test": []}
    tr_idx, va_idx, te_idx, row_ids = _resolve_chemprop_split_positions(curated, splits)
    assert tr_idx == [0, 1]
    assert va_idx == []
    assert te_idx == [2]
    assert row_ids == [10, 11, 12]


def test_resolve_chemprop_split_positions_rejects_legacy_positions_by_default() -> None:
    curated = pd.DataFrame(
        {
            "__row_index": [100, 101, 102],
            "canonical_smiles": ["CC", "CCC", "CCCC"],
            "label": [0, 1, 0],
        }
    )
    splits = {"train": [0], "val": [1], "test": [2]}
    with pytest.raises(ValueError, match="allow_legacy_split_positions"):
        _resolve_chemprop_split_positions(curated, splits)


def test_resolve_chemprop_split_positions_allows_legacy_positions_when_opted_in() -> None:
    curated = pd.DataFrame(
        {
            "__row_index": [100, 101, 102],
            "canonical_smiles": ["CC", "CCC", "CCCC"],
            "label": [0, 1, 0],
        }
    )
    splits = {"train": [0], "val": [1], "test": [2]}
    tr_idx, va_idx, te_idx, row_ids = _resolve_chemprop_split_positions(
        curated,
        splits,
        allow_legacy_positions=True,
    )
    assert tr_idx == [0]
    assert va_idx == [1]
    assert te_idx == [2]
    assert row_ids == [100, 101, 102]


def test_resolve_chemprop_split_positions_rejects_unknown_row_ids() -> None:
    curated = pd.DataFrame(
        {
            "__row_index": [100, 101, 102],
            "canonical_smiles": ["CC", "CCC", "CCCC"],
            "label": [0, 1, 0],
        }
    )
    splits = {"train": [999], "val": [], "test": [101]}
    with pytest.raises(ValueError, match="do not match curated row IDs"):
        _resolve_chemprop_split_positions(curated, splits)


def test_resolve_chemprop_predictor_ctor_requires_regression_ffn() -> None:
    class _NNModule:
        BinaryClassificationFFN = object

    with pytest.raises(ValueError, match="RegressionFFN"):
        _resolve_chemprop_predictor_ctor(_NNModule, "regression")


def test_train_chemprop_model_regression_rejects_non_numeric_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("MLModels.train_models._require_chemprop", lambda: None)

    curated = pd.DataFrame(
        {
            "canonical_smiles": ["CC", "CCC", "CCCC"],
            "target": [1.2, "bad", 3.4],
        }
    )
    splits = {"train": [0], "val": [1], "test": [2]}

    with pytest.raises(ValueError, match="numeric target values"):
        train_chemprop_model(
            curated_df=curated,
            target_column="target",
            split_indices=splits,
            output_dir=str(tmp_path),
            random_state=42,
            task_type="regression",
            model_config={},
        )
