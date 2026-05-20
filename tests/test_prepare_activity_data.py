import pandas as pd
import pytest

from utilities.prepareActivityData import DataPreparer, _normalize_dedupe_strategy


def test_normalize_dedupe_strategy_aliases() -> None:
    assert _normalize_dedupe_strategy("keep_first") == "first"
    assert _normalize_dedupe_strategy("keep_last") == "last"
    assert _normalize_dedupe_strategy("majority") == "majority"
    assert _normalize_dedupe_strategy(None) is None


def test_normalize_dedupe_strategy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported dedupe_strategy"):
        _normalize_dedupe_strategy("unsupported_value")


def test_handle_missing_data_keep_last(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCO", "CCN"],
            "Activity": [1, 2, 3],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(
        str(raw),
        str(preprocessed),
        str(curated),
        keep_all_columns=False,
    )
    preparer.handle_missing_data_and_duplicates(
        properties_of_interest=["Activity"],
        dedupe_strategy="keep_last",
    )

    out = pd.read_csv(preprocessed)
    assert len(out) == 2
    row = out.loc[out["canonical_smiles"] == "CCO"].iloc[0]
    assert int(row["Activity"]) == 2


def test_handle_missing_data_drops_missing_target_and_required_columns(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN", None],
            "target": [1.0, None, 3.0],
            "aux": [10.0, 20.0, None],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)
    preparer.handle_missing_data_and_duplicates(
        properties_of_interest=["target", "aux"],
        target_column="target",
        drop_missing_target=True,
        required_non_null_columns=["aux"],
    )

    out = pd.read_csv(preprocessed)
    assert len(out) == 1
    assert out.iloc[0]["canonical_smiles"] == "CCO"
    assert float(out.iloc[0]["target"]) == 1.0


def test_handle_missing_data_rejects_missing_required_columns(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1.0]}).to_csv(raw, index=False)
    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)

    with pytest.raises(ValueError, match="required_non_null_columns contain missing columns"):
        preparer.handle_missing_data_and_duplicates(
            properties_of_interest=["target"],
            required_non_null_columns=["does_not_exist"],
        )


def test_handle_missing_data_applies_row_filters_before_dedupe(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCO", "CCN"],
            "target_chembl_id": ["CHEMBL9999999", "CHEMBL3885651", "CHEMBL3885651"],
            "standard_value": [1.0, 2.0, 3.0],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)
    preparer.handle_missing_data_and_duplicates(
        properties_of_interest=["target_chembl_id", "standard_value"],
        dedupe_strategy="first",
        row_filters={"target_chembl_id": "CHEMBL3885651"},
    )

    out = pd.read_csv(preprocessed)
    assert len(out) == 2
    assert set(out["canonical_smiles"]) == {"CCO", "CCN"}
    filtered_row = out.loc[out["canonical_smiles"] == "CCO"].iloc[0]
    assert filtered_row["target_chembl_id"] == "CHEMBL3885651"
    assert float(filtered_row["standard_value"]) == 2.0


def test_handle_missing_data_rejects_missing_row_filter_columns(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1.0]}).to_csv(raw, index=False)
    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)

    with pytest.raises(ValueError, match="row_filters configured column"):
        preparer.handle_missing_data_and_duplicates(
            properties_of_interest=["target"],
            row_filters={"target_chembl_id": "CHEMBL3885651"},
        )


def test_required_non_null_columns_accepts_original_smiles_name(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "SMILES": ["CCO", None, "CCC"],
            "target": [1.0, 2.0, 3.0],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)
    preparer.handle_missing_data_and_duplicates(
        smiles_column="SMILES",
        properties_of_interest=["target"],
        required_non_null_columns=["SMILES", "target"],
    )

    out = pd.read_csv(preprocessed)
    assert "canonical_smiles" in out.columns
    assert len(out) == 2


def test_handle_missing_data_preserves_row_index_in_selective_mode(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "__row_index": [100, 101, 102],
            "SMILES": ["CCO", "CCN", "CCC"],
            "target": [1.0, 2.0, 3.0],
            "ignored": ["a", "b", "c"],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)
    preparer.handle_missing_data_and_duplicates(
        smiles_column="SMILES",
        properties_of_interest=["target"],
        target_column="target",
    )

    out = pd.read_csv(preprocessed)
    assert "__row_index" in out.columns
    assert out["__row_index"].tolist() == [100, 101, 102]
    assert "ignored" not in out.columns


def test_label_bioactivity_preserves_row_index_in_selective_mode(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"

    pd.DataFrame(
        {
            "__row_index": [200, 201, 202],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "standard_value": [500, 5000, 20000],
            "extra": [1, 2, 3],
        }
    ).to_csv(raw, index=False)

    preparer = DataPreparer(str(raw), str(preprocessed), str(curated), keep_all_columns=False)
    preparer.handle_missing_data_and_duplicates(
        properties_of_interest=["standard_value"],
    )
    preparer.label_bioactivity(active_threshold=1000, inactive_threshold=10000)

    out = pd.read_csv(curated)
    assert "__row_index" in out.columns
    assert out["__row_index"].tolist() == [200, 201, 202]
    assert set(out["class"].tolist()) == {"active", "intermediate", "inactive"}
