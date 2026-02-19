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
