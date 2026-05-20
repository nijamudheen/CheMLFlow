import logging

import pandas as pd
import pytest

from MLModels import data_preprocessing


def test_load_features_labels_drops_excluded_columns(tmp_path):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"

    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [0, 1, 2],
            "feature_a": [1.0, 2.0, 3.0],
            "FP Calc.": [100.0, 101.0, 102.0],
            "SMILES": ["C", "CC", "CCC"],
        }
    ).to_csv(features_path, index=False)

    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [0, 1, 2],
            "FP Exp.": [150.0, 151.0, 152.0],
        }
    ).to_csv(labels_path, index=False)

    X, y = data_preprocessing.load_features_labels(
        str(features_path),
        str(labels_path),
        target_column="FP Exp.",
        exclude_columns=["FP Calc."],
    )

    assert "FP Calc." not in X.columns
    assert "SMILES" not in X.columns
    assert list(X.columns) == ["feature_a"]
    assert y.tolist() == [150.0, 151.0, 152.0]


def test_load_features_labels_surfaces_original_exception_type(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        data_preprocessing.load_features_labels(
            str(missing),
            str(missing),
            target_column="pIC50",
        )


def test_load_features_labels_retains_duplicate_rows_when_requested(tmp_path, caplog):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"

    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [10, 11, 12],
            "feature_a": [1.0, 1.0, 2.0],
        }
    ).to_csv(features_path, index=False)
    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [10, 11, 12],
            "FP Exp.": [288.0, 288.0, 300.0],
        }
    ).to_csv(labels_path, index=False)

    with caplog.at_level(logging.WARNING):
        X, y = data_preprocessing.load_features_labels(
            str(features_path),
            str(labels_path),
            target_column="FP Exp.",
            drop_duplicate_rows=False,
            fail_on_duplicate_rows=False,
        )

    assert X.index.tolist() == [10, 11, 12]
    assert y.index.tolist() == [10, 11, 12]
    assert "Retaining 1 duplicate aligned feature/label row" in caplog.text
