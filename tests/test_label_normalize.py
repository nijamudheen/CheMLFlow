import numpy as np
import pandas as pd

from utilities import label_normalize


def test_label_normalize_numeric_values():
    df = pd.DataFrame(
        {
            "Activity": [1, 0, "1", "0", np.int64(1), np.float64(0.0)],
            "smiles": ["C"] * 6,
        }
    )
    out = label_normalize.normalize_labels(
        df,
        source_column="Activity",
        target_column="label",
        positive=["active"],
        negative=["inactive"],
        drop_unmapped=True,
    )
    assert len(out) == 6
    assert set(out["label"].unique()) == {0, 1}


def test_label_normalize_string_labels():
    df = pd.DataFrame(
        {
            "Activity": ["active", "inactive", "active"],
            "smiles": ["C", "CC", "CCC"],
        }
    )
    out = label_normalize.normalize_labels(
        df,
        source_column="Activity",
        target_column="label",
        positive=["active"],
        negative=["inactive"],
        drop_unmapped=True,
    )
    assert out["label"].tolist() == [1, 0, 1]
