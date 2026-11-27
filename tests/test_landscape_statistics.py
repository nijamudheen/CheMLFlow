from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from molecular_landscape.statistics import kruskal_across_groups


def test_kruskal_epsilon_squared_matches_tomczak_definition():
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(0.0, 1.0, 30), rng.normal(0.5, 1.0, 30), rng.normal(1.0, 1.0, 25)]
    )
    groups = np.repeat(["a", "b", "c"], [30, 30, 25])
    df = pd.DataFrame({"MolWt": values, "activity_class": groups})

    out = kruskal_across_groups(df, ["MolWt"], "activity_class")

    h = stats.kruskal(values[:30], values[30:60], values[60:]).statistic
    n = values.size
    # Tomczak & Tomczak (2014): epsilon^2 = H / ((n^2 - 1) / (n + 1)) = H / (n - 1)
    assert out.loc[0, "epsilon_squared"] == pytest.approx(h / (n - 1))


def test_kruskal_epsilon_squared_bounds():
    df_sep = pd.DataFrame(
        {"MolWt": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "activity_class": ["a", "a", "a", "b", "b", "b"]}
    )
    out_sep = kruskal_across_groups(df_sep, ["MolWt"], "activity_class")
    assert out_sep.loc[0, "epsilon_squared"] == pytest.approx(1.0)

    df_same = pd.DataFrame(
        {"MolWt": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], "activity_class": ["a", "a", "a", "b", "b", "b"]}
    )
    out_same = kruskal_across_groups(df_same, ["MolWt"], "activity_class")
    assert out_same.loc[0, "epsilon_squared"] == pytest.approx(0.0)
