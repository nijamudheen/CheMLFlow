"""Rigorous, exploratory dataset statistics for ML-readiness EDA.

Additive and opt-in: this does not change the validated workflow. Design choices
follow the project's honesty philosophy:

* Every association is **univariate and exploratory**, never causal.
* Findings carry an **effect size** and **sample size**, not just a p-value.
* Multiple testing is controlled with **Benjamini-Hochberg FDR** (q-values), so a
  screen over many descriptors does not manufacture significance.
* **Non-parametric** tests are preferred (Spearman, Kruskal-Wallis, Mann-Whitney)
  because molecular descriptors are rarely normal; parametric counterparts are
  reported alongside for reference.

These are intended to inform ML training decisions (feature relevance, target
transformation, class balance, feature redundancy), not to assert biology.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (monotone, clipped to [0, 1])."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def describe_property(values: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics plus a normality test (informs target transforms)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = int(v.size)
    out: Dict[str, Any] = {
        "n": n,
        "mean": float(np.mean(v)) if n else float("nan"),
        "median": float(np.median(v)) if n else float("nan"),
        "std": float(np.std(v, ddof=1)) if n > 1 else float("nan"),
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))) if n else float("nan"),
        "min": float(v.min()) if n else float("nan"),
        "max": float(v.max()) if n else float("nan"),
        "skewness": float(stats.skew(v)) if n > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(v)) if n > 3 else float("nan"),
    }
    if n >= 20:
        result = stats.normaltest(v)
        out.update(normality_test="dagostino_pearson",
                   normality_stat=float(result.statistic), normality_p=float(result.pvalue))
    elif n >= 3:
        w, p = stats.shapiro(v)
        out.update(normality_test="shapiro_wilk", normality_stat=float(w), normality_p=float(p))
    out["appears_normal_at_0.05"] = bool(out.get("normality_p", 0.0) > 0.05)
    return out


def descriptor_property_association(
    df: pd.DataFrame, descriptors: Sequence[str], property_col: str
) -> pd.DataFrame:
    """Univariate descriptor<->property correlations (Spearman + Pearson) with FDR."""
    y = pd.to_numeric(df[property_col], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        if descriptor not in df.columns:
            continue
        x = pd.to_numeric(df[descriptor], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
            continue
        pearson = stats.pearsonr(x[mask], y[mask])
        spearman = stats.spearmanr(x[mask], y[mask])
        rows.append({
            "descriptor": descriptor, "n": n,
            "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_q_bh"] = benjamini_hochberg(out["spearman_p"])
        out["pearson_q_bh"] = benjamini_hochberg(out["pearson_p"])
        out = out.iloc[out["spearman_r"].abs().argsort()[::-1]].reset_index(drop=True)
    return out


def kruskal_across_groups(
    df: pd.DataFrame, descriptors: Sequence[str], group_col: str
) -> pd.DataFrame:
    """Kruskal-Wallis test of each descriptor across groups, with epsilon-squared + FDR."""
    groups = [g for g in df[group_col].dropna().unique()]
    rows: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        if descriptor not in df.columns:
            continue
        samples = [
            pd.to_numeric(df.loc[df[group_col] == g, descriptor], errors="coerce").dropna().to_numpy()
            for g in groups
        ]
        samples = [s for s in samples if s.size > 0]
        total = int(sum(s.size for s in samples))
        k = len(samples)
        if k < 2 or total <= k:
            continue
        result = stats.kruskal(*samples)
        # Tomczak & Tomczak (2014) epsilon-squared: H / ((n^2 - 1) / (n + 1)) = H / (n - 1).
        epsilon_sq = float(result.statistic / (total - 1))
        rows.append({
            "descriptor": descriptor, "n": total, "n_groups": k,
            "kruskal_h": float(result.statistic), "p": float(result.pvalue),
            "epsilon_squared": min(1.0, max(0.0, epsilon_sq)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = benjamini_hochberg(out["p"])
        out = out.sort_values("epsilon_squared", ascending=False).reset_index(drop=True)
    return out


def binary_contrast(
    df: pd.DataFrame, descriptors: Sequence[str], group_col: str,
    positive: str, negative: str,
) -> pd.DataFrame:
    """Two-group contrast per descriptor: Mann-Whitney + Welch t, with effect sizes + FDR."""
    rows: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        if descriptor not in df.columns:
            continue
        a = pd.to_numeric(df.loc[df[group_col] == positive, descriptor], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(df.loc[df[group_col] == negative, descriptor], errors="coerce").dropna().to_numpy()
        if a.size < 2 or b.size < 2:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        cliffs_delta = float(2.0 * u.statistic / (a.size * b.size) - 1.0)
        welch = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        cohens_d = float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else float("nan")
        rows.append({
            "descriptor": descriptor, "n_positive": int(a.size), "n_negative": int(b.size),
            "mannwhitney_p": float(u.pvalue), "cliffs_delta": cliffs_delta,
            "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
            "cohens_d": cohens_d,
            "median_positive": float(np.median(a)), "median_negative": float(np.median(b)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mannwhitney_q_bh"] = benjamini_hochberg(out["mannwhitney_p"])
        out = out.iloc[out["cliffs_delta"].abs().argsort()[::-1]].reset_index(drop=True)
    return out


def categorical_association(a: Sequence, b: Sequence) -> Dict[str, Any]:
    """Chi-square test of independence between two categoricals, with Cramer's V."""
    table = pd.crosstab(pd.Series(a), pd.Series(b))
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    min_dim = min(table.shape)
    cramers_v = float(np.sqrt(chi2 / (n * (min_dim - 1)))) if min_dim > 1 and n else float("nan")
    return {
        "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": cramers_v, "n": n,
        "min_expected_count": float(np.min(expected)),
        "low_expected_count_warning": bool(np.min(expected) < 5),
    }


def class_balance(series: Sequence) -> Dict[str, Any]:
    """Class counts and the imbalance ratio (max/min) — informs resampling/weighting."""
    counts = pd.Series(series).dropna().value_counts()
    values = counts.to_numpy()
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "n": int(values.sum()), "n_classes": int(values.size),
        "imbalance_ratio": float(values.max() / values.min()) if values.size and values.min() > 0 else float("inf"),
    }


def descriptor_collinearity(
    df: pd.DataFrame, descriptors: Sequence[str], threshold: float = 0.8
) -> pd.DataFrame:
    """Descriptor pairs with |Spearman r| >= threshold — flags redundant features."""
    present = [d for d in descriptors if d in df.columns]
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman")
    rows: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"descriptor_a": cols[i], "descriptor_b": cols[j], "spearman_r": float(r)})
    out = pd.DataFrame(rows, columns=["descriptor_a", "descriptor_b", "spearman_r"])
    if not out.empty:
        out = out.iloc[out["spearman_r"].abs().argsort()[::-1]].reset_index(drop=True)
    return out
