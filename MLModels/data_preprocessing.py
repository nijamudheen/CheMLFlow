# data_preprocessing.py
import os
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from typing import Iterable, Tuple, Dict, Any, Sequence

# ── 1) Load data ───────────────────────────────────────────────────────────────
def load_data(features_file: str, labels_file: str):
    """Load the RDKit descriptors as features (X) and the pIC50 values as target (y)."""
    try:
        logging.info(f"Loading features from {features_file}")
        X = pd.read_csv(features_file)
        logging.info(f"Loading labels (pIC50) from {labels_file}")
        y_df = pd.read_csv(labels_file)

        if X.shape[0] != y_df.shape[0]:
            raise ValueError("The number of samples in the features file and the labels file do not match.")
        if 'pIC50' not in y_df.columns:
            raise ValueError("The labels file must contain a 'pIC50' column.")

        y = y_df['pIC50']
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = ~X.isna().any(axis=1) & ~y.isna()

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Remove duplicates across X|y join
        combined = pd.concat([X_clean, y_clean], axis=1)
        before = combined.shape[0]
        combined = combined.drop_duplicates()
        after = combined.shape[0]
        logging.info(f"Removed {before - after} duplicate entries.")

        X_clean = combined.drop('pIC50', axis=1)
        y_clean = combined['pIC50']

        logging.info(f"Data shape after initial cleaning: {X_clean.shape}")
        return X_clean, y_clean
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        raise

# ── 2) Preprocess ──────────────────────────────────────────────────────────────
class _IdentityScaler:
    def fit_transform(self, X):
        return np.asarray(X, dtype=float)

    def transform(self, X):
        return np.asarray(X, dtype=float)


def _resolve_scaler(scaler_name: str):
    normalized = str(scaler_name or "robust").strip().lower()
    if normalized == "robust":
        return normalized, RobustScaler()
    if normalized == "standard":
        return normalized, StandardScaler()
    if normalized == "minmax":
        return normalized, MinMaxScaler(clip=True)
    if normalized == "none":
        return normalized, _IdentityScaler()
    raise ValueError(
        "preprocess.scaler must be one of: 'robust', 'standard', 'minmax', 'none'."
    )


def preprocess_data(
    X: pd.DataFrame,
    variance_threshold: float = 0.8 * (1 - 0.8),
    corr_threshold: float = 0.95,
    clip_range: tuple = (-1e10, 1e10),
    scaler: str = "robust",
    allow_full_fit: bool = False,
):
    """Preprocess the data by scaling and removing low-variance and highly correlated features."""
    try:
        if not allow_full_fit:
            raise ValueError(
                "preprocess_data fits on the full dataset and is deprecated. "
                "Use fit_preprocessor/transform_preprocessor with a train/test split."
            )

        scaler_name, scaler_obj = _resolve_scaler(scaler)
        logging.info("Scaling features using %s scaler.", scaler_name)
        X_scaled = scaler_obj.fit_transform(X) #doing this to the whole data is considered data leakage
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Removing low-variance features (on both training dataset and test) - OR DO BEFORE SCALING
        logging.info("Removing low-variance features.")
        selector = VarianceThreshold(threshold=variance_threshold)
        X_reduced = selector.fit_transform(X_scaled_df)
        features_after_variance = X_scaled_df.columns[selector.get_support(indices=True)]
        X_reduced_df = pd.DataFrame(X_reduced, columns=features_after_variance)
        logging.info(f"Data reduced to {X_reduced_df.shape[1]} features after removing low-variance columns.")

        # Removing highly correlated features
        logging.info("Removing highly correlated features.")
        corr_matrix = X_reduced_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Adjust the threshold to 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        X_final = X_reduced_df.drop(columns=to_drop)
        logging.info(f"Data reduced to {X_final.shape[1]} features after removing highly correlated features.")
        logging.info(f"Data shape after preprocessing: {X_final.shape}")
        
        # Clip extreme values to a safe range for float32 conversion.
        # This ensures that when the model or pandas casts the data to float32,
        # the values do not exceed the representable limits.
        min_threshold, max_threshold = clip_range  # Adjust these thresholds based on your data's expected range.
        X_final = X_final.clip(lower=min_threshold, upper=max_threshold)
        logging.info(f"Data clipped to range [{min_threshold}, {max_threshold}].")
        logging.info(f"Final data range: min={X_final.min().min()}, max={X_final.max().max()}")

        return X_final
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise


def fit_preprocessor(
    X_train: pd.DataFrame,
    variance_threshold: float,
    corr_threshold: float,
    clip_range: tuple,
    scaler: str = "robust",
) -> Dict[str, Any]:
    """Fit preprocessing steps on training data only."""
    scaler_name, scaler_obj = _resolve_scaler(scaler)
    X_scaled = scaler_obj.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)

    selector = VarianceThreshold(threshold=variance_threshold)
    X_reduced = selector.fit_transform(X_scaled_df)
    features_after_variance = X_scaled_df.columns[selector.get_support(indices=True)]
    X_reduced_df = pd.DataFrame(X_reduced, columns=features_after_variance)

    corr_matrix = X_reduced_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    return {
        "scaler": scaler_obj,
        "scaler_name": scaler_name,
        "selector": selector,
        "variance_features": list(features_after_variance),
        "corr_drop": to_drop,
        "clip_range": clip_range,
    }


def transform_preprocessor(X: pd.DataFrame, preprocessor: Dict[str, Any]) -> pd.DataFrame:
    """Apply a fitted preprocessor to new data."""
    scaler = preprocessor["scaler"]
    selector = preprocessor["selector"]
    variance_features = preprocessor["variance_features"]
    to_drop = preprocessor["corr_drop"]
    clip_range = preprocessor["clip_range"]

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    X_reduced = selector.transform(X_scaled_df)
    X_reduced_df = pd.DataFrame(X_reduced, columns=variance_features)

    X_final = X_reduced_df.drop(columns=to_drop)

    min_threshold, max_threshold = clip_range
    X_final = X_final.clip(lower=min_threshold, upper=max_threshold)
    return X_final

# ── 3) Stable feature selection ────────────────────────────────────────────────
def select_stable_features(X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    k: int = 50,
    out_path: str | None = None,):
    """Select features based on cross-validation stability."""
    try:
        logging.info("Selecting features based on cross-validation stability (RandomForest).")
        importances_accum = np.zeros(X.shape[1])
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        for tr, _ in cv.split(X):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            rf = RandomForestRegressor(random_state=random_state)
            rf.fit(X_tr, y_tr)
            importances_accum += rf.feature_importances_

        importances_accum /= cv.get_n_splits()
        s = pd.Series(importances_accum, index=X.columns)
        top_features = s.nlargest(k).index
        X_sel = X[top_features]
        logging.info(f"Selected top {k} stable features. Shape: {X_sel.shape}")

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                for name in top_features:
                    f.write(f"{name}\n")
            logging.info(f"Selected feature list written to {out_path}")

        return X_sel
    except Exception as e:
        logging.error(f"Error during stable feature selection: {e}", exc_info=True)
        raise


def _find_target_column(columns: Iterable[str], target_column: str) -> tuple[str, int]:
    target_lower = target_column.lower()
    for idx, col in enumerate(columns):
        if col.lower() == target_lower:
            return col, idx
    raise ValueError(f"The labels file must contain a '{target_column}' column.")


def select_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    target_col, target_idx = _find_target_column(df.columns, target_column)
    series = df.iloc[:, target_idx]
    series.name = target_col
    return series


def _drop_smiles_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.lower() in {"smiles", "canonical_smiles"}]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _one_hot_categorical(df: pd.DataFrame, categorical_features: Sequence[str] | None) -> pd.DataFrame:
    if not categorical_features:
        return df
    cat_cols = [c for c in categorical_features if c in df.columns]
    if not cat_cols:
        return df
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    return pd.get_dummies(df, columns=cat_cols, dummy_na=False)


def load_features_labels(
    features_file: str,
    labels_file: str,
    target_column: str,
    categorical_features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    restrict_to_indices: Sequence[int] | None = None,
    *,
    drop_invalid_rows: bool = True,
    drop_duplicate_rows: bool = True,
    fail_on_invalid_rows: bool = False,
    fail_on_duplicate_rows: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and labels, align them, and drop duplicates and target columns."""
    try:
        logging.info(f"Loading features from {features_file}")
        X = pd.read_csv(features_file)
        logging.info(f"Loading labels ({target_column}) from {labels_file}")
        y_df = pd.read_csv(labels_file)

        if ROW_INDEX_COL in X.columns:
            X = X.set_index(ROW_INDEX_COL, drop=True)
        if ROW_INDEX_COL in y_df.columns:
            y_df = y_df.set_index(ROW_INDEX_COL, drop=True)

        if X.shape[0] != y_df.shape[0]:
            raise ValueError("The number of samples in the features file and the labels file do not match.")

        target_col, _ = _find_target_column(y_df.columns, target_column)
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        y = select_target_series(y_df, target_column)
        if not X.index.equals(y.index):
            y = y.reindex(X.index)
        if restrict_to_indices is not None:
            requested_indices = [int(i) for i in restrict_to_indices]
            if requested_indices:
                available_index = set(X.index.tolist()) & set(y.index.tolist())
                missing = [i for i in requested_indices if i not in available_index]
                if missing:
                    preview = missing[:10]
                    raise ValueError(
                        "Feature/label files are missing split rows. "
                        f"missing_count={len(missing)} preview={preview}"
                    )
                X = X.reindex(requested_indices)
                y = y.reindex(requested_indices)
        if y.dtype == object:
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.notna().sum() >= 0.9 * y.notna().sum():
                y = y_num

        # Drop explicitly excluded columns (e.g., known leakage fields).
        if exclude_columns:
            # Preserve order while de-duplicating configured names.
            configured = list(dict.fromkeys(str(col) for col in exclude_columns))
            drop_cols = [col for col in configured if col in X.columns]
            if drop_cols:
                logging.info("Dropping user-excluded feature columns: %s", drop_cols)
                X = X.drop(columns=drop_cols)

        # Drop SMILES-like columns from features.
        X = _drop_smiles_columns(X)

        # One-hot encode allow-listed categorical features.
        X = _one_hot_categorical(X, categorical_features)

        # Coerce feature columns to numeric; drop columns that become entirely NaN.
        non_numeric = []
        for col in X.columns:
            if not is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            if X[col].isna().all():
                non_numeric.append(col)
        if non_numeric:
            logging.info("Dropping non-numeric feature columns: %s", non_numeric)
            X = X.drop(columns=non_numeric)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = ~X.isna().any(axis=1) & ~y.isna()
        invalid_rows = int((~valid_mask).sum())
        if invalid_rows:
            if not drop_invalid_rows and fail_on_invalid_rows:
                raise ValueError(
                    f"Found {invalid_rows} row(s) with NaN/Inf after feature coercion. "
                    "Set curate/pre-split cleaning so rows are filtered before split."
                )
            if drop_invalid_rows:
                logging.info("Dropping %s invalid row(s) with NaN/Inf in features/labels.", invalid_rows)
                X = X[valid_mask]
                y = y[valid_mask]

        combined = pd.concat([X, y], axis=1)
        duplicate_rows = int(combined.duplicated().sum())
        if duplicate_rows:
            if not drop_duplicate_rows and fail_on_duplicate_rows:
                raise ValueError(
                    f"Found {duplicate_rows} duplicate row(s) after alignment. "
                    "Set curate/pre-split cleaning so duplicates are removed before split."
                )
            if drop_duplicate_rows:
                combined = combined.drop_duplicates()
                logging.info("Removed %s duplicate entries.", duplicate_rows)
            else:
                logging.warning(
                    "Retaining %s duplicate aligned feature/label row(s). "
                    "This can happen when distinct molecules collide under a feature representation; "
                    "row IDs are preserved for split comparability.",
                    duplicate_rows,
                )

        X_clean = combined.drop(columns=[target_col])
        y_clean = combined[target_col]

        logging.info(f"Data shape after initial cleaning: {X_clean.shape}")
        return X_clean, y_clean
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        raise

# ── 4) Data quality ───────────────────────────────────────────────────────────
def verify_data_quality(X: pd.DataFrame, y: pd.Series):
    """Check for data quality issues."""
    try:
        logging.info("Verifying data quality.")
        # Check for duplicates
        if X.duplicated().any():
            logging.warning("Duplicates found in feature data.")
        # Check for constant features
        if (X.nunique() == 1).any():
            logging.warning("Constant features found in data.")
        # Additional checks can be added as needed
    except Exception as e:
        logging.error(f"Error verifying data quality: {e}")

# ── 5) Leakage check ──────────────────────────────────────────────────────────
def check_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Check for exact row overlap between training and test sets."""
    
    try:
        logging.info("Checking for exact row overlap between training and test sets.")
        intersection = pd.merge(X_train, X_test, how='inner')
        if not intersection.empty:
            logging.warning("Potential row overlap detected between training and test sets.")
        else:
            logging.info("No row overlap detected.")
    except Exception as e:
        logging.error(f"Error checking data leakage: {e}")
ROW_INDEX_COL = "__row_index"
