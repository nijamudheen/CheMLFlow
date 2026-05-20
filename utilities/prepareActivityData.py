import argparse
import json
import logging

import pandas as pd

from rdkit import Chem


_DEDUPE_STRATEGY_ALIASES = {
    "keep_first": "first",
    "keep_last": "last",
}
_VALID_DEDUPE_STRATEGIES = {"first", "last", "drop_conflicts", "majority"}


def _normalize_dedupe_strategy(dedupe_strategy: str | None) -> str | None:
    if dedupe_strategy is None:
        return None
    normalized = str(dedupe_strategy).strip().lower()
    normalized = _DEDUPE_STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_DEDUPE_STRATEGIES:
        allowed = ", ".join(sorted(_VALID_DEDUPE_STRATEGIES | set(_DEDUPE_STRATEGY_ALIASES.keys())))
        raise ValueError(f"Unsupported dedupe_strategy={dedupe_strategy!r}. Allowed values: {allowed}.")
    return normalized


def _normalize_row_filters(row_filters: dict[str, object] | None) -> dict[str, list[object]]:
    if row_filters is None:
        return {}
    if not isinstance(row_filters, dict):
        raise ValueError("row_filters must be a mapping of column -> allowed value(s).")

    normalized: dict[str, list[object]] = {}
    for raw_column, raw_values in row_filters.items():
        column = str(raw_column).strip()
        if not column:
            continue
        if isinstance(raw_values, (list, tuple, set)):
            values = [value for value in raw_values]
        else:
            values = [raw_values]
        if not values:
            raise ValueError(f"row_filters[{column!r}] must contain at least one allowed value.")
        normalized[column] = values
    return normalized


class DataPreparer:
    def __init__(self, raw_data_file, preprocessed_file, curated_file, keep_all_columns=False):
        self.raw_data_file = raw_data_file
        self.preprocessed_file = preprocessed_file
        self.curated_file = curated_file
        self.keep_all_columns = keep_all_columns
        logging.info(f"Initialized DataPreparer with raw data file: {self.raw_data_file}")

    def handle_missing_data_and_duplicates(
        self,
        smiles_column=None,
        properties_of_interest=None,
        dedupe_strategy: str | None = None,
        label_column: str | None = None,
        drop_missing_smiles: bool = True,
        target_column: str | None = None,
        drop_missing_target: bool = True,
        required_non_null_columns: list[str] | None = None,
        row_filters: dict[str, object] | None = None,
    ):
        """Load data, identify the SMILES column, handle missing values, and remove duplicates.
        - Auto-detect the SMILES column if not provided (try 'canonical_smiles', 'smiles', 'SMILES').
        - Keep only rows with non-null SMILES.
        - Rename the SMILES column to 'canonical_smiles' for downstream consistency.
        - If properties_of_interest is provided, keep only ID + 'canonical_smiles' + those properties (if present).
          Otherwise, keep only ID (if present) and 'canonical_smiles' (no dataset-specific defaults).
        """
        try:
            logging.info(f"Loading data from {self.raw_data_file}")
            df = pd.read_csv(self.raw_data_file)
            dedupe_strategy = _normalize_dedupe_strategy(dedupe_strategy)
            normalized_row_filters = _normalize_row_filters(row_filters)

            # Detect the SMILES column
            candidate_smiles = [smiles_column] if smiles_column else [
                "canonical_smiles",
                "smiles",
                "SMILES",
                "Smiles",
                "Drug",
                "drug",
            ]
            smiles_col = next((c for c in candidate_smiles if c and c in df.columns), None)
            if smiles_col is None:
                raise ValueError("Could not find a SMILES column. Tried: " + ", ".join([c for c in candidate_smiles if c]))

            logging.info(f"Using '{smiles_col}' as the SMILES column.")
            original_smiles_col = smiles_col

            # Optionally drop rows with missing SMILES at raw-curation stage.
            if drop_missing_smiles:
                before = len(df)
                df_clean = df[df[smiles_col].notna()].copy()
                removed = before - len(df_clean)
                if removed:
                    logging.info("Dropped %s row(s) with missing SMILES values.", removed)
            else:
                df_clean = df.copy()

            # Normalize name for downstream steps
            if smiles_col != 'canonical_smiles':
                df_clean = df_clean.rename(columns={smiles_col: 'canonical_smiles'})
                smiles_col = 'canonical_smiles'

            # Optionally drop rows with missing target values when available at curate time.
            if drop_missing_target and target_column:
                resolved_target = str(target_column).strip()
                if resolved_target:
                    if resolved_target in df_clean.columns:
                        before = len(df_clean)
                        df_clean = df_clean[df_clean[resolved_target].notna()].copy()
                        removed = before - len(df_clean)
                        if removed:
                            logging.info(
                                "Dropped %s row(s) with missing target values in %r.",
                                removed,
                                resolved_target,
                            )
                    else:
                        logging.warning(
                            "drop_missing_target=true but target_column=%r was not found during curate; skipping.",
                            resolved_target,
                        )

            required_cols_raw = [str(c).strip() for c in (required_non_null_columns or []) if str(c).strip()]
            required_cols: list[str] = []
            for col in required_cols_raw:
                normalized = col
                if original_smiles_col != "canonical_smiles" and col == original_smiles_col:
                    normalized = "canonical_smiles"
                if normalized not in required_cols:
                    required_cols.append(normalized)
            if required_cols:
                missing_required = [c for c in required_cols if c not in df_clean.columns]
                if missing_required:
                    raise ValueError(
                        "required_non_null_columns contain missing columns: "
                        + ", ".join(sorted(missing_required))
                    )
                before = len(df_clean)
                df_clean = df_clean.dropna(subset=required_cols).copy()
                removed = before - len(df_clean)
                if removed:
                    logging.info(
                        "Dropped %s row(s) missing required_non_null_columns=%s.",
                        removed,
                        required_cols,
                    )

            if normalized_row_filters:
                for col, allowed_values in normalized_row_filters.items():
                    normalized_col = col
                    if original_smiles_col != "canonical_smiles" and col == original_smiles_col:
                        normalized_col = "canonical_smiles"
                    if normalized_col not in df_clean.columns:
                        raise ValueError(
                            f"row_filters configured column {normalized_col!r}, but it was not found in the dataset."
                        )
                    before = len(df_clean)
                    df_clean = df_clean[df_clean[normalized_col].isin(allowed_values)].copy()
                    removed = before - len(df_clean)
                    if removed:
                        logging.info(
                            "Dropped %s row(s) outside row_filters[%r]=%s.",
                            removed,
                            normalized_col,
                            allowed_values,
                        )

            # Remove exact duplicates by SMILES text (pre-canonicalization)
            if dedupe_strategy in {None, "first", "last"}:
                before = len(df_clean)
                keep = "last" if dedupe_strategy == "last" else "first"
                df_clean = df_clean.drop_duplicates(subset=[smiles_col], keep=keep)
                logging.info(
                    "Removed %s duplicate rows based on '%s' (keep=%s).",
                    before - len(df_clean),
                    smiles_col,
                    keep,
                )
            else:
                logging.info(
                    "Skipping pre-canonicalization dedupe because dedupe_strategy=%s.",
                    dedupe_strategy,
                )

            # Always keep an ID column if present
            id_candidates = ['molecule_chembl_id', 'mol_id', 'compound_id', 'id', 'ID']
            id_col = next((c for c in id_candidates if c in df_clean.columns), None)

            if self.keep_all_columns:
                selected_columns = df_clean.columns.tolist()
                df_clean = df_clean[selected_columns]
                logging.info("Keeping all columns for preprocessing.")
            else:
                # Decide which columns to keep
                present_props = [c for c in (properties_of_interest or []) if c in df_clean.columns]
                selected_columns = []
                if id_col:
                    selected_columns.append(id_col)
                if "__row_index" in df_clean.columns and "__row_index" not in selected_columns:
                    selected_columns.append("__row_index")
                selected_columns.append(smiles_col)
                selected_columns.extend([c for c in present_props if c not in selected_columns])
                if label_column and label_column in df_clean.columns and label_column not in selected_columns:
                    selected_columns.append(label_column)

                df_clean = df_clean[selected_columns]
                logging.info(f"Selected columns for preprocessing: {selected_columns}")

            # Save preprocessed data
            df_clean.to_csv(self.preprocessed_file, index=False)
            logging.info(f"Preprocessed data saved to {self.preprocessed_file}")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def label_bioactivity(self, active_threshold=1000, inactive_threshold=10000):
        """ If a 'standard_value' column exists, label classes; otherwise, skip gracefully.
        - In datasets without bioactivity, this will be a no-op (preprocessed -> curated).
        - If 'standard_value' exists (e.g., ChEMBL downloads), apply labeling using thresholds.
        """
        try:
            logging.info(f"Loading preprocessed data from {self.preprocessed_file}")
            df = pd.read_csv(self.preprocessed_file)

            if 'standard_value' not in df.columns:
                logging.info("No 'standard_value' column found; skipping bioactivity labeling and passing data through.")
                df.to_csv(self.curated_file, index=False)
                logging.info(f"Curated data (pass-through) saved to {self.curated_file}")
                return
            
            # Filter out rows with missing 'standard_value'
            df_clean = df[df.standard_value.notna()]

            # Label as active / intermediate / inactive
            import numpy as np
            
            conditions = [
                (df_clean['standard_value'] <= active_threshold),
                (df_clean['standard_value'] > inactive_threshold)
            ]
            choices = ['active', 'inactive']
            labeled = np.select(conditions, choices, default='intermediate')
            df_clean['class'] = pd.Categorical(labeled, categories=['active', 'intermediate', 'inactive'], ordered=True)

            if self.keep_all_columns:
                df_clean.to_csv(self.curated_file, index=False)
            else:
                # Keep typical useful columns if present
                keep_cols = [
                    c
                    for c in [
                        'molecule_chembl_id',
                        '__row_index',
                        'canonical_smiles',
                        'standard_value',
                        'class',
                    ]
                    if c in df_clean.columns
                ]
                if not keep_cols:
                    keep_cols = df_clean.columns.tolist()
                df_clean[keep_cols].to_csv(self.curated_file, index=False)
            logging.info(f"Curated data with labels saved to {self.curated_file}")
        except Exception as e:
            logging.error(f"Error during labeling: {e}")
            raise


    def clean_smiles_column(
        self,
        curated_smiles_output,
        require_neutral_charge=False,
        prefer_largest_fragment=True,
        dedupe_strategy: str | None = None,
        label_column: str | None = None,
        drop_invalid_smiles: bool = True,
    ):
        """Clean the 'canonical_smiles' column:
        - Canonicalize SMILES with RDKit if available.
        - For multi-fragment SMILES, keep the largest (by heavy-atom count) fragment by default (toggle with prefer_largest_fragment).
        - Drop molecules that fail sanitization; optionally enforce neutral charge.
        - Remove duplicates *after* canonicalization.
        """
        try:
            logging.info(f"Loading curated data from {self.curated_file}")
            df = pd.read_csv(self.curated_file)
            dedupe_strategy = _normalize_dedupe_strategy(dedupe_strategy)

            if 'canonical_smiles' not in df.columns:
                raise ValueError("Expected a 'canonical_smiles' column in curated data.")

            def rdkit_canonical(smiles):
                if Chem is None:
                    return smiles, True  # RDKit not available; pass-through
                try:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is None:
                        return None, False

                    # Split into fragments if any '.'
                    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False) or []
                    if not frags:
                        frags = [mol]

                    # Sanitize each fragment, optionally pick largest by heavy atom count
                    best = None
                    best_heavy = -1
                    for m in frags:
                        try:
                            Chem.SanitizeMol(m)
                        except Exception:
                            continue
                        heavy = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
                        if not prefer_largest_fragment:
                            best = m
                            best_heavy = heavy
                            break
                        if heavy > best_heavy:
                            best_heavy = heavy
                            best = m
                    if best is None:
                        return None, False

                    # Optional element/charge checks
                    if require_neutral_charge and best.GetFormalCharge() != 0:
                        return None, False

                    can = Chem.MolToSmiles(best, canonical=True)
                    return can, True
                except Exception:
                    return None, False

            # Apply canonicalization row-wise.
            cans = []
            keep_mask = []
            for original in df['canonical_smiles']:
                if pd.isna(original) or not str(original).strip():
                    c, ok = None, False
                else:
                    c, ok = rdkit_canonical(str(original))
                if ok and c is not None:
                    cans.append(c)
                    keep_mask.append(True)
                else:
                    cans.append(original)
                    keep_mask.append(not drop_invalid_smiles)

            if Chem is None:
                logging.warning("RDKit not available; SMILES were not re-canonicalized. Install RDKit for full cleaning.")

            df['canonical_smiles'] = cans
            before_filter = len(df)
            df = df[keep_mask].copy()
            dropped_invalid = before_filter - len(df)
            if drop_invalid_smiles:
                logging.info(
                    "After sanitization/canonicalization: %s rows remain (dropped %s invalid SMILES row(s)).",
                    len(df),
                    dropped_invalid,
                )
            else:
                logging.info(
                    "After sanitization/canonicalization: %s rows remain (invalid SMILES retained).",
                    len(df),
                )

            # Drop or reconcile duplicates after canonicalization
            before = len(df)
            if label_column and label_column in df.columns and dedupe_strategy in {"drop_conflicts", "majority"}:
                grouped = df.groupby("canonical_smiles", dropna=False)
                keep_rows = []
                dropped = 0
                for _, group in grouped:
                    labels = group[label_column].dropna().unique()
                    if len(labels) == 0:
                        keep_rows.append(group.iloc[0])
                        continue
                    if len(labels) == 1:
                        keep_rows.append(group.iloc[0])
                        continue
                    if dedupe_strategy == "drop_conflicts":
                        dropped += len(group)
                        continue
                    counts = group[label_column].value_counts()
                    if counts.iloc[0] == counts.iloc[1:2].max():
                        dropped += len(group)
                        continue
                    winner = counts.idxmax()
                    keep_rows.append(group[group[label_column] == winner].iloc[0])
                df = pd.DataFrame(keep_rows)
                logging.info(
                    "Reconciled duplicates after canonicalization with strategy '%s' (dropped %s rows).",
                    dedupe_strategy,
                    dropped,
                )
            else:
                keep = "last" if dedupe_strategy == "last" else "first"
                df = df.drop_duplicates(subset=["canonical_smiles"], keep=keep)
                logging.info(
                    "Removed %s duplicates after canonicalization (keep=%s).",
                    before - len(df),
                    keep,
                )

            # Save cleaned SMILES to a separate file
            df[['canonical_smiles']].to_csv(curated_smiles_output, index=False)
            logging.info(f"Curated SMILES saved to {curated_smiles_output}")

            # Overwrite curated_file with the cleaned DataFrame too
            df.to_csv(self.curated_file, index=False)
            logging.info(f"Updated curated dataset saved to {self.curated_file}")
        except Exception as e:
            logging.error(f"Error during SMILES cleaning: {e}")
            raise


def main(
    raw_data_file,
    preprocessed_file,
    curated_file,
    curated_smiles_output,
    active_threshold,
    inactive_threshold,
    smiles_column=None,
    properties_of_interest=None,
    require_neutral_charge=False,
    prefer_largest_fragment=True,
    keep_all_columns=False,
    dedupe_strategy=None,
    label_column=None,
    drop_missing_smiles=True,
    target_column=None,
    drop_missing_target=True,
    required_non_null_columns=None,
    drop_invalid_smiles=True,
    row_filters=None,
):
    """Main function to preprocess, optionally label, and clean SMILES data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dedupe_strategy = _normalize_dedupe_strategy(dedupe_strategy)
    preparer = DataPreparer(raw_data_file, preprocessed_file, curated_file, keep_all_columns=keep_all_columns)
    preparer.handle_missing_data_and_duplicates(
        smiles_column=smiles_column,
        properties_of_interest=properties_of_interest,
        dedupe_strategy=dedupe_strategy,
        label_column=label_column,
        drop_missing_smiles=drop_missing_smiles,
        target_column=target_column,
        drop_missing_target=drop_missing_target,
        required_non_null_columns=required_non_null_columns,
        row_filters=row_filters,
    )
    # Label if bioactivity present; otherwise pass-through
    preparer.label_bioactivity(active_threshold=active_threshold, inactive_threshold=inactive_threshold)
    # Canonicalize and filter
    preparer.clean_smiles_column(
        curated_smiles_output,
        require_neutral_charge=require_neutral_charge,
        prefer_largest_fragment=prefer_largest_fragment,
        dedupe_strategy=dedupe_strategy,
        label_column=label_column,
        drop_invalid_smiles=drop_invalid_smiles,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and process molecular data (generalized, dataset-agnostic)."
    )
    parser.add_argument('raw_data_file', type=str, help="Input CSV file with raw molecular data.")
    parser.add_argument('preprocessed_file', type=str, help="Output CSV file for preprocessed data (selected columns).")
    parser.add_argument('curated_file', type=str, help="Output CSV file for curated data (labeled if bioactivity present).")
    parser.add_argument('curated_smiles_output', type=str, help="Output CSV file for curated, canonical SMILES.")
    parser.add_argument('--active_threshold', type=float, default=1000, help="(ChEMBL) Threshold for labeling active compounds.")
    parser.add_argument('--inactive_threshold', type=float, default=10000, help="(ChEMBL) Threshold for labeling inactive compounds.")
    parser.add_argument('--smiles_column', type=str, default=None, help="Name of the SMILES column if not auto-detected.")
    parser.add_argument('--properties', type=str, default=None,
                        help="Comma-separated list of property column names to retain.")
    parser.add_argument('--require_neutral_charge', action='store_true',
                        help="If set, drop molecules with non-zero formal charge.")
    parser.add_argument('--prefer_largest_fragment', action='store_true',
                        help="If set, keep the largest fragment by heavy-atom count when multiple fragments are present.")
    parser.add_argument('--no_prefer_largest_fragment', dest='prefer_largest_fragment', action='store_false',
                        help="Disable largest-fragment preference and keep the first valid fragment.")
    parser.add_argument('--keep_all_columns', action='store_true',
                        help="If set, keep all columns (do not drop to only selected properties).")
    parser.add_argument('--dedupe_strategy', type=str, default=None,
                        help="Duplicate handling strategy: keep_first|keep_last|first|last|drop_conflicts|majority.")
    parser.add_argument('--label_column', type=str, default=None,
                        help="Label column name for dedupe resolution (required for conflict handling).")
    parser.add_argument('--target_column', type=str, default=None,
                        help="Optional target column to enforce non-null values during curate.")
    parser.add_argument('--required_non_null_columns', type=str, default=None,
                        help="Optional comma-separated columns that must be non-null; rows missing values are dropped.")
    parser.add_argument('--row_filters', type=str, default=None,
                        help="Optional JSON object mapping column names to allowed exact values.")
    parser.add_argument('--drop_missing_smiles', action='store_true',
                        help="Drop rows with missing SMILES during curate preprocessing.")
    parser.add_argument('--no_drop_missing_smiles', dest='drop_missing_smiles', action='store_false',
                        help="Keep rows with missing SMILES during curate preprocessing.")
    parser.add_argument('--drop_missing_target', action='store_true',
                        help="Drop rows with missing target_column values during curate when target_column exists.")
    parser.add_argument('--no_drop_missing_target', dest='drop_missing_target', action='store_false',
                        help="Keep rows with missing target_column values during curate.")
    parser.add_argument('--drop_invalid_smiles', action='store_true',
                        help="Drop rows that fail SMILES sanitization/canonicalization.")
    parser.add_argument('--no_drop_invalid_smiles', dest='drop_invalid_smiles', action='store_false',
                        help="Keep rows even when SMILES sanitization fails.")
    parser.set_defaults(
        prefer_largest_fragment=True,
        drop_missing_smiles=True,
        drop_missing_target=True,
        drop_invalid_smiles=True,
    )
    args = parser.parse_args()

    props = None
    if args.properties:
        props = [p.strip() for p in args.properties.split(",") if p.strip()]
    required_cols = None
    if args.required_non_null_columns:
        required_cols = [c.strip() for c in args.required_non_null_columns.split(",") if c.strip()]
    row_filters = None
    if args.row_filters:
        row_filters = json.loads(args.row_filters)

    main(
        args.raw_data_file,
        args.preprocessed_file,
        args.curated_file,
        args.curated_smiles_output,
        args.active_threshold,
        args.inactive_threshold,
        smiles_column=args.smiles_column,
        properties_of_interest=props,
        require_neutral_charge=args.require_neutral_charge,
        prefer_largest_fragment=args.prefer_largest_fragment,
        keep_all_columns=args.keep_all_columns,
        dedupe_strategy=args.dedupe_strategy,
        label_column=args.label_column,
        drop_missing_smiles=args.drop_missing_smiles,
        target_column=args.target_column,
        drop_missing_target=args.drop_missing_target,
        required_non_null_columns=required_cols,
        drop_invalid_smiles=args.drop_invalid_smiles,
        row_filters=row_filters,
    )
