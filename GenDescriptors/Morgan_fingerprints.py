import argparse
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROW_INDEX_COL = "__row_index"


def _find_smiles_column(columns: List[str]) -> Optional[str]:
    candidates = ["canonical_smiles", "smiles", "SMILES", "Smiles", "Drug", "drug"]
    for name in candidates:
        if name in columns:
            return name
    return None


def _morgan_fingerprint(smiles: str, generator) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main(
    input_file: str,
    output_file: str,
    radius: int,
    n_bits: int,
    labeled_output_file: Optional[str],
    property_columns: List[str],
):
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(input_file)
    smiles_col = _find_smiles_column(list(df.columns))
    if smiles_col is None:
        raise ValueError("Input file must contain a SMILES column.")

    if ROW_INDEX_COL not in df.columns:
        df[ROW_INDEX_COL] = df.index.astype(int)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps = []
    valid_rows: List[int] = []
    row_ids: List[int] = []
    for idx, smi in enumerate(df[smiles_col].astype(str).tolist()):
        arr = _morgan_fingerprint(smi, generator)
        if arr is None:
            continue
        fps.append(arr)
        valid_rows.append(idx)
        row_ids.append(int(df.iloc[idx][ROW_INDEX_COL]))

    if not fps:
        raise ValueError("No valid SMILES for Morgan fingerprint generation.")

    fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(n_bits)])
    fp_df[ROW_INDEX_COL] = row_ids
    fp_df.to_csv(output_file, index=False)
    logging.info("Morgan fingerprints saved to %s", output_file)

    if labeled_output_file:
        labels = df.iloc[valid_rows][[c for c in property_columns if c in df.columns]].copy()
        labels[ROW_INDEX_COL] = [int(df.iloc[i][ROW_INDEX_COL]) for i in valid_rows]
        if labels.empty:
            logging.warning("No property columns found; labeled output will include fingerprints only.")
            combined = fp_df
        else:
            combined = pd.concat(
                [
                    fp_df.reset_index(drop=True),
                    labels.reset_index(drop=True).drop(columns=[ROW_INDEX_COL]),
                ],
                axis=1,
            )
        combined.to_csv(labeled_output_file, index=False)
        logging.info("Labeled fingerprints saved to %s", labeled_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Morgan fingerprints from SMILES.")
    parser.add_argument("input_file", type=str, help="Input CSV with SMILES.")
    parser.add_argument("output_file", type=str, help="Output CSV for Morgan fingerprints.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Fingerprint length.")
    parser.add_argument(
        "--labeled-output-file",
        type=str,
        default=None,
        help="Optional output CSV with fingerprints + labels.",
    )
    parser.add_argument(
        "--property-columns",
        type=str,
        default=None,
        help="Comma-separated property columns to append to labeled output.",
    )

    args = parser.parse_args()
    prop_cols = []
    if args.property_columns:
        prop_cols = [p.strip() for p in args.property_columns.split(",") if p.strip()]

    main(
        args.input_file,
        args.output_file,
        args.radius,
        args.n_bits,
        args.labeled_output_file,
        prop_cols,
    )
