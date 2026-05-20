import argparse
import logging
from typing import List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROW_INDEX_COL = "__row_index"


class RDKitDescriptorCalculator:
    """Calculate RDKit descriptors and attach label columns."""

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        logging.info(f"Initialized RDKitDescriptorCalculator with input file: {self.input_file}")

    def load_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            if "canonical_smiles" not in df.columns:
                raise ValueError("Input file must contain 'canonical_smiles' column.")
            return df
        except Exception as exc:
            logging.error(f"Error loading input data: {exc}")
            raise

    def calculate_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        try:
            logging.info("Calculating RDKit descriptors.")
            mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
            descriptors_list = []

            for mol in mols:
                if mol is not None:
                    descriptors_list.append(self.get_molecule_descriptors(mol))
                else:
                    descriptors_list.append({desc_name: None for desc_name, _ in Descriptors._descList})

            df_descriptors = pd.DataFrame(descriptors_list)
            logging.info("Descriptor calculation completed.")
            return df_descriptors
        except Exception as exc:
            logging.error(f"Error calculating RDKit descriptors: {exc}")
            raise

    def get_molecule_descriptors(self, mol):
        res = {}
        for desc_name, desc_fn in Descriptors._descList:
            try:
                res[desc_name] = desc_fn(mol)
            except Exception as exc:
                logging.warning(f"Error calculating descriptor {desc_name}: {exc}")
                res[desc_name] = None
        return res

    def save_descriptors(self, descriptors_df: pd.DataFrame) -> None:
        try:
            logging.info(f"Saving RDKit descriptors to {self.output_file}")
            descriptors_df.to_csv(self.output_file, index=False)
            logging.info(f"RDKit descriptors saved successfully to {self.output_file}")
        except Exception as exc:
            logging.error(f"Error saving the descriptors: {exc}")
            raise


def main(
    input_file: str,
    output_file: str,
    labeled_output_file: Optional[str],
    property_columns: List[str],
):
    logging.basicConfig(level=logging.INFO)

    calculator = RDKitDescriptorCalculator(input_file, output_file)
    df = calculator.load_data()
    if ROW_INDEX_COL not in df.columns:
        df[ROW_INDEX_COL] = df.index.astype(int)
    else:
        df[ROW_INDEX_COL] = pd.to_numeric(df[ROW_INDEX_COL], errors="raise").astype(int)

    smiles_list = df["canonical_smiles"].astype(str).tolist()
    descriptors_df = calculator.calculate_descriptors(smiles_list)
    descriptors_df[ROW_INDEX_COL] = df[ROW_INDEX_COL].tolist()

    calculator.save_descriptors(descriptors_df)

    if labeled_output_file:
        labels = df[[c for c in property_columns if c in df.columns]].copy()
        labels[ROW_INDEX_COL] = df[ROW_INDEX_COL].tolist()
        if labels.empty:
            logging.warning("No property columns found in input; labeled output will include descriptors only.")
            combined = descriptors_df
        else:
            combined = pd.concat(
                [
                    descriptors_df.reset_index(drop=True),
                    labels.reset_index(drop=True).drop(columns=[ROW_INDEX_COL]),
                ],
                axis=1,
            )
        combined.to_csv(labeled_output_file, index=False)
        logging.info(f"Labeled descriptors saved to {labeled_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate RDKit descriptors and optionally attach label columns."
    )
    parser.add_argument("input_file", type=str, help="Input CSV file containing canonical_smiles.")
    parser.add_argument("output_file", type=str, help="Output CSV file to save the RDKit descriptors.")
    parser.add_argument(
        "--labeled-output-file",
        type=str,
        default=None,
        help="Optional CSV output file containing descriptors + selected property columns.",
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

    main(args.input_file, args.output_file, args.labeled_output_file, prop_cols)
