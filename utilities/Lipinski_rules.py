import numpy as np
import pandas as pd
import argparse
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

class LipinskiRuleEvaluator:
    """Class to evaluate ADME profiles using Lipinski's Rule of Five."""
    
    def __init__(self, smiles_file, output_file):
        self.smiles_file = smiles_file
        self.output_file = output_file
        logging.info(f"Initialized LipinskiRuleEvaluator with SMILES file: {self.smiles_file}")

    def load_smiles(self):
        """Load SMILES data from a CSV file."""
        try:
            logging.info(f"Loading SMILES data from {self.smiles_file}")
            df = pd.read_csv(self.smiles_file)
            if 'canonical_smiles' not in df.columns:
                raise ValueError(f"'canonical_smiles' column not found in {self.smiles_file}.")
            return df
        except Exception as e:
            logging.error(f"Error loading SMILES data: {e}")
            raise

    def compute_lipinski_descriptors(self, smiles):
        """Compute Lipinski descriptors for a list of SMILES."""
        try:
            logging.info("Computing Lipinski descriptors.")
            descriptors_list = []
            for smile in smiles:
                if smile is None or pd.isna(smile):
                    descriptors_list.append([np.nan, np.nan, np.nan, np.nan])
                    continue
                mol = Chem.MolFromSmiles(str(smile))
                if mol is not None:
                    desc_MW = Descriptors.MolWt(mol)
                    desc_LogP = Descriptors.MolLogP(mol)
                    desc_NumHDonors = Descriptors.NumHDonors(mol)
                    desc_NumHAcceptors = Descriptors.NumHAcceptors(mol)
                    descriptors_list.append([desc_MW, desc_LogP, desc_NumHDonors, desc_NumHAcceptors])
                else:
                    descriptors_list.append([np.nan, np.nan, np.nan, np.nan])

            columns = ["MolecularWeight", "LogP", "HydrogenDonors", "HydrogenAcceptors"]
            descriptors_df = pd.DataFrame(descriptors_list, columns=columns)
            if len(descriptors_df) != len(smiles):
                raise ValueError("Lipinski descriptor row count must match input SMILES row count.")
            logging.info("Lipinski descriptors computation completed.")
            return descriptors_df
        except Exception as e:
            logging.error(f"Error during Lipinski descriptor computation: {e}")
            raise

    def save_combined_data(self, original_df, descriptors_df):
        """Save the combined original data and computed descriptors to a CSV file."""
        try:
            logging.info(f"Saving combined data to {self.output_file}")
            combined_df = pd.concat([original_df, descriptors_df], axis=1)
            combined_df.to_csv(self.output_file, index=False)
            logging.info(f"Combined data saved successfully to {self.output_file}")
        except Exception as e:
            logging.error(f"Error saving combined data: {e}")
            raise

def main(smiles_file, output_file):
    """Main function to evaluate Lipinski's Rule of Five for a set of SMILES."""
    logging.basicConfig(level=logging.INFO)

    # Initialize the evaluator class
    evaluator = LipinskiRuleEvaluator(smiles_file, output_file)

    try:
        # Load SMILES data
        df_smiles = evaluator.load_smiles()

        # Compute Lipinski descriptors
        lipinski_descriptors_df = evaluator.compute_lipinski_descriptors(df_smiles['canonical_smiles'])

        # Save the combined data
        evaluator.save_combined_data(df_smiles, lipinski_descriptors_df)

    except Exception as e:
        logging.error(f"An error occurred during Lipinski evaluation: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADME profile using Lipinski's Rule of Five.")
    parser.add_argument('smiles_file', type=str, help="Input CSV file containing SMILES data.")
    parser.add_argument('output_file', type=str, help="Output CSV file to save the combined results.")
    
    args = parser.parse_args()

    main(args.smiles_file, args.output_file)
