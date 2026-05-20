import pandas as pd
import argparse
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors

class RDKitDescriptorCalculator:
    """Class to calculate RDKit descriptors for molecules."""
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        logging.info(f"Initialized RDKitDescriptorCalculator with input file: {self.input_file}")
    
    def load_data(self):
        """Load the bioactivity dataset with canonical_smiles."""
        try:
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            if 'canonical_smiles' not in df.columns:
                raise ValueError(f"Input file must contain 'canonical_smiles' column.")
            return df['canonical_smiles']
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            raise

    def calculate_descriptors(self, smiles_list):
        """Calculate RDKit descriptors for a list of SMILES strings."""
        try:
            logging.info("Calculating RDKit descriptors.")
            mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
            descriptors_list = []

            for mol in mols:
                if mol is not None:
                    descriptors_list.append(self.get_molecule_descriptors(mol))
                else:
                    descriptors_list.append({desc_name: None for desc_name, _ in Descriptors._descList})

            # Convert list of descriptors into a DataFrame
            df_descriptors = pd.DataFrame(descriptors_list)
            logging.info("Descriptor calculation completed.")
            return df_descriptors
        except Exception as e:
            logging.error(f"Error calculating RDKit descriptors: {e}")
            raise

    def get_molecule_descriptors(self, mol):
        """Calculate RDKit molecular descriptors for a single molecule."""
        res = {}
        for desc_name, desc_fn in Descriptors._descList:
            try:
                res[desc_name] = desc_fn(mol)
            except Exception as e:
                logging.warning(f"Error calculating descriptor {desc_name}: {e}")
                res[desc_name] = None
        return res

    def save_descriptors(self, descriptors_df):
        """Save the RDKit descriptors DataFrame."""
        try:
            logging.info(f"Saving RDKit descriptors to {self.output_file}")
            
            # Save the descriptors only
            descriptors_df.to_csv(self.output_file, index=False)
            logging.info(f"RDKit descriptors saved successfully to {self.output_file}")
        except Exception as e:
            logging.error(f"Error saving the descriptors: {e}")
            raise

def main(input_file, output_file):
    """Main function to calculate RDKit descriptors and save only the descriptors."""
    logging.basicConfig(level=logging.INFO)

    # Initialize the descriptor calculator
    calculator = RDKitDescriptorCalculator(input_file, output_file)

    # Load the dataset and extract the SMILES strings
    smiles_list = calculator.load_data()

    # Calculate RDKit descriptors
    descriptors_df = calculator.calculate_descriptors(smiles_list)

    # Save only the RDKit descriptors
    calculator.save_descriptors(descriptors_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RDKit descriptors for a dataset and save only the descriptors.")
    parser.add_argument('input_file', type=str, help="Input CSV file containing canonical_smiles.")
    parser.add_argument('output_file', type=str, help="Output CSV file to save the RDKit descriptors.")
    
    args = parser.parse_args()

    main(args.input_file, args.output_file)
