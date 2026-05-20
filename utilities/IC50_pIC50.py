import numpy as np
import pandas as pd
import argparse
import logging

class IC50Converter:
    """Class to handle IC50 normalization and conversion to pIC50."""
    
    def __init__(self, input_file, output_file_3class, output_file_2class):
        self.input_file = input_file
        self.output_file_3class = output_file_3class
        self.output_file_2class = output_file_2class
        logging.info(f"Initialized IC50Converter with input file: {self.input_file}")

    def load_data(self):
        """Load the input data from CSV."""
        try:
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            if 'standard_value' not in df.columns:
                raise ValueError(f"'standard_value' column not found in {self.input_file}")
            return df
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            raise

    def normalize_values(self, df):
        """Normalize IC50 values, capping them at 100,000,000 nM."""
        try:
            logging.info("Normalizing IC50 values.")
            df['standard_value_norm'] = df['standard_value'].apply(lambda x: min(x, 100000000))
            df_normalized = df.drop('standard_value', axis=1)
            logging.info("IC50 normalization completed.")
            return df_normalized
        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            raise

    def convert_to_pIC50(self, df):
        """Convert normalized IC50 to pIC50 using -log10(IC50)."""
        try:
            logging.info("Converting IC50 to pIC50.")
            df['pIC50'] = df['standard_value_norm'].apply(lambda x: -np.log10(x * 10**-9))
            df_converted = df.drop('standard_value_norm', axis=1)
            logging.info("Conversion to pIC50 completed.")
            return df_converted
        except Exception as e:
            logging.error(f"Error during pIC50 conversion: {e}")
            raise

    def save_data(self, df_3class, df_2class):
        """Save the processed data to CSV files."""
        try:
            logging.info(f"Saving 3-class data to {self.output_file_3class}")
            df_3class.to_csv(self.output_file_3class, index=False)
            logging.info(f"Saving 2-class data to {self.output_file_2class}")
            df_2class.to_csv(self.output_file_2class, index=False)
            logging.info("Data saved successfully.")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise

    def process_data(self):
        """Main process to normalize IC50, convert to pIC50, and save results."""
        # Step 1: Load the input data
        df = self.load_data()

        # Step 2: Normalize the IC50 values
        df_normalized = self.normalize_values(df)

        # Step 3: Convert IC50 to pIC50
        df_pIC50 = self.convert_to_pIC50(df_normalized)

        # Step 4: Handle 3-class and 2-class datasets
        logging.info("Processing 3-class and 2-class datasets.")
        df_3class = df_pIC50

        # For 2-class, filter out the 'intermediate' class and drop NaN pIC50 values
        df_2class = df_3class[df_3class['class'] != 'intermediate'].dropna(subset=['pIC50'])
        df_2class = df_2class[np.isfinite(df_2class['pIC50'])]

        # Step 5: Save the processed data to CSV files
        self.save_data(df_3class, df_2class)

def main(input_file, output_file_3class, output_file_2class):
    """Main function to handle IC50 normalization and pIC50 conversion."""
    logging.basicConfig(level=logging.INFO)

    # Initialize the converter class
    converter = IC50Converter(input_file, output_file_3class, output_file_2class)

    try:
        # Process the data: normalize, convert, and save
        converter.process_data()
    except Exception as e:
        logging.error(f"An error occurred during the IC50 to pIC50 conversion process: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IC50 to pIC50 and normalize IC50 values.")
    parser.add_argument('input_file', type=str, help="Input CSV file with bioactivity data (e.g., curated_smiles.csv or lipinski_results.csv).")
    parser.add_argument('output_file_3class', type=str, help="Output CSV file for 3-class bioactivity data with pIC50.")
    parser.add_argument('output_file_2class', type=str, help="Output CSV file for 2-class bioactivity data with pIC50.")
    
    args = parser.parse_args()

    main(args.input_file, args.output_file_3class, args.output_file_2class)
