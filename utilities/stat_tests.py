import pandas as pd
import argparse
import logging
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
import numpy as np

class StatisticalTests:
    """Class to perform various statistical tests."""
    
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        logging.info(f"Initialized StatisticalTests with input file: {self.input_file}")
    
    def load_data(self):
        """Load the CSV data file."""
        try:
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            if 'class' not in df.columns:
                raise ValueError(f"'class' column not found in {self.input_file}")
            return df
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            raise
    
    def mannwhitney_test(self, df, descriptor):
        """Perform the Mann-Whitney U test on the given descriptor."""
        try:
            logging.info(f"Performing Mann-Whitney U test for descriptor: {descriptor}")
            active = df[df['class'] == 'active'][descriptor]
            inactive = df[df['class'] == 'inactive'][descriptor]

            stat, p = mannwhitneyu(active, inactive)
            alpha = 0.05
            interpretation = 'Same distribution (fail to reject H0)' if p > alpha else 'Different distribution (reject H0)'

            results = pd.DataFrame({'Descriptor': [descriptor],
                                    'Statistics': [stat],
                                    'p': [p],
                                    'alpha': [alpha],
                                    'Interpretation': [interpretation]})

            filename = f"{self.output_dir}/mannwhitneyu_{descriptor}.csv"
            results.to_csv(filename, index=False)
            logging.info(f"Mann-Whitney U test results saved to {filename}")
            return results
        except Exception as e:
            logging.error(f"Error in Mann-Whitney U test: {e}")
            raise
    
    def ttest(self, df, descriptor):
        """Perform the independent t-test on the given descriptor."""
        try:
            logging.info(f"Performing t-test for descriptor: {descriptor}")
            active = df[df['class'] == 'active'][descriptor]
            inactive = df[df['class'] == 'inactive'][descriptor]

            stat, p = ttest_ind(active, inactive)
            alpha = 0.05
            interpretation = 'Same distribution (fail to reject H0)' if p > alpha else 'Different distribution (reject H0)'

            results = pd.DataFrame({'Descriptor': [descriptor],
                                    'Statistics': [stat],
                                    'p': [p],
                                    'alpha': [alpha],
                                    'Interpretation': [interpretation]})

            filename = f"{self.output_dir}/ttest_{descriptor}.csv"
            results.to_csv(filename, index=False)
            logging.info(f"t-test results saved to {filename}")
            return results
        except Exception as e:
            logging.error(f"Error in t-test: {e}")
            raise

    def chi_squared_test(self, df, descriptor):
        """Perform the chi-squared test on the given descriptor."""
        try:
            logging.info(f"Performing chi-squared test for descriptor: {descriptor}")
            active = df[df['class'] == 'active'][descriptor]
            inactive = df[df['class'] == 'inactive'][descriptor]

            # Build contingency table
            contingency_table = pd.crosstab(df['class'], df[descriptor])
            stat, p, dof, expected = chi2_contingency(contingency_table)
            alpha = 0.05
            interpretation = 'Same distribution (fail to reject H0)' if p > alpha else 'Different distribution (reject H0)'

            results = pd.DataFrame({'Descriptor': [descriptor],
                                    'Statistics': [stat],
                                    'p': [p],
                                    'alpha': [alpha],
                                    'Interpretation': [interpretation]})

            filename = f"{self.output_dir}/chi2_{descriptor}.csv"
            results.to_csv(filename, index=False)
            logging.info(f"Chi-squared test results saved to {filename}")
            return results
        except Exception as e:
            logging.error(f"Error in chi-squared test: {e}")
            raise

    def run_test(self, test_type, descriptor):
        """Run the specified statistical test."""
        # Load data
        df = self.load_data()

        # Run the desired test
        if test_type == 'mannwhitney':
            return self.mannwhitney_test(df, descriptor)
        elif test_type == 'ttest':
            return self.ttest(df, descriptor)
        elif test_type == 'chi2':
            return self.chi_squared_test(df, descriptor)
        else:
            logging.error(f"Unknown test type: {test_type}")
            raise ValueError(f"Unknown test type: {test_type}")

def main(input_file, output_dir, test_type, descriptor):
    """Main function to run statistical tests."""
    logging.basicConfig(level=logging.INFO)

    # Initialize the StatisticalTests class
    tester = StatisticalTests(input_file, output_dir)

    # Run the specified test
    try:
        tester.run_test(test_type, descriptor)
    except Exception as e:
        logging.error(f"An error occurred during the statistical test: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical tests on bioactivity data.")
    parser.add_argument('input_file', type=str, help="Input CSV file containing bioactivity data (e.g., 2-class dataset with pIC50 values).")
    parser.add_argument('output_dir', type=str, help="Directory to save the statistical test results.")
    parser.add_argument('test_type', type=str, choices=['mannwhitney', 'ttest', 'chi2'], help="Type of statistical test to perform (mannwhitney, ttest, chi2).")
    parser.add_argument('descriptor', type=str, help="Descriptor/column on which to perform the test (e.g., pIC50, MolecularWeight).")
    
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.test_type, args.descriptor)

