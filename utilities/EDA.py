import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging
import argparse
from stat_tests import StatisticalTests

class EDA:
    """Class to perform Exploratory Data Analysis and statistical tests."""
    
    def __init__(self, input_file_2class, input_file_3class, output_dir):
        self.input_file_2class = input_file_2class
        self.input_file_3class = input_file_3class
        self.output_dir = output_dir
        self.ensure_output_dir()
        logging.info(f"Initialized EDA with files: {self.input_file_2class}, {self.input_file_3class}")
    
    def ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logging.info(f"Output directory set to {self.output_dir}")
    
    def load_data(self):
        """Load the 2-class and 3-class datasets."""
        try:
            logging.info(f"Loading data from {self.input_file_2class} and {self.input_file_3class}")
            df_2class = pd.read_csv(self.input_file_2class)
            df_3class = pd.read_csv(self.input_file_3class)
            return df_2class, df_3class
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def plot_and_save(self, plot_func, filename, **kwargs):
        """General function to create and save plots."""
        try:
            plt.figure(figsize=(5.5, 5.5))
            plot_func(**kwargs)
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
            logging.info(f"Saved plot as {filename}")
            plt.close()
        except Exception as e:
            logging.error(f"Error in plotting {filename}: {e}")
            raise

    def generate_plots(self, df_2class):
        """Generate and save EDA plots."""
        # Frequency plot of bioactivity class
        self.plot_and_save(
            sns.countplot, 'plot_bioactivity_class.png', x='class', data=df_2class, edgecolor='black', palette="Set2"
        )

        # Violin plot for pIC50 values
        self.plot_and_save(
            sns.violinplot, 'plot_bioactivity_violin.png', x='class', y='pIC50', data=df_2class, palette="Set2"
        )

        # Scatter plot for MolecularWeight vs LogP
        self.plot_and_save(
            sns.scatterplot, 'plot_MolecularWeight_vs_LogP.png', x='MolecularWeight', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7
        )

        # Box plots for descriptors and pIC50
        descriptors = ['pIC50', 'MolecularWeight', 'LogP', 'HydrogenDonors', 'HydrogenAcceptors']
        for desc in descriptors:
            self.plot_and_save(
                sns.boxplot, f'plot_{desc}.png', x='class', y=desc, data=df_2class, palette="Set2"
            )
    
    def run_statistical_tests(self, df_2class):
        """Run statistical tests (Mann-Whitney U, t-test, chi-squared) for descriptors and save results."""
        tester = StatisticalTests(self.input_file_2class, self.output_dir)
        results = []

        descriptors = ['MolecularWeight', 'LogP', 'HydrogenDonors', 'HydrogenAcceptors', 'pIC50']
        for desc in descriptors:
            # Mann-Whitney U Test
            logging.info(f"Running Mann-Whitney U test for {desc}")
            mw_result = tester.run_test('mannwhitney', desc)
            mw_result['Test'] = 'Mann-Whitney U'
            results.append(mw_result)

            # t-Test
            logging.info(f"Running t-test for {desc}")
            ttest_result = tester.run_test('ttest', desc)
            ttest_result['Test'] = 't-test'
            results.append(ttest_result)

        # Chi-Squared Test for the 'class' column
        logging.info(f"Running chi-squared test for 'class'")
        chi2_result = tester.run_test('chi2', 'class')
        chi2_result['Test'] = 'chi-squared'
        results.append(chi2_result)

        # Combine all results into a DataFrame and save as CSV
        combined_results = pd.concat(results, ignore_index=True)
        combined_results_path = os.path.join(self.output_dir, 'statistical_test_results.csv')
        combined_results.to_csv(combined_results_path, index=False)
        logging.info(f"Statistical test results saved to {combined_results_path}")

    def create_grouped_figures(self):
        """Create a figure that groups all important figures."""
        try:
            logging.info("Generating grouped figure.")
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Load and display individual plots
            img_paths = [
                'plot_bioactivity_violin.png', 'plot_MolecularWeight_vs_LogP.png', 'plot_MolecularWeight.png', 'plot_LogP.png'
            ]
            for i, img_path in enumerate(img_paths):
                img = plt.imread(os.path.join(self.output_dir, img_path))
                axs[i//2, i%2].imshow(img)
                axs[i//2, i%2].axis('off')

            # Save grouped figure
            grouped_figure_path = os.path.join(self.output_dir, 'grouped_figure.png')
            plt.savefig(grouped_figure_path, bbox_inches='tight')
            logging.info(f"Grouped figure saved to {grouped_figure_path}")
            plt.close()
        except Exception as e:
            logging.error(f"Error creating grouped figure: {e}")
            raise

def main(input_file_2class, input_file_3class, output_dir):
    logging.basicConfig(level=logging.INFO)

    # Initialize the EDA process
    eda = EDA(input_file_2class, input_file_3class, output_dir)

    # Load the data
    df_2class, df_3class = eda.load_data()

    # Generate plots
    eda.generate_plots(df_2class)

    # Run statistical tests
    eda.run_statistical_tests(df_2class)

    # Generate grouped figure
    eda.create_grouped_figures()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis and Statistical Tests.")
    parser.add_argument('input_file_2class', type=str, help="Input CSV file containing 2-class bioactivity data.")
    parser.add_argument('input_file_3class', type=str, help="Input CSV file containing 3-class bioactivity data.")
    parser.add_argument('output_dir', type=str, help="Directory to save EDA figures and test results.")

    args = parser.parse_args()
    
    main(args.input_file_2class, args.input_file_3class, args.output_dir)
