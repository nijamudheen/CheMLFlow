import numpy as np
import pandas as pd
import argparse
import logging
import sys

try:
    from chembl_webresource_client.new_client import new_client
except Exception as exc:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.error("ChEMBL API is unavailable: %s", exc)
    print(
        "ChEMBL API is unavailable right now. Please retry later or switch to a cached/local CSV."
    )
    raise SystemExit(1)


class ChemblDataRetriever:
    """Class to retrieve bioactivity data from ChEMBL for a given target name."""

    def __init__(self, target_name, bioactivity, target_chembl_id=None):
        """Initialize the data retriever with the target name and bioactivity setting."""
        self.target_name = target_name
        self.target_chembl_id = str(target_chembl_id or "").strip()
        self.target = new_client.target
        self.activity = new_client.activity
        self.selected_targets = []
        # Normalize bioactivity input:
        # if bioactivity is ['all'], set it to None to indicate no standard_type filter
        if isinstance(bioactivity, list):
            if len(bioactivity) == 1 and bioactivity[0].lower() == 'all':
                self.bioactivity = None
            else:
                self.bioactivity = bioactivity
        else:
            # If single string is provided
            if bioactivity.lower() == 'all':
                self.bioactivity = None
            else:
                self.bioactivity = [bioactivity]

        logging.info(
            "Initialized ChemblDataRetriever for target=%s target_chembl_id=%s bioactivity=%s",
            self.target_name,
            self.target_chembl_id or "(search)",
            self.bioactivity or "all",
        )

    def search_target(self):
        """
        Search for targets matching the target name and select all relevant targets.

        Returns:
            A list of ChEMBL IDs for the selected targets.
        """
        try:
            if self.target_chembl_id:
                self.selected_targets = [self.target_chembl_id]
                logging.info("Using pinned target ChEMBL ID: %s", self.target_chembl_id)
                return self.selected_targets

            logging.info(f"Searching for target: {self.target_name}")
            target_query = self.target.search(self.target_name)
            if not target_query:
                logging.error(f"Target '{self.target_name}' not found.")
                raise ValueError(f"Target '{self.target_name}' not found.")

            targets_df = pd.DataFrame.from_dict(target_query)
            self.selected_targets = targets_df['target_chembl_id'].tolist()
            logging.info(f"Selected target ChEMBL IDs: {self.selected_targets}")
            return self.selected_targets
        except Exception as e:
            logging.error(f"Error in target search: {e}")
            raise

    def retrieve_bioactivity_data(self):
        """
        Retrieve bioactivity data for the selected targets.

        Returns:
            A DataFrame containing bioactivity data, or None if no data is found.
        """
        if not self.selected_targets:
            raise ValueError("Targets not selected. Run search_target first.")

        try:
            logging.info(f"Retrieving bioactivity data for targets: {self.selected_targets}")
            all_data = []
            for target_id in self.selected_targets:
                logging.info(f"Retrieving data for target ID: {target_id}")
                query = self.activity.filter(target_chembl_id=target_id)
                
                # Apply bioactivity filters if specified
                if self.bioactivity is not None:
                    if len(self.bioactivity) == 1:
                        # Single bioactivity type
                        query = query.filter(standard_type=self.bioactivity[0])
                    else:
                        # Multiple bioactivity types
                        # Note: The chembl_webresource_client supports __in queries for many fields.
                        query = query.filter(standard_type__in=self.bioactivity)

                res_list = list(query)
                if res_list:
                    df = pd.json_normalize(res_list)
                    all_data.append(df)
                else:
                    logging.info(f"No data found for target ID: {target_id}")

            if not all_data:
                logging.warning(f"No bioactivity data found for targets {self.selected_targets}.")
                return None

            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Identify and remove unhashable columns (dict, list)
            unhashable_cols = combined_df.columns[
                combined_df.apply(lambda col: col.map(type).isin([dict, list]).any())
            ]
            if not unhashable_cols.empty:
                logging.info(f"Removing unhashable columns: {list(unhashable_cols)}")
                combined_df = combined_df.drop(columns=unhashable_cols)

            combined_df = combined_df.drop_duplicates()
            logging.info(f"Retrieved {len(combined_df)} unique bioactivity records.")
            return combined_df
        except Exception as e:
            logging.error(f"Error retrieving bioactivity data: {e}")
            raise

    def save_data(self, df, filename):
        """
        Save bioactivity data to a CSV file.

        Args:
            df: The DataFrame containing bioactivity data.
            filename: The filename to save the data to.
        """
        try:
            logging.info(f"Saving data to {filename}")
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise


def main(target_name, output_file, bioactivity, target_chembl_id=None):
    """
    Main function to retrieve and save bioactivity data.

    Args:
        target_name: The name of the target to search for.
        output_file: The CSV file to save the bioactivity data.
        bioactivity: A list of bioactivity measures or 'all'.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    retriever = ChemblDataRetriever(target_name, bioactivity, target_chembl_id=target_chembl_id)

    try:
        retriever.search_target()
        bioactivity_data = retriever.retrieve_bioactivity_data()

        if bioactivity_data is not None:
            retriever.save_data(bioactivity_data, output_file)
        else:
            logging.warning(f"No data to save for target: {target_name}")

    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(
            "ChEMBL fetch failed. Please retry later or switch to a cached/local CSV."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save bioactivity data from ChEMBL.")
    parser.add_argument(
        'target_name',
        type=str,
        help="Name of the target to search for (e.g., acetylcholinesterase)"
    )
    parser.add_argument(
        'output_file',
        type=str,
        help="Output CSV file to save the bioactivity data"
    )
    parser.add_argument(
        '--bioactivity',
        nargs='*',
        default=['IC50'],
        help=(
            "Bioactivity type(s) to retrieve. Specify one or more standard_types (e.g. IC50 Ki EC50 Kd) "
            "or 'all' to retrieve all available bioactivities. Common types include: "
            "IC50, Ki, EC50, Kd, MIC, AC50, Potency, LD50, ED50, and more. "
            "Defaults to IC50."
        )
    )
    parser.add_argument(
        '--target-chembl-id',
        type=str,
        default='',
        help="Optional exact ChEMBL target ID to pin (for example CHEMBL3885651).",
    )

    args = parser.parse_args()
    main(args.target_name, args.output_file, args.bioactivity, target_chembl_id=args.target_chembl_id)
