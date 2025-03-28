# import libraries
import subprocess
import os

# Lipinski rules (rule of five) application.
script_path = os.path.join('utilities', 'Lipinski.py')
smiles_file = 'data/covid19_bio_curated_smiles.csv'
output_file = 'data/lipinski_results.csv'
subprocess.run(['python', script_path, smiles_file, output_file])
