.PHONY: install env clean

# Configuration
ENV_NAME = chemlflow_env
PYTHON_VERSION = 3.13

# Create Conda environment with Python and rdkit
env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION) rdkit -c conda-forge
	@echo "Conda environment '$(ENV_NAME)' created with Python $(PYTHON_VERSION) and rdkit."
	@echo "Activate it with: conda activate $(ENV_NAME)"

# Install the package inside the environment
install:
	conda run -n $(ENV_NAME) pip install -e .

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info config/generated runs

