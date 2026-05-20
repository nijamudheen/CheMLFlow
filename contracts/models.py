from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractSpec(BaseModel):
    name: str
    output_path: Optional[str] = None
    output_kind: Literal["csv", "dir", "file"] = "csv"
    description: Optional[str] = None
    required_columns: List[str] = Field(default_factory=list)
    required_any_of: List[List[str]] = Field(default_factory=list)
    min_columns: int = 0
    sample_model: Optional[type[BaseModel]] = None


class BaseRow(BaseModel):
    model_config = ConfigDict(extra="allow")


class RawBioactivityRow(BaseRow):
    molecule_chembl_id: Optional[str] = None
    mol_id: Optional[str] = None
    compound_id: Optional[str] = None
    id: Optional[str] = None
    canonical_smiles: Optional[str] = None
    smiles: Optional[str] = None
    SMILES: Optional[str] = None
    Smiles: Optional[str] = None
    standard_value: Optional[float] = None
    standard_type: Optional[str] = None


class PreprocessedRow(BaseRow):
    canonical_smiles: Optional[str] = None
    molecule_chembl_id: Optional[str] = None
    standard_value: Optional[float] = None


class CuratedRow(BaseRow):
    canonical_smiles: Optional[str] = None
    class_: Optional[str] = Field(default=None, alias="class")


class LipinskiRow(BaseRow):
    canonical_smiles: Optional[str] = None
    class_: Optional[str] = Field(default=None, alias="class")


class PIC50Row(BaseRow):
    pIC50: Optional[float] = None
    canonical_smiles: Optional[str] = None
    class_: Optional[str] = Field(default=None, alias="class")


class DescriptorRow(BaseRow):
    # Descriptors are dynamic; we only assert that at least one column exists.
    # Validation happens at the contract level.
    pass


def make_target_column_contract(
    name: str,
    target_column: str,
    output_path: Optional[str] = None,
    output_kind: Literal["csv", "dir", "file"] = "csv",
    description: Optional[str] = None,
) -> ContractSpec:
    return ContractSpec(
        name=name,
        output_path=output_path,
        output_kind=output_kind,
        description=description,
        required_columns=[target_column],
    )


GET_DATA_OUTPUT_CONTRACT = ContractSpec(
    name="get_data_output",
    description="Raw dataset exported by get_data.",
    min_columns=1,
    sample_model=None,
)

GET_DATA_INPUT_CONTRACT = ContractSpec(
    name="get_data_input",
    output_kind="file",
    description="Pipeline config file must exist for get_data.",
    sample_model=None,
)

CURATE_INPUT_CONTRACT = ContractSpec(
    name="curate_input",
    description="Raw dataset CSV for curate step.",
    min_columns=1,
    sample_model=None,
)

PREPROCESSED_CONTRACT = ContractSpec(
    name="preprocessed",
    description="Preprocessed output from curate (may include canonical_smiles).",
    required_columns=["canonical_smiles"],
    sample_model=PreprocessedRow,
)

CURATE_OUTPUT_CONTRACT = ContractSpec(
    name="curate_output",
    description="Curated dataset; requires canonical_smiles for downstream chemistry steps.",
    required_columns=["canonical_smiles"],
    sample_model=CuratedRow,
)

FEATURIZE_LIPINSKI_INPUT_CONTRACT = ContractSpec(
    name="featurize_lipinski_input",
    description="Requires canonical_smiles; remove featurize.lipinski if dataset lacks SMILES.",
    required_columns=["canonical_smiles"],
    sample_model=CuratedRow,
)

FEATURIZE_LIPINSKI_OUTPUT_CONTRACT = ContractSpec(
    name="featurize_lipinski_output",
    description="Lipinski descriptors appended to input dataset.",
    required_columns=["canonical_smiles"],
    sample_model=LipinskiRow,
)

LABEL_IC50_INPUT_CONTRACT = ContractSpec(
    name="label_ic50_input",
    description="Requires standard_value; remove label.ic50 for non-IC50 datasets.",
    required_columns=["standard_value"],
    sample_model=None,
)

LABEL_IC50_OUTPUT_3CLASS_CONTRACT = ContractSpec(
    name="label_ic50_output_3class",
    description="pIC50 3-class labels output.",
    required_columns=["pIC50", "canonical_smiles"],
    sample_model=PIC50Row,
)

LABEL_IC50_OUTPUT_2CLASS_CONTRACT = ContractSpec(
    name="label_ic50_output_2class",
    description="pIC50 2-class labels output.",
    required_columns=["pIC50", "canonical_smiles"],
    sample_model=PIC50Row,
)

ANALYZE_STATS_INPUT_CONTRACT = ContractSpec(
    name="analyze_stats_input",
    description="2-class labeled CSV for statistical tests.",
    required_columns=["pIC50"],
    sample_model=PIC50Row,
)

ANALYZE_EDA_GENERIC_INPUT_CONTRACT = ContractSpec(
    name="analyze_eda_generic_input",
    description="Generic tabular CSV input for EDA.",
    min_columns=1,
    sample_model=None,
)

ANALYZE_STATS_OUTPUT_CONTRACT = ContractSpec(
    name="analyze_stats_output",
    output_kind="dir",
    description="Output directory for statistical test artifacts.",
    sample_model=None,
)

ANALYZE_EDA_INPUT_2CLASS_CONTRACT = ContractSpec(
    name="analyze_eda_input_2class",
    description="2-class labeled CSV for EDA.",
    required_columns=["pIC50"],
    sample_model=PIC50Row,
)

ANALYZE_EDA_INPUT_3CLASS_CONTRACT = ContractSpec(
    name="analyze_eda_input_3class",
    description="3-class labeled CSV for EDA.",
    required_columns=["pIC50"],
    sample_model=PIC50Row,
)

ANALYZE_EDA_OUTPUT_CONTRACT = ContractSpec(
    name="analyze_eda_output",
    output_kind="dir",
    description="Output directory for EDA artifacts.",
    sample_model=None,
)

FEATURIZE_RDKIT_INPUT_CONTRACT = ContractSpec(
    name="featurize_rdkit_input",
    description="Requires canonical_smiles; remove featurize.rdkit if dataset lacks SMILES.",
    required_columns=["canonical_smiles"],
    sample_model=PIC50Row,
)

FEATURIZE_RDKIT_OUTPUT_CONTRACT = ContractSpec(
    name="featurize_rdkit_output",
    description="RDKit descriptor features generated from SMILES.",
    min_columns=1,
    sample_model=DescriptorRow,
)

FEATURIZE_RDKIT_LABELED_INPUT_CONTRACT = ContractSpec(
    name="featurize_rdkit_labeled_input",
    description="Requires canonical_smiles; remove featurize.rdkit_labeled if dataset lacks SMILES.",
    required_columns=["canonical_smiles"],
    sample_model=CuratedRow,
)

FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT = ContractSpec(
    name="featurize_rdkit_labeled_output_labels",
    description="RDKit labeled descriptor file (features + labels).",
    min_columns=1,
    sample_model=DescriptorRow,
)

FEATURIZE_MORGAN_INPUT_CONTRACT = ContractSpec(
    name="featurize_morgan_input",
    description="Requires canonical_smiles; remove featurize.morgan if dataset lacks SMILES.",
    required_columns=["canonical_smiles"],
    sample_model=CuratedRow,
)

FEATURIZE_MORGAN_OUTPUT_CONTRACT = ContractSpec(
    name="featurize_morgan_output",
    description="Morgan fingerprint features generated from SMILES.",
    min_columns=1,
    sample_model=DescriptorRow,
)

PREPROCESS_FEATURES_INPUT_CONTRACT = ContractSpec(
    name="preprocess_features_input",
    description="Descriptor feature matrix for preprocessing.",
    min_columns=1,
    sample_model=DescriptorRow,
)

PREPROCESS_FEATURES_OUTPUT_CONTRACT = ContractSpec(
    name="preprocess_features_output",
    description="Preprocessed feature matrix (scaled/filtered).",
    min_columns=1,
    sample_model=DescriptorRow,
)

PREPROCESS_LABELS_OUTPUT_CONTRACT = ContractSpec(
    name="preprocess_labels_output",
    description="Aligned labels for preprocessed features.",
    required_any_of=[["pIC50", "gap", "standard_value", "label"]],
    sample_model=None,
)

SELECT_FEATURES_INPUT_FEATURES_CONTRACT = ContractSpec(
    name="select_features_input_features",
    description="Preprocessed feature matrix for stable feature selection.",
    min_columns=1,
    sample_model=DescriptorRow,
)

SELECT_FEATURES_INPUT_LABELS_CONTRACT = ContractSpec(
    name="select_features_input_labels",
    description="Labels for stable feature selection.",
    required_any_of=[["pIC50", "gap", "standard_value", "label"]],
    sample_model=None,
)

SELECT_FEATURES_OUTPUT_CONTRACT = ContractSpec(
    name="select_features_output",
    description="Selected feature matrix.",
    min_columns=1,
    sample_model=DescriptorRow,
)

SELECT_FEATURES_LIST_CONTRACT = ContractSpec(
    name="select_features_list",
    output_kind="file",
    description="Text file containing selected feature names.",
    sample_model=None,
)

PREPROCESS_ARTIFACTS_CONTRACT = ContractSpec(
    name="preprocess_artifacts",
    output_kind="file",
    description="Serialized preprocessing artifacts (scaler/selector/config).",
    sample_model=None,
)

EXPLAIN_INPUT_MODEL_CONTRACT = ContractSpec(
    name="explain_input_model",
    output_kind="file",
    description="Serialized trained model for explainability.",
    sample_model=None,
)

EXPLAIN_OUTPUT_CONTRACT = ContractSpec(
    name="explain_output",
    output_kind="dir",
    description="Explainability artifacts (plots, importance files).",
    sample_model=None,
)

TRAIN_INPUT_FEATURES_CONTRACT = ContractSpec(
    name="train_input_features",
    description="Feature matrix for training.",
    min_columns=1,
    sample_model=DescriptorRow,
)

TRAIN_INPUT_LABELS_CONTRACT = ContractSpec(
    name="train_input_labels",
    description="Labels for training; must include target column.",
    min_columns=1,
    sample_model=None,
)

TRAIN_OUTPUT_CONTRACT = ContractSpec(
    name="train_output",
    output_kind="dir",
    description="Output directory for trained model artifacts.",
    sample_model=None,
)

LIPINSKI_CONTRACT = ContractSpec(
    name="lipinski",
    required_columns=["canonical_smiles"],
    sample_model=LipinskiRow,
)

PIC50_3CLASS_CONTRACT = ContractSpec(
    name="pic50_3class",
    required_columns=["pIC50", "canonical_smiles"],
    sample_model=PIC50Row,
)

PIC50_2CLASS_CONTRACT = ContractSpec(
    name="pic50_2class",
    required_columns=["pIC50", "canonical_smiles"],
    sample_model=PIC50Row,
)

DESCRIPTORS_CONTRACT = ContractSpec(
    name="descriptors",
    min_columns=1,
    sample_model=DescriptorRow,
)

MODEL_LABELS_CONTRACT = ContractSpec(
    name="model_labels",
    required_columns=["pIC50"],
    sample_model=PIC50Row,
)

QM9_LABELED_CONTRACT = ContractSpec(
    name="qm9_labeled_descriptors",
    required_columns=[],
    sample_model=None,
)

IC50_INPUT_CONTRACT = ContractSpec(
    name="ic50_input",
    required_columns=["standard_value"],
    sample_model=None,
)
