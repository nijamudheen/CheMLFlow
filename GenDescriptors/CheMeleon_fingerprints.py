from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from importlib import metadata
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

ROW_INDEX_COL = "__row_index"
FEATURE_COUNT = 2048


class Fingerprinter(Protocol):
    def __call__(self, molecules: list[str]) -> np.ndarray: ...


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(torch_module, requested: str) -> str:
    normalized = str(requested or "auto").strip().lower() or "auto"
    if normalized != "auto":
        return normalized
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class CheMeleonFingerprinter:
    """Frozen CheMeleon message-passing encoder with mean graph pooling."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        try:
            import torch
            from chemprop.data import BatchMolGraph
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            from chemprop.models import MPNN
            from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
        except ImportError as exc:
            raise ImportError(
                "featurize.chemeleon_fp requires Chemprop and PyTorch. "
                "Install the CheMLFlow chemprop extra before running this node."
            ) from exc

        # The Intel macOS PyTorch 2.2 wheel can leave OpenMP workers alive while
        # Python finalizes RDKit/Chemprop objects, causing a shutdown SIGSEGV.
        # This node runs in its own process, so limiting that affected runtime to
        # one Torch thread is isolated from training and the parent process.
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            torch.set_num_threads(1)

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"CheMeleon checkpoint not found: {checkpoint}. "
                "Set featurize.checkpoint to an existing chemeleon_mp.pt file."
            )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("CheMeleon checkpoint must contain a mapping payload.")
        hyper_parameters = payload.get("hyper_parameters")
        state_dict = payload.get("state_dict")
        if not isinstance(hyper_parameters, dict) or not isinstance(state_dict, dict):
            raise ValueError(
                "CheMeleon checkpoint must contain 'hyper_parameters' and 'state_dict' mappings."
            )

        message_passing = BondMessagePassing(**hyper_parameters)
        message_passing.load_state_dict(state_dict)
        if int(message_passing.output_dim) != FEATURE_COUNT:
            raise ValueError(
                "CheMeleon checkpoint output dimension must be 2048; "
                f"got {message_passing.output_dim}."
            )
        model = MPNN(
            message_passing=message_passing,
            agg=MeanAggregation(),
            predictor=RegressionFFN(input_dim=message_passing.output_dim),
        )
        model.requires_grad_(False)
        model.eval()

        self.device = _resolve_device(torch, device)
        model.to(device=self.device)
        self._torch = torch
        self._batch_mol_graph_cls = BatchMolGraph
        self._mol_graph_featurizer = SimpleMoleculeMolGraphFeaturizer()
        self._model = model

    def __call__(self, molecules: list[str]) -> np.ndarray:
        from rdkit import Chem

        mol_graphs = []
        for smiles in molecules:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid SMILES reached CheMeleon encoder: {smiles!r}.")
            mol_graphs.append(self._mol_graph_featurizer(molecule))

        batch_graph = self._batch_mol_graph_cls(mol_graphs)
        batch_graph.to(device=self.device)
        with self._torch.no_grad():
            fingerprints = self._model.fingerprint(batch_graph)
        return fingerprints.detach().cpu().numpy().astype(np.float32, copy=False)


def _normalize_row_ids(df: pd.DataFrame) -> pd.Series:
    if ROW_INDEX_COL not in df.columns:
        return pd.Series(df.index.astype(int), index=df.index, name=ROW_INDEX_COL)
    numeric = pd.to_numeric(df[ROW_INDEX_COL], errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{ROW_INDEX_COL} must contain finite integer row IDs.")
    row_ids = numeric.astype(int)
    if row_ids.duplicated().any():
        duplicates = sorted(row_ids[row_ids.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"{ROW_INDEX_COL} contains duplicate row IDs: {duplicates[:10]}.")
    return row_ids


def _validate_smiles(df: pd.DataFrame, row_ids: pd.Series) -> list[str]:
    from rdkit import Chem

    if "canonical_smiles" not in df.columns:
        raise ValueError("CheMeleon fingerprint input must contain 'canonical_smiles'.")
    smiles_values = df["canonical_smiles"].tolist()
    invalid_row_ids: list[int] = []
    normalized: list[str] = []
    for position, value in enumerate(smiles_values):
        smiles = value.strip() if isinstance(value, str) else ""
        if not smiles or Chem.MolFromSmiles(smiles) is None:
            invalid_row_ids.append(int(row_ids.iloc[position]))
        normalized.append(smiles)
    if invalid_row_ids:
        raise ValueError(
            "CheMeleon fingerprinting received invalid SMILES at row IDs "
            f"{invalid_row_ids[:10]}. Run curation with invalid-SMILES filtering enabled."
        )
    if not normalized:
        raise ValueError("CheMeleon fingerprint input contains no molecules.")
    return normalized


def _batched_fingerprints(
    smiles: Sequence[str],
    fingerprinter: Fingerprinter,
    batch_size: int,
) -> np.ndarray:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("featurize.batch_size must be a positive integer.")
    batches: list[np.ndarray] = []
    for start in range(0, len(smiles), int(batch_size)):
        batch_smiles = list(smiles[start : start + int(batch_size)])
        values = np.asarray(fingerprinter(batch_smiles), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(batch_smiles):
            raise ValueError(
                "CheMeleon encoder returned an invalid batch shape: "
                f"expected ({len(batch_smiles)}, {FEATURE_COUNT}), got {values.shape}."
            )
        batches.append(values)
    fingerprints = np.vstack(batches).astype(np.float32, copy=False)
    if fingerprints.shape != (len(smiles), FEATURE_COUNT):
        raise ValueError(
            "CheMeleon encoder must return one 2048-number fingerprint per molecule; "
            f"got {fingerprints.shape}."
        )
    if not np.isfinite(fingerprints).all():
        raise ValueError("CheMeleon encoder returned non-finite fingerprint values.")
    return fingerprints


def generate_fingerprint_artifacts(
    *,
    input_path: str | Path,
    output_path: str | Path,
    labeled_output_path: str | Path,
    metadata_output_path: str | Path,
    checkpoint_path: str | Path,
    property_columns: Iterable[str],
    batch_size: int = 64,
    device: str = "auto",
    fingerprinter: Fingerprinter | None = None,
) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    labeled_output_file = Path(labeled_output_path)
    metadata_output_file = Path(metadata_output_path)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"CheMeleon checkpoint not found: {checkpoint_file}.")

    df = pd.read_csv(input_file)
    row_ids = _normalize_row_ids(df)
    requested_properties = [str(column) for column in property_columns if str(column)]
    missing_properties = [column for column in requested_properties if column not in df.columns]
    if missing_properties:
        raise ValueError(
            "CheMeleon labeled output is missing requested property columns: "
            + ", ".join(missing_properties)
            + "."
        )

    # On Intel macOS, importing/initializing RDKit before loading this Torch
    # checkpoint can trigger a native-library crash. Construct the real encoder
    # first; injected test/custom fingerprinters remain dependency-light.
    active_fingerprinter = fingerprinter
    if active_fingerprinter is None:
        active_fingerprinter = CheMeleonFingerprinter(
            checkpoint_file,
            device=device,
        )
    smiles = _validate_smiles(df, row_ids)
    fingerprints = _batched_fingerprints(smiles, active_fingerprinter, int(batch_size))
    feature_columns = [f"chemeleon_fp_{index:04d}" for index in range(FEATURE_COUNT)]
    features = pd.DataFrame(fingerprints, columns=feature_columns)
    features[ROW_INDEX_COL] = row_ids.to_numpy(dtype=int)
    labels = df[requested_properties].reset_index(drop=True)
    labeled = pd.concat([features.reset_index(drop=True), labels], axis=1)

    for destination in (output_file, labeled_output_file, metadata_output_file):
        destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)
    labeled.to_csv(labeled_output_file, index=False)

    resolved_device = str(getattr(active_fingerprinter, "device", device))
    metadata_payload = {
        "representation": "chemeleon_fp",
        "pooling": "mean",
        "feature_count": FEATURE_COUNT,
        "row_count": int(len(features)),
        "batch_size": int(batch_size),
        "device": resolved_device,
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "versions": {
            "chemprop": _package_version("chemprop"),
            "torch": _package_version("torch"),
            "rdkit": _package_version("rdkit"),
        },
    }
    metadata_output_file.write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )
    logging.info(
        "Saved %d CheMeleon fingerprints with %d features to %s.",
        len(features),
        FEATURE_COUNT,
        output_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frozen 2048-dimensional CheMeleon fingerprints from SMILES."
    )
    parser.add_argument("input_file", help="Curated CSV containing canonical_smiles.")
    parser.add_argument("output_file", help="Output CSV containing fingerprint features.")
    parser.add_argument("--labeled-output-file", required=True)
    parser.add_argument("--metadata-output-file", required=True)
    parser.add_argument("--checkpoint", required=True, help="Existing chemeleon_mp.pt checkpoint.")
    parser.add_argument("--property-columns", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    property_columns = [
        column.strip()
        for column in str(args.property_columns).split(",")
        if column.strip()
    ]
    generate_fingerprint_artifacts(
        input_path=args.input_file,
        output_path=args.output_file,
        labeled_output_path=args.labeled_output_file,
        metadata_output_path=args.metadata_output_file,
        checkpoint_path=args.checkpoint,
        property_columns=property_columns,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
