from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from GenDescriptors.CheMeleon_fingerprints import generate_fingerprint_artifacts


class _FakeFingerprinter:
    def __call__(self, molecules: list[str]) -> np.ndarray:
        rows = []
        for index, _molecule in enumerate(molecules, start=1):
            rows.append(np.full(2048, float(index), dtype=np.float32))
        return np.vstack(rows)


def test_generate_chemeleon_fingerprints_preserves_rows_and_labels(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_path = tmp_path / "curated.csv"
    features_path = tmp_path / "features.csv"
    labeled_path = tmp_path / "labeled.csv"
    metadata_path = tmp_path / "metadata.json"
    checkpoint_path = tmp_path / "chemeleon_mp.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    pd.DataFrame(
        {
            "__row_index": [10, 12],
            "canonical_smiles": ["CCO", "c1ccccc1"],
            "Activity": [1, 0],
        }
    ).to_csv(input_path, index=False)

    generate_fingerprint_artifacts(
        input_path=input_path,
        output_path=features_path,
        labeled_output_path=labeled_path,
        metadata_output_path=metadata_path,
        checkpoint_path=checkpoint_path,
        property_columns=["Activity"],
        batch_size=1,
        device="cpu",
        fingerprinter=_FakeFingerprinter(),
    )

    features = pd.read_csv(features_path)
    labeled = pd.read_csv(labeled_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert features.shape == (2, 2049)
    assert features["__row_index"].tolist() == [10, 12]
    assert "chemeleon_fp_0000" in features.columns
    assert "chemeleon_fp_2047" in features.columns
    assert labeled["Activity"].tolist() == [1, 0]
    assert metadata["feature_count"] == 2048
    assert metadata["row_count"] == 2
    assert metadata["checkpoint_sha256"]


def test_generate_chemeleon_fingerprints_rejects_invalid_smiles(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_path = tmp_path / "curated.csv"
    checkpoint_path = tmp_path / "chemeleon_mp.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    pd.DataFrame(
        {
            "__row_index": [10],
            "canonical_smiles": ["not-a-smiles"],
            "Activity": [1],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="row IDs.*10"):
        generate_fingerprint_artifacts(
            input_path=input_path,
            output_path=tmp_path / "features.csv",
            labeled_output_path=tmp_path / "labeled.csv",
            metadata_output_path=tmp_path / "metadata.json",
            checkpoint_path=checkpoint_path,
            property_columns=["Activity"],
            batch_size=32,
            device="cpu",
            fingerprinter=_FakeFingerprinter(),
        )
