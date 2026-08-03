from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import main


def test_chemeleon_fp_node_wires_checkpoint_and_outputs(monkeypatch, tmp_path: Path) -> None:
    paths = main.build_paths(str(tmp_path / "data"))
    Path(paths["split_dir"]).mkdir(parents=True, exist_ok=True)
    curated_path = Path(paths["curated"])
    pd.DataFrame(
        {
            "__row_index": [10],
            "canonical_smiles": ["CCO"],
            "Activity": [1],
        }
    ).to_csv(curated_path, index=False)
    captured: list[str] = []

    def _fake_run_subprocess(command: list[str], **_kwargs):
        captured.extend(command)
        pd.DataFrame(
            {"chemeleon_fp_0000": [0.5], "__row_index": [10]}
        ).to_csv(paths["chemeleon_fingerprints"], index=False)
        pd.DataFrame(
            {
                "chemeleon_fp_0000": [0.5],
                "__row_index": [10],
                "Activity": [1],
            }
        ).to_csv(paths["chemeleon_labeled"], index=False)
        Path(paths["chemeleon_meta"]).write_text(
            json.dumps({"feature_count": 2048}), encoding="utf-8"
        )

    monkeypatch.setattr(main, "_run_subprocess", _fake_run_subprocess)
    context = {
        "paths": paths,
        "curated_path": str(curated_path),
        "target_column": "Activity",
        "featurize_config": {
            "checkpoint": "models/chemeleon_mp.pt",
            "batch_size": 16,
            "device": "cpu",
        },
    }

    main.run_node_featurize_chemeleon_fp(context)

    assert "--checkpoint" in captured
    assert captured[captured.index("--checkpoint") + 1] == "models/chemeleon_mp.pt"
    assert captured[captured.index("--batch-size") + 1] == "16"
    assert captured[captured.index("--device") + 1] == "cpu"
    assert context["feature_matrix"] == paths["chemeleon_labeled"]
    assert context["labels_matrix"] == paths["chemeleon_labeled"]
    assert context["feature_method"] == "chemeleon_fp"
