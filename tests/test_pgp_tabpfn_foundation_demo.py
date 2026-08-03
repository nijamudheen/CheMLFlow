from __future__ import annotations

from pathlib import Path

import yaml

from utilities.doe import generate_doe, load_doe_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pgp_tabpfn_foundation_demo_generates_two_models_by_three_folds(
    tmp_path: Path,
) -> None:
    spec_path = REPO_ROOT / "doe" / "pgp_tabpfn_foundation_demo.yaml"
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(tmp_path / "generated")
    isolated_spec = tmp_path / "pgp_tabpfn_foundation_demo.yaml"
    isolated_spec.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = generate_doe(load_doe_spec(isolated_spec), doe_path=str(isolated_spec))
    summary = result["summary"]

    assert summary["valid_cases"] == 6
    assert summary["skipped_cases"] == 0
    assert summary["valid_parent_cases"] == 2
    assert summary["skipped_parent_cases"] == 0
    assert {
        row["factors"]["pipeline.feature_input"] for row in result["valid_cases"]
    } == {"featurize.rdkit", "featurize.chemeleon_fp"}

    for row in result["valid_cases"]:
        config = yaml.safe_load(Path(row["config_path"]).read_text(encoding="utf-8"))
        assert config["train"]["model"]["type"] == "tabpfn"
        assert config["preprocess"] == {
            "scaler": "standard",
            "variance_threshold": None,
            "corr_threshold": 1.0,
            "clip": {"min": -6, "max": 6},
        }
        assert "preprocess.features" in config["pipeline"]["nodes"]
        if row["factors"]["pipeline.feature_input"] == "featurize.chemeleon_fp":
            assert config["featurize"]["checkpoint"] == "models/chemeleon_mp.pt"
