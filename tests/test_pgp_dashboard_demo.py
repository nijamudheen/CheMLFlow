from __future__ import annotations

from pathlib import Path

import yaml

from utilities.doe import generate_doe, load_doe_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pgp_dashboard_demo_generates_three_models_by_three_folds(
    tmp_path: Path,
) -> None:
    spec_path = REPO_ROOT / "doe" / "pgp_dashboard_demo.yaml"
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(tmp_path / "generated")
    isolated_spec = tmp_path / "pgp_dashboard_demo.yaml"
    isolated_spec.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = generate_doe(load_doe_spec(isolated_spec), doe_path=str(isolated_spec))
    summary = result["summary"]

    assert summary["valid_cases"] == 9
    assert summary["skipped_cases"] == 9
    assert summary["valid_parent_cases"] == 3
    assert summary["skipped_parent_cases"] == 3
    assert {row["factors"]["train.model.type"] for row in result["valid_cases"]} == {
        "random_forest",
        "dl_simple",
        "chemprop",
    }
    assert raw["dataset"]["sample"] == {
        "fraction": 0.25,
        "seed": 42,
        "strategy": "stratified",
    }
