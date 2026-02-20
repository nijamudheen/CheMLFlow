import pandas as pd
import pytest

import main


def test_exclude_feature_columns_accepts_comma_separated_string() -> None:
    context = {"train_config": {"features": {"exclude_columns": "a, b ,c"}}}
    assert main._exclude_feature_columns(context) == ["a", "b", "c"]


def test_exclude_feature_columns_rejects_invalid_type() -> None:
    context = {"train_config": {"features": {"exclude_columns": 123}}}
    with pytest.raises(ValueError, match="list or comma-separated string"):
        main._exclude_feature_columns(context)


def test_run_node_curate_accepts_properties_list(monkeypatch, tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"
    curated_smiles = tmp_path / "curated_smiles.csv"
    raw_df = pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1], "aux": [2]})
    raw_df.to_csv(raw, index=False)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(main, "validate_contract", lambda *args, **kwargs: None)

    def _fake_run_subprocess(cmd: list[str], *, cwd: str | None = None):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(main, "_run_subprocess", _fake_run_subprocess)

    context = {
        "config_path": str(tmp_path / "config.yaml"),
        "paths": {
            "raw": str(raw),
            "raw_sample": str(tmp_path / "raw_sample.csv"),
            "preprocessed": str(preprocessed),
            "curated": str(curated),
            "curated_smiles": str(curated_smiles),
        },
        "pipeline_type": "chembl",
        "get_data_config": {},
        "curate_config": {"properties": ["target", "aux"]},
        "target_column": "target",
        "task_type": "regression",
        "active_threshold": 1000,
        "inactive_threshold": 10000,
        "keep_all_columns": False,
    }

    main.run_node_curate(context)

    cmd = captured["cmd"]
    prop_idx = cmd.index("--properties")
    assert cmd[prop_idx + 1] == "target,aux"
