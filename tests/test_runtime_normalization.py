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


def test_run_node_curate_passes_drop_controls(monkeypatch, tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    preprocessed = tmp_path / "preprocessed.csv"
    curated = tmp_path / "curated.csv"
    curated_smiles = tmp_path / "curated_smiles.csv"
    pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1], "aux": [2]}).to_csv(raw, index=False)

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
        "pipeline_type": "flash",
        "get_data_config": {},
        "curate_config": {
            "properties": ["target", "aux"],
            "drop_missing_smiles": False,
            "drop_invalid_smiles": False,
            "drop_missing_target": True,
            "required_non_null_columns": ["target", "aux"],
        },
        "target_column": "target",
        "task_type": "regression",
        "active_threshold": 1000,
        "inactive_threshold": 10000,
        "keep_all_columns": False,
    }

    main.run_node_curate(context)

    cmd = captured["cmd"]
    assert "--target_column" in cmd
    target_idx = cmd.index("--target_column")
    assert cmd[target_idx + 1] == "target"
    assert "--no_drop_missing_smiles" in cmd
    assert "--no_drop_invalid_smiles" in cmd
    assert "--drop_missing_target" in cmd
    req_idx = cmd.index("--required_non_null_columns")
    assert cmd[req_idx + 1] == "target,aux"


def test_lipinski_featurizer_is_not_a_runtime_node() -> None:
    assert "featurize.lipinski" not in main.NODE_REGISTRY
    assert "featurize.lipinski" not in main._SPLIT_MUST_FOLLOW


def test_canonicalize_pipeline_nodes_maps_legacy_alias() -> None:
    nodes = ["get_data", "use.curated_features", "train"]
    assert main._canonicalize_pipeline_nodes(nodes) == ["get_data", "featurize.none", "train"]


def test_runtime_config_normalization_stabilizes_alias_hashes() -> None:
    alias_cfg = {
        "global": {
            "pipeline_type": "flash",
            "base_dir": "data/flash",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": "runs/example",
        },
        "pipeline": {"nodes": ["get_data", "curate", "split", "use.curated_features", "train"]},
    }
    canonical_cfg = {
        "global": {
            "pipeline_type": "flash",
            "base_dir": "data/flash",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": "runs/example",
        },
        "pipeline": {"nodes": ["get_data", "curate", "split", "featurize.none", "train"]},
    }

    alias_nodes = main._canonicalize_pipeline_nodes(alias_cfg["pipeline"]["nodes"])
    canonical_nodes = main._canonicalize_pipeline_nodes(canonical_cfg["pipeline"]["nodes"])
    alias_runtime_cfg = main._normalize_runtime_config(alias_cfg, alias_nodes)
    canonical_runtime_cfg = main._normalize_runtime_config(canonical_cfg, canonical_nodes)

    assert alias_runtime_cfg["pipeline"]["nodes"] == canonical_runtime_cfg["pipeline"]["nodes"]
    assert main._stable_hash(alias_runtime_cfg) == main._stable_hash(canonical_runtime_cfg)
    assert main._stable_hash(main._hashable_config_payload(alias_runtime_cfg)) == main._stable_hash(
        main._hashable_config_payload(canonical_runtime_cfg)
    )
