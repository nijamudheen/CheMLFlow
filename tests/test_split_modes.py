import json
from pathlib import Path

import pandas as pd
import pytest

from main import (
    _resolve_feature_inputs,
    _resolve_split_partitions,
    build_paths,
    run_node_explain,
    run_node_preprocess_features,
    run_node_split,
)


def _make_curated_df(n_rows: int = 60) -> pd.DataFrame:
    base_smiles = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "CCCl",
        "CCBr",
        "CC(=O)O",
        "c1ccccc1",
    ]
    smiles = [base_smiles[i % len(base_smiles)] for i in range(n_rows)]
    labels = [i % 2 for i in range(n_rows)]
    return pd.DataFrame({"canonical_smiles": smiles, "label": labels})


def _build_context(
    tmp_path: Path,
    *,
    split_config: dict,
    curated_df: pd.DataFrame,
    run_name: str,
    base_name: str = "data",
    global_random_state: int = 42,
) -> dict:
    base_dir = tmp_path / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = build_paths(str(base_dir))
    Path(paths["split_dir"]).mkdir(parents=True, exist_ok=True)
    curated_path = Path(paths["curated"])
    curated_df.to_csv(curated_path, index=False)
    return {
        "base_dir": str(base_dir),
        "run_dir": str(run_dir),
        "paths": paths,
        "split_config": split_config,
        "global_random_state": int(global_random_state),
        "target_column": "label",
        "source": {},
    }


def test_cv_mode_fold_run_is_reproducible_and_writes_meta(tmp_path: Path) -> None:
    curated_df = _make_curated_df(60)
    split_cfg = {
        "mode": "cv",
        "strategy": "random",
        "stratify": True,
        "stratify_column": "label",
        "cv": {"n_splits": 5, "repeats": 2, "fold_index": 3, "repeat_index": 1, "random_state": 123},
        "val_from_train": {"val_size": 0.1, "stratify": True, "random_state": 456},
        "require_disjoint": True,
    }

    context_a = _build_context(tmp_path, split_config=split_cfg, curated_df=curated_df, run_name="run_a")
    run_node_split(context_a)
    context_b = _build_context(tmp_path, split_config=split_cfg, curated_df=curated_df, run_name="run_b")
    run_node_split(context_b)

    assert context_a["split_indices"] == context_b["split_indices"]
    assert context_a["split_meta"]["plan_id"] == context_b["split_meta"]["plan_id"]
    assert context_a["split_meta"]["mode"] == "cv"
    assert context_a["split_meta"]["stage"] == "cv"
    assert context_a["split_meta"]["fold_index"] == 3
    assert context_a["split_meta"]["repeat_index"] == 1

    split_meta_path = Path(context_a["run_dir"]) / "split_meta.json"
    split_indices_path = Path(context_a["run_dir"]) / "split_indices.json"
    assert split_meta_path.exists()
    assert split_indices_path.exists()
    persisted_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
    assert persisted_meta["plan_id"] == context_a["split_meta"]["plan_id"]


def test_nested_holdout_cv_inner_never_touches_outer_test(tmp_path: Path) -> None:
    curated_df = _make_curated_df(80)
    inner_cfg = {
        "mode": "nested_holdout_cv",
        "stage": "inner",
        "strategy": "random",
        "stratify": True,
        "stratify_column": "label",
        "outer": {"test_size": 0.2, "random_state": 111},
        "inner": {"n_splits": 5, "repeats": 2, "fold_index": 2, "repeat_index": 1, "random_state": 222},
        "val_from_train": {"val_size": 0.1, "stratify": True, "random_state": 333},
        "require_disjoint": True,
    }
    outer_cfg = {
        "mode": "nested_holdout_cv",
        "stage": "outer",
        "strategy": "random",
        "stratify": True,
        "stratify_column": "label",
        "outer": {"test_size": 0.2, "random_state": 111},
        "val_from_train": {"val_size": 0.1, "stratify": True, "random_state": 444},
        "require_disjoint": True,
    }

    inner_ctx = _build_context(
        tmp_path,
        split_config=inner_cfg,
        curated_df=curated_df,
        run_name="run_nested_inner",
        base_name="shared_nested_data",
    )
    run_node_split(inner_ctx)
    outer_ctx = _build_context(
        tmp_path,
        split_config=outer_cfg,
        curated_df=curated_df,
        run_name="run_nested_outer",
        base_name="shared_nested_data",
    )
    run_node_split(outer_ctx)

    outer_test = set(outer_ctx["split_indices"]["test"])
    inner_used = set(inner_ctx["split_indices"]["train"]) | set(inner_ctx["split_indices"]["val"]) | set(
        inner_ctx["split_indices"]["test"]
    )
    assert inner_used.isdisjoint(outer_test)
    assert set(outer_ctx["split_indices"]["train"]).isdisjoint(outer_test)
    assert set(outer_ctx["split_indices"]["val"]).isdisjoint(outer_test)
    assert inner_ctx["split_meta"]["stage"] == "inner"
    assert outer_ctx["split_meta"]["stage"] == "outer"


def test_resolve_split_partitions_enforces_strict_test_coverage() -> None:
    context = {
        "split_indices": {"train": [0, 1, 2], "val": [3], "test": [4, 5]},
        "split_config": {"require_full_test_coverage": True},
    }
    with pytest.raises(ValueError, match="require_full_test_coverage"):
        _resolve_split_partitions(context, pd.Index([0, 1, 2, 3, 4]))


def test_resolve_split_partitions_allows_relaxed_coverage() -> None:
    context = {
        "split_indices": {"train": [0, 1, 2], "val": [3], "test": [4, 5]},
        "split_config": {},
    }
    train_idx, val_idx, test_idx = _resolve_split_partitions(context, pd.Index([0, 1, 2, 3, 4]))
    assert train_idx == [0, 1, 2]
    assert val_idx == [3]
    assert test_idx == [4]


def test_fold_selection_is_model_agnostic(tmp_path: Path) -> None:
    curated_df = _make_curated_df(50)
    split_cfg = {
        "mode": "cv",
        "strategy": "random",
        "stratify": True,
        "stratify_column": "label",
        "cv": {"n_splits": 5, "repeats": 1, "fold_index": 0, "repeat_index": 0, "random_state": 77},
        "val_from_train": {"val_size": 0.1, "stratify": True, "random_state": 88},
    }

    rf_ctx = _build_context(tmp_path, split_config=split_cfg, curated_df=curated_df, run_name="run_rf")
    rf_ctx["model_type"] = "random_forest"
    run_node_split(rf_ctx)

    dl_ctx = _build_context(tmp_path, split_config=split_cfg, curated_df=curated_df, run_name="run_dl")
    dl_ctx["model_type"] = "chemprop"
    run_node_split(dl_ctx)

    assert rf_ctx["split_indices"] == dl_ctx["split_indices"]
    assert rf_ctx["split_meta"]["plan_id"] == dl_ctx["split_meta"]["plan_id"]
    assert rf_ctx["split_meta"]["fold_index"] == dl_ctx["split_meta"]["fold_index"]
    assert rf_ctx["split_meta"]["repeat_index"] == dl_ctx["split_meta"]["repeat_index"]


def test_holdout_split_uses_global_seed_when_split_seed_missing(tmp_path: Path) -> None:
    curated_df = _make_curated_df(60)
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": True,
        "stratify_column": "label",
    }

    ctx_a = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_global_seed_a",
        base_name="shared_global_seed_data",
        global_random_state=777,
    )
    run_node_split(ctx_a)

    ctx_b = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_global_seed_b",
        base_name="shared_global_seed_data",
        global_random_state=777,
    )
    run_node_split(ctx_b)

    assert ctx_a["split_indices"] == ctx_b["split_indices"]
    assert int(ctx_a["split_meta"]["random_state"]) == 777
    assert int(ctx_b["split_meta"]["random_state"]) == 777


def test_split_skips_stale_default_feature_files_without_explicit_feature_node(tmp_path: Path) -> None:
    curated_df = _make_curated_df(20)
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": False,
    }
    ctx = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_no_feature_input",
        base_name="shared_no_feature_input_data",
    )
    ctx["pipeline_nodes"] = ["get_data", "curate", "split", "train"]

    stale_features = pd.DataFrame(
        {
            "__row_index": [0, 1, 2, 3, 4],
            "f1": [1, 2, 3, 4, 5],
        }
    )
    stale_labels = pd.DataFrame(
        {
            "__row_index": [0, 1, 2, 3, 4],
            "label": [0, 1, 0, 1, 0],
        }
    )
    pd.DataFrame(stale_features).to_csv(ctx["paths"]["rdkit_descriptors"], index=False)
    pd.DataFrame(stale_labels).to_csv(ctx["paths"]["pic50_3class"], index=False)

    run_node_split(ctx)

    assert ctx["split_meta"]["split_source"] == "curated_no_explicit_feature_input"
    assert int(ctx["split_meta"]["dataset_rows"]) == len(curated_df)


def test_split_rejects_duplicate_row_index_values(tmp_path: Path) -> None:
    curated_df = _make_curated_df(12)
    curated_df["__row_index"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11]
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": False,
    }
    ctx = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_duplicate_row_index",
        base_name="shared_duplicate_row_index_data",
    )
    with pytest.raises(ValueError, match="must be unique"):
        run_node_split(ctx)


def test_split_rejects_non_integer_row_index_values(tmp_path: Path) -> None:
    curated_df = _make_curated_df(12)
    curated_df["__row_index"] = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10.5",
        "11",
    ]
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": False,
    }
    ctx = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_non_integer_row_index",
        base_name="shared_non_integer_row_index_data",
    )
    with pytest.raises(ValueError, match="integer-like"):
        run_node_split(ctx)


def test_split_rejects_feature_cleaned_row_id_mismatch(tmp_path: Path) -> None:
    curated_df = _make_curated_df(20)
    curated_df["__row_index"] = list(range(100, 120))
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": False,
    }
    ctx = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_row_id_mismatch",
        base_name="shared_row_id_mismatch_data",
    )
    ctx["pipeline_nodes"] = ["get_data", "curate", "featurize.rdkit", "split", "train"]

    pd.DataFrame(
        {
            "__row_index": list(range(20)),
            "f1": list(range(20)),
            "f2": [v + 1 for v in range(20)],
        }
    ).to_csv(ctx["paths"]["rdkit_descriptors"], index=False)
    pd.DataFrame(
        {
            "__row_index": list(range(20)),
            "label": [i % 2 for i in range(20)],
        }
    ).to_csv(ctx["paths"]["pic50_3class"], index=False)

    with pytest.raises(ValueError, match="not present in curated row-id universe"):
        run_node_split(ctx)


def test_split_retains_duplicate_feature_rows_before_split(tmp_path: Path) -> None:
    curated_df = _make_curated_df(20)
    curated_df["__row_index"] = list(range(20))
    split_cfg = {
        "mode": "holdout",
        "strategy": "random",
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify": False,
    }
    ctx = _build_context(
        tmp_path,
        split_config=split_cfg,
        curated_df=curated_df,
        run_name="run_retained_duplicate_features",
        base_name="shared_retained_duplicate_features_data",
    )
    ctx["pipeline_nodes"] = ["get_data", "curate", "featurize.morgan", "split", "train"]

    feature_df = pd.DataFrame(
        {
            "__row_index": list(range(20)),
            "fp_0": list(range(20)),
            "fp_1": [v + 100 for v in range(20)],
            "label": curated_df["label"].tolist(),
        }
    )
    feature_df.loc[4, ["fp_0", "fp_1"]] = feature_df.loc[2, ["fp_0", "fp_1"]].to_numpy()
    feature_path = Path(ctx["paths"]["morgan_labeled"])
    feature_df.to_csv(feature_path, index=False)
    ctx["feature_matrix"] = str(feature_path)
    ctx["labels_matrix"] = str(feature_path)

    run_node_split(ctx)

    assigned = sorted(
        ctx["split_indices"]["train"] + ctx["split_indices"]["val"] + ctx["split_indices"]["test"]
    )
    assert assigned == list(range(len(curated_df)))
    assert ctx["split_meta"]["split_source"] == "feature_cleaned"
    assert int(ctx["split_meta"]["dataset_rows"]) == len(curated_df)
    assert int(ctx["split_meta"]["coverage"]["dropped_before_split"]) == 0
    assert int(ctx["split_meta"]["retained_duplicate_feature_label_rows"]) == 1


def test_resolve_feature_inputs_prefers_curated_path_for_missing_labels() -> None:
    context = {
        "feature_matrix": "features.csv",
        "labels_matrix": None,
        "curated_path": "curated_latest.csv",
        "paths": {
            "curated": "curated_default.csv",
            "rdkit_descriptors": "rdkit.csv",
            "pic50_3class": "labels.csv",
            "rdkit_labeled": "rdkit_labeled.csv",
        },
    }
    features_file, labels_file = _resolve_feature_inputs(context)
    assert features_file == "features.csv"
    assert labels_file == "curated_latest.csv"


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_run_node_explain_writes_skip_marker_for_chemprop_like(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    output_dir = tmp_path / "run_chemprop_explain"
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "run_dir": str(output_dir),
        "model_type": model_type,
        "target_column": "label",
        "paths": build_paths(str(tmp_path / "data_chemprop_explain")),
        "pipeline_nodes": ["get_data", "curate", "split", "train", "explain"],
        "pipeline_feature_input": "smiles_native",
    }

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("Explain skip should return before loading data or model.")

    monkeypatch.setattr("main.data_preprocessing.load_features_labels", _unexpected)
    monkeypatch.setattr("main.train_models.load_model", _unexpected)

    run_node_explain(context)

    payload = json.loads((output_dir / "explain_skipped.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["reason"] == "chemprop_explainability_not_supported"
    assert payload["model_type"] == model_type
    assert payload["feature_method"] == "smiles_native"


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_run_node_preprocess_is_noop_for_chemprop_like_with_none_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    base_dir = tmp_path / "data_chemprop_preprocess"
    run_dir = tmp_path / "run_chemprop_preprocess"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = build_paths(str(base_dir))
    Path(paths["split_dir"]).mkdir(parents=True, exist_ok=True)
    curated_df = _make_curated_df(12)
    curated_path = Path(paths["curated"])
    curated_df.to_csv(curated_path, index=False)

    context = {
        "base_dir": str(base_dir),
        "run_dir": str(run_dir),
        "paths": paths,
        "model_type": model_type,
        "target_column": "label",
        "pipeline_nodes": ["get_data", "curate", "split", "preprocess.features", "train"],
        "pipeline_feature_input": "smiles_native",
        "preprocess_config": {"scaler": "none"},
        "split_indices": {"train": list(range(8)), "val": [8, 9], "test": [10, 11]},
        "curated_path": str(curated_path),
    }

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("Chemprop preprocess no-op should not load tabular feature artifacts.")

    monkeypatch.setattr("main.data_preprocessing.load_features_labels", _unexpected)

    run_node_preprocess_features(context)

    assert context["preprocessed_ready"] is False
    assert not Path(paths["preprocessed_features"]).exists()
    assert not Path(paths["preprocessed_labels"]).exists()


def test_run_node_explain_writes_skip_marker_for_morgan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "run_morgan_explain"
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "run_dir": str(output_dir),
        "model_type": "random_forest",
        "target_column": "label",
        "paths": build_paths(str(tmp_path / "data_morgan_explain")),
        "pipeline_nodes": ["get_data", "curate", "featurize.morgan", "split", "train", "explain"],
    }

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("Morgan explain skip should return before loading data or model.")

    monkeypatch.setattr("main.data_preprocessing.load_features_labels", _unexpected)
    monkeypatch.setattr("main.train_models.load_model", _unexpected)

    run_node_explain(context)

    payload = json.loads((output_dir / "explain_skipped.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["reason"] == "morgan_explainability_disabled"
    assert payload["model_type"] == "random_forest"
    assert payload["feature_method"] == "morgan"
