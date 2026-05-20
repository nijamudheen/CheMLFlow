import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MLModels import train_models
from analysis import (
    GeneralizationRecord,
    _aggregate_all_runs_metric_rows,
    _aggregate_generalization_records,
    _infer_feature_input,
    _resolve_metrics_path,
    _resolve_split_metrics,
    _scientific_config_id,
)


def test_train_model_writes_split_metrics_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(36, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(2.5 * X["f0"] - 0.8 * X["f1"] + rng.normal(scale=0.2, size=len(X)))

    X_train = X.iloc[:24]
    y_train = y.iloc[:24]
    X_val = X.iloc[24:30]
    y_val = y.iloc[24:30]
    X_test = X.iloc[30:]
    y_test = y.iloc[30:]

    _, train_result = train_models.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="decision_tree",
        output_dir=str(tmp_path),
        cv_folds=2,
        search_iters=1,
        task_type="regression",
        model_config={"plot_split_performance": True, "n_jobs": 1},
        X_val=X_val,
        y_val=y_val,
    )

    metrics = json.loads(Path(train_result.metrics_path).read_text(encoding="utf-8"))
    split_metrics_path = metrics.get("split_metrics_path")
    split_plot_path = metrics.get("split_metrics_plot_path")

    assert split_metrics_path is not None
    assert split_plot_path is not None
    assert Path(split_metrics_path).exists()
    assert Path(split_plot_path).exists()
    parity_train_path = metrics.get("parity_plot_train_path")
    parity_val_path = metrics.get("parity_plot_val_path")
    parity_test_path = metrics.get("parity_plot_test_path")
    parity_all_splits_path = metrics.get("parity_plot_all_splits_path")
    assert parity_train_path is not None
    assert parity_val_path is not None
    assert parity_test_path is not None
    assert parity_all_splits_path is not None
    assert Path(parity_train_path).exists()
    assert Path(parity_val_path).exists()
    assert Path(parity_test_path).exists()
    assert Path(parity_all_splits_path).exists()

    split_metrics = json.loads(Path(split_metrics_path).read_text(encoding="utf-8"))
    assert set(split_metrics.keys()) == {"train", "val", "test"}
    assert {"r2", "mae"}.issubset(split_metrics["train"].keys())


def test_aggregate_all_runs_metric_rows_groups_execution_slices():
    rows = [
        {
            "case_name": "case_0001",
            "parent_case_id": "parent_0001",
            "scientific_config_id": "cfg-a",
            "job_id": "101",
            "state": "COMPLETED",
            "failure_reason": "",
            "profile": "reg_local_csv",
            "model_type": "random_forest",
            "feature_input": "featurize.morgan",
            "split_mode": "cv",
            "split_strategy": "random",
            "execution_label": "rep0_fold0",
            "config_path": "/tmp/case_0001.yaml",
            "run_dir": "/tmp/run1",
            "metrics_path": "/tmp/m1.json",
            "test_r2": 0.2,
            "test_mae": 1.0,
        },
        {
            "case_name": "case_0002",
            "parent_case_id": "parent_0001",
            "scientific_config_id": "cfg-a",
            "job_id": "102",
            "state": "COMPLETED",
            "failure_reason": "",
            "profile": "reg_local_csv",
            "model_type": "random_forest",
            "feature_input": "featurize.morgan",
            "split_mode": "cv",
            "split_strategy": "random",
            "execution_label": "rep0_fold1",
            "config_path": "/tmp/case_0002.yaml",
            "run_dir": "/tmp/run2",
            "metrics_path": "/tmp/m2.json",
            "test_r2": 0.4,
            "test_mae": 2.0,
        },
        {
            "case_name": "case_0003",
            "parent_case_id": "parent_0002",
            "scientific_config_id": "cfg-b",
            "job_id": "103",
            "state": "FAILED",
            "failure_reason": "timeout",
            "profile": "reg_local_csv",
            "model_type": "svm",
            "feature_input": "featurize.rdkit",
            "split_mode": "cv",
            "split_strategy": "scaffold",
            "execution_label": "rep0_fold0",
            "config_path": "/tmp/case_0003.yaml",
            "run_dir": "/tmp/run3",
            "metrics_path": "",
        },
    ]

    aggregated = _aggregate_all_runs_metric_rows(rows)

    assert len(aggregated) == 2
    first = aggregated[0]
    second = aggregated[1]

    assert first["parent_case_id"] == "parent_0001"
    assert first["scientific_config_id"] == "cfg-a"
    assert first["state"] == "COMPLETED"
    assert first["slice_count"] == 2
    assert first["completed_slices"] == 2
    assert first["failed_slices"] == 0
    assert abs(float(first["test_r2"]) - 0.3) < 1e-12
    assert abs(float(first["test_r2_std"]) - 0.1) < 1e-12
    assert first["test_mae"] == 1.5

    assert second["parent_case_id"] == "parent_0002"
    assert second["scientific_config_id"] == "cfg-b"
    assert second["state"] == "FAILED"
    assert second["completed_slices"] == 0
    assert second["failure_reason"] == "timeout=1"
    assert second["test_r2"] == ""


def test_aggregate_generalization_records_marks_partial_configs():
    records = [
        GeneralizationRecord(
            case_name="case_0001",
            model_type="random_forest",
            split_mode="cv",
            split_strategy="random",
            feature_input="featurize.morgan",
            metric_name="r2",
            train_value=0.8,
            test_value=0.5,
            val_value=0.55,
            gap_train_minus_test=0.3,
            gap_train_minus_test_std=None,
            overfit_flag=True,
            underfit_flag=False,
            config_path="/tmp/case_0001.yaml",
            run_dir="/tmp/run1",
            parent_case_id="parent_0001",
            scientific_config_id="cfg-a",
            execution_label="rep0_fold0",
            state="COMPLETED",
            failure_reason=None,
            slice_count=1,
            completed_slices=1,
            failed_slices=0,
            config_paths=("/tmp/case_0001.yaml",),
            run_dirs=("/tmp/run1",),
        ),
        GeneralizationRecord(
            case_name="case_0002",
            model_type="random_forest",
            split_mode="cv",
            split_strategy="random",
            feature_input="featurize.morgan",
            metric_name="r2",
            train_value=0.9,
            test_value=0.7,
            val_value=0.72,
            gap_train_minus_test=0.2,
            gap_train_minus_test_std=None,
            overfit_flag=True,
            underfit_flag=False,
            config_path="/tmp/case_0002.yaml",
            run_dir="/tmp/run2",
            parent_case_id="parent_0001",
            scientific_config_id="cfg-a",
            execution_label="rep0_fold1",
            state="COMPLETED",
            failure_reason=None,
            slice_count=1,
            completed_slices=1,
            failed_slices=0,
            config_paths=("/tmp/case_0002.yaml",),
            run_dirs=("/tmp/run2",),
        ),
    ]

    aggregated = _aggregate_generalization_records(
        records=records,
        overfit_threshold=0.2,
        underfit_threshold_r2=0.2,
        underfit_threshold_auc=0.65,
        config_summary={
            "parent_0001": {
                "state": "PARTIAL",
                "failure_reason": "timeout=1",
                "slice_count": 3,
                "completed_slices": 2,
                "failed_slices": 1,
            }
        },
    )

    assert len(aggregated) == 1
    record = aggregated[0]
    assert record.parent_case_id == "parent_0001"
    assert record.scientific_config_id == "cfg-a"
    assert record.slice_count == 3
    assert record.metric_slices_used == 2
    assert record.completed_slices == 2
    assert record.failed_slices == 1
    assert record.state == "PARTIAL"
    assert record.failure_reason == "timeout=1"
    assert abs(record.train_value - 0.85) < 1e-12
    assert record.test_value == 0.6
    assert record.gap_train_minus_test == 0.25
    assert abs((record.gap_train_minus_test_std or 0.0) - 0.05) < 1e-12
    assert record.overfit_flag is True
    assert record.execution_label == "aggregated"


def test_aggregate_generalization_records_marks_missing_metric_slices_partial():
    records = [
        GeneralizationRecord(
            case_name="case_0001",
            model_type="random_forest",
            split_mode="cv",
            split_strategy="random",
            feature_input="featurize.morgan",
            metric_name="r2",
            train_value=0.8,
            test_value=0.5,
            val_value=0.55,
            gap_train_minus_test=0.3,
            gap_train_minus_test_std=None,
            overfit_flag=True,
            underfit_flag=False,
            config_path="/tmp/case_0001.yaml",
            run_dir="/tmp/run1",
            parent_case_id="parent_0001",
            scientific_config_id="cfg-a",
            execution_label="rep0_fold0",
            state="COMPLETED",
            failure_reason=None,
            slice_count=1,
            metric_slices_used=1,
            completed_slices=1,
            failed_slices=0,
            config_paths=("/tmp/case_0001.yaml",),
            run_dirs=("/tmp/run1",),
        ),
        GeneralizationRecord(
            case_name="case_0002",
            model_type="random_forest",
            split_mode="cv",
            split_strategy="random",
            feature_input="featurize.morgan",
            metric_name="r2",
            train_value=0.9,
            test_value=0.7,
            val_value=0.72,
            gap_train_minus_test=0.2,
            gap_train_minus_test_std=None,
            overfit_flag=True,
            underfit_flag=False,
            config_path="/tmp/case_0002.yaml",
            run_dir="/tmp/run2",
            parent_case_id="parent_0001",
            scientific_config_id="cfg-a",
            execution_label="rep0_fold1",
            state="COMPLETED",
            failure_reason=None,
            slice_count=1,
            metric_slices_used=1,
            completed_slices=1,
            failed_slices=0,
            config_paths=("/tmp/case_0002.yaml",),
            run_dirs=("/tmp/run2",),
        ),
    ]

    aggregated = _aggregate_generalization_records(
        records=records,
        overfit_threshold=0.2,
        underfit_threshold_r2=0.2,
        underfit_threshold_auc=0.65,
        config_summary={
            "parent_0001": {
                "state": "COMPLETED",
                "failure_reason": "",
                "slice_count": 3,
                "completed_slices": 3,
                "failed_slices": 0,
            }
        },
    )

    assert len(aggregated) == 1
    record = aggregated[0]
    assert record.state == "PARTIAL"
    assert record.slice_count == 3
    assert record.metric_slices_used == 2
    assert record.completed_slices == 2
    assert record.failed_slices == 1
    assert record.failure_reason == "missing_generalization_metrics=1"


def test_analysis_prefers_explicit_feature_node_over_declared_pipeline_feature_input() -> None:
    config = {
        "pipeline": {
            "feature_input": "smiles_native",
            "nodes": ["get_data", "curate", "featurize.rdkit", "split", "train"],
        }
    }

    assert _infer_feature_input(config) == "featurize.rdkit"


def test_analysis_recognizes_legacy_use_curated_features_alias() -> None:
    config = {
        "pipeline": {
            "nodes": ["get_data", "curate", "use.curated_features", "split", "train"],
        }
    }

    assert _infer_feature_input(config) == "featurize.none"


def test_analysis_normalizes_configured_use_curated_features_alias() -> None:
    config = {
        "pipeline": {
            "feature_input": "use.curated_features",
            "nodes": ["get_data", "curate", "split", "train"],
        }
    }

    assert _infer_feature_input(config) == "featurize.none"


def test_analysis_infers_smiles_native_for_historical_chemprop_runs() -> None:
    config = {
        "pipeline": {
            "nodes": ["get_data", "curate", "split", "train"],
        },
        "train": {
            "model": {"type": "chemprop"},
        },
    }

    assert _infer_feature_input(config) == "smiles_native"


def test_analysis_scientific_config_id_ignores_artifact_retention() -> None:
    base_config = {
        "global": {
            "pipeline_type": "pgp",
            "base_dir": "/tmp/data",
            "run_dir": "/tmp/run",
            "artifact_retention": "full",
            "runs": {"enabled": True, "id": "case_001"},
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "split", "train"],
            "feature_input": "featurize.morgan",
        },
        "train": {"model": {"type": "random_forest"}},
        "split": {"mode": "cv", "cv": {"repeat_index": 0, "fold_index": 1}},
    }
    audit_config = json.loads(json.dumps(base_config))
    audit_config["global"]["artifact_retention"] = "audit_light"

    assert _scientific_config_id(base_config) == _scientific_config_id(audit_config)


def test_analysis_uses_chemprop_artifacts_for_chemeleon(tmp_path: Path) -> None:
    run_dir = tmp_path / "chemeleon_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "chemprop_metrics.json"
    split_metrics_path = run_dir / "chemprop_split_metrics.json"
    metrics_path.write_text(json.dumps({"split_metrics_path": str(split_metrics_path)}), encoding="utf-8")
    split_metrics_path.write_text(json.dumps({"train": {"r2": 0.9}, "test": {"r2": 0.7}}), encoding="utf-8")

    assert _resolve_metrics_path(run_dir, "chemeleon") == metrics_path
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    resolved = _resolve_split_metrics("chemeleon", run_dir, payload)
    assert resolved == {"train": {"r2": 0.9}, "test": {"r2": 0.7}}


def test_classification_split_plot_artifacts(tmp_path):
    split_outputs = {
        "train": {
            "y_true": np.array([0, 0, 1, 1, 1, 0]),
            "y_proba": np.array([0.05, 0.2, 0.7, 0.8, 0.9, 0.3]),
            "y_pred": np.array([0, 0, 1, 1, 1, 0]),
        },
        "val": {
            "y_true": np.array([0, 1, 0, 1]),
            "y_proba": np.array([0.15, 0.75, 0.35, 0.65]),
            "y_pred": np.array([0, 1, 0, 1]),
        },
        "test": {
            "y_true": np.array([0, 1, 0, 1, 1]),
            "y_proba": np.array([0.12, 0.61, 0.44, 0.79, 0.83]),
            "y_pred": np.array([0, 1, 0, 1, 1]),
        },
    }

    paths = train_models._save_classification_split_plots(
        output_dir=str(tmp_path),
        model_type="catboost_classifier",
        split_outputs=split_outputs,
    )

    expected_keys = {
        "pr_curve_train_path",
        "pr_curve_val_path",
        "pr_curve_test_path",
        "pr_curve_all_splits_path",
        "confusion_matrix_train_path",
        "confusion_matrix_val_path",
        "confusion_matrix_test_path",
        "confusion_matrix_all_splits_path",
    }
    assert expected_keys.issubset(paths.keys())
    for key in expected_keys:
        assert Path(paths[key]).exists()


def test_train_model_sanitizes_xgboost_feature_names(tmp_path):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.normal(size=(60, 4)),
        columns=["A[1]", "B<2>", "C]3[", "regular"],
    )
    y = pd.Series(1.8 * X["A[1]"] - 0.6 * X["B<2>"] + rng.normal(scale=0.1, size=len(X)))

    X_train = X.iloc[:40]
    y_train = y.iloc[:40]
    X_test = X.iloc[40:]
    y_test = y.iloc[40:]

    _, train_result = train_models.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="xgboost",
        output_dir=str(tmp_path),
        cv_folds=2,
        search_iters=1,
        task_type="regression",
        model_config={
            "n_jobs": 1,
            "params": {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        },
    )

    metrics = json.loads(Path(train_result.metrics_path).read_text(encoding="utf-8"))
    map_path = metrics.get("feature_name_map_path")
    assert map_path is not None
    assert Path(map_path).exists()

    payload = json.loads(Path(map_path).read_text(encoding="utf-8"))
    sanitized_cols = payload.get("sanitized_columns")
    assert isinstance(sanitized_cols, list)
    assert len(sanitized_cols) == X_train.shape[1]
    assert all("[" not in c and "]" not in c and "<" not in c and ">" not in c for c in sanitized_cols)


@pytest.mark.parametrize(
    "model_type",
    ["random_forest", "decision_tree", "xgboost", "svm", "ensemble"],
)
def test_train_model_supports_classification_for_tabular_models(tmp_path, model_type):
    rng = np.random.default_rng(11)
    n_rows = 80
    labels = np.array([i % 2 for i in range(n_rows)], dtype=int)
    X = pd.DataFrame(
        {
            "f0": labels + rng.normal(scale=0.1, size=n_rows),
            "f1": (1 - labels) + rng.normal(scale=0.1, size=n_rows),
            "f2": rng.normal(size=n_rows),
            "f3": rng.normal(size=n_rows),
        }
    )
    y = pd.Series(labels)

    X_train = X.iloc[:50]
    y_train = y.iloc[:50]
    X_val = X.iloc[50:65]
    y_val = y.iloc[50:65]
    X_test = X.iloc[65:]
    y_test = y.iloc[65:]

    params_by_model = {
        "random_forest": {"n_estimators": 30, "max_depth": 4},
        "decision_tree": {"max_depth": 4},
        "xgboost": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "svm": {"kernel": "linear", "C": 1.0, "probability": True},
        "ensemble": {
            "voting": "soft",
            "rf_params": {"n_estimators": 30, "max_depth": 4},
            "xgb_params": {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        },
    }

    _, train_result = train_models.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        output_dir=str(tmp_path / model_type),
        cv_folds=2,
        search_iters=1,
        task_type="classification",
        model_config={
            "n_jobs": 1,
            "plot_split_performance": True,
            "params": params_by_model[model_type],
        },
        X_val=X_val,
        y_val=y_val,
    )

    metrics = json.loads(Path(train_result.metrics_path).read_text(encoding="utf-8"))
    assert {"auc", "auprc", "accuracy", "f1"}.issubset(metrics.keys())
    assert metrics.get("split_metrics_path")
    assert metrics.get("split_metrics_plot_path")

    preds_path = Path(tmp_path / model_type / f"{model_type}_predictions.csv")
    assert preds_path.exists()
    preds = pd.read_csv(preds_path)
    assert {"y_true", "y_score", "y_proba", "y_pred"}.issubset(preds.columns)
    assert preds["y_proba"].between(0, 1).all()


@pytest.mark.parametrize(
    ("model_type", "search_type", "estimator_type"),
    [
        ("random_forest", "RandomizedSearchCV", "RandomForestClassifier"),
        ("decision_tree", "RandomizedSearchCV", "DecisionTreeClassifier"),
        ("xgboost", "RandomizedSearchCV", "XGBClassifier"),
        ("svm", "GridSearchCV", "SVC"),
    ],
)
def test_initialize_model_builds_classification_train_cv_searchers(
    model_type, search_type, estimator_type
):
    model = train_models._initialize_model(
        model_type=model_type,
        random_state=42,
        cv_folds=2,
        search_iters=1,
        n_jobs=1,
        tuning_method="train_cv",
        model_params={},
        task_type="classification",
    )

    assert model.__class__.__name__ == search_type
    assert getattr(model, "estimator").__class__.__name__ == estimator_type


def test_predict_classification_outputs_uses_decision_function_when_no_proba():
    class _DecisionEstimator:
        def decision_function(self, X):
            return np.array([-2.0, 0.0, 2.0], dtype=float)

        def predict(self, X):
            return np.array([0, 0, 1], dtype=int)

    X = pd.DataFrame({"f0": [0.1, 0.2, 0.3]})
    y_proba, y_pred, y_score = train_models._predict_classification_outputs(
        estimator=_DecisionEstimator(),
        model_type="svm",
        X=X,
    )

    assert np.allclose(y_score, np.array([-2.0, 0.0, 2.0]))
    assert np.all((y_proba >= 0.0) & (y_proba <= 1.0))
    assert np.array_equal(y_pred, np.array([0, 0, 1], dtype=int))


def test_predict_classification_outputs_falls_back_to_hard_predictions():
    class _PredictOnlyEstimator:
        def predict(self, X):
            return np.array([0, 1, 1, 0], dtype=int)

    X = pd.DataFrame({"f0": [1, 2, 3, 4]})
    y_proba, y_pred, y_score = train_models._predict_classification_outputs(
        estimator=_PredictOnlyEstimator(),
        model_type="unknown_model",
        X=X,
    )

    assert np.array_equal(y_pred, np.array([0, 1, 1, 0], dtype=int))
    assert np.array_equal(y_proba, y_pred.astype(float))
    assert np.array_equal(y_score, y_pred.astype(float))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_train_model_dl_classification_raises_clear_error_for_non_finite_scores(
    monkeypatch,
    tmp_path,
    bad_value,
):
    class _FakeDlModel:
        def __init__(self, params: dict[str, object]) -> None:
            self.params = dict(params)

        def state_dict(self) -> dict[str, object]:
            return {}

    cfg = train_models.DLSearchConfig(
        model_class=_FakeDlModel,
        search_space={},
        default_params={
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "hidden_dim": 16,
        },
    )

    monkeypatch.setattr(train_models, "_initialize_model", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(train_models, "_seed_dl_runtime", lambda _seed: None)
    monkeypatch.setattr(
        train_models,
        "_train_dl",
        lambda *args, **kwargs: {"model": args[0], "best_params": {"epochs": 5}},
    )
    monkeypatch.setattr(
        train_models,
        "_predict_dl",
        lambda model, X: np.array([0.0, bad_value, 1.0, 1.5], dtype=float),
    )

    X = pd.DataFrame(np.ones((16, 4), dtype=float), columns=["f0", "f1", "f2", "f3"])
    y = pd.Series([0, 1] * 8, dtype=int)
    X_train, X_val, X_test = X.iloc[:8], X.iloc[8:12], X.iloc[12:]
    y_train, y_val, y_test = y.iloc[:8], y.iloc[8:12], y.iloc[12:]

    with pytest.raises(
        ValueError,
        match="dl_simple classification raw scores: non-finite values in classification scores",
    ):
        train_models.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type="dl_simple",
            output_dir=str(tmp_path),
            task_type="classification",
            model_config={"params": {"epochs": 5, "batch_size": 8, "learning_rate": 1e-3}},
            X_val=X_val,
            y_val=y_val,
        )
