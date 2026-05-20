from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MLModels.training import cli


def _make_regression_frame(n_rows: int = 40) -> pd.DataFrame:
    x1 = np.linspace(0.0, 1.0, num=n_rows)
    x2 = np.linspace(1.0, 2.0, num=n_rows)
    y = 2.0 * x1 - 0.5 * x2
    return pd.DataFrame(
        {
            "f1": x1,
            "f2": x2,
            "target": y,
        }
    )


def test_cli_help_lists_subcommands(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert int(excinfo.value.code) == 0
    out = capsys.readouterr().out
    assert "train" in out
    assert "predict" in out
    assert "explain" in out


def test_cli_train_and_predict_regression_smoke(tmp_path) -> None:
    df = _make_regression_frame()
    data_path = tmp_path / "train.csv"
    preds_path = tmp_path / "predictions.csv"
    out_dir = tmp_path / "out"
    df.to_csv(data_path, index=False)

    rc_train = cli.main(
        [
            "train",
            "--data-path",
            str(data_path),
            "--target-column",
            "target",
            "--model-type",
            "random_forest",
            "--task-type",
            "regression",
            "--output-dir",
            str(out_dir),
            "--test-size",
            "0.2",
            "--val-size",
            "0.0",
        ]
    )
    assert rc_train == 0
    assert (out_dir / "random_forest_best_model.pkl").exists()
    assert (out_dir / "random_forest_metrics.json").exists()
    assert (out_dir / "random_forest_best_params.pkl").exists()

    rc_predict = cli.main(
        [
            "predict",
            "--test-path",
            str(data_path),
            "--model-path",
            str(out_dir / "random_forest_best_model.pkl"),
            "--model-type",
            "random_forest",
            "--task-type",
            "regression",
            "--target-column",
            "target",
            "--preds-path",
            str(preds_path),
        ]
    )
    assert rc_predict == 0
    pred_df = pd.read_csv(preds_path)
    assert "pred_0" in pred_df.columns
    assert len(pred_df) == len(df)


def test_cli_explain_calls_api(monkeypatch, tmp_path) -> None:
    train_df = _make_regression_frame(n_rows=24)
    test_df = _make_regression_frame(n_rows=12)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    called: dict[str, object] = {}

    def _fake_load(*, model_path: str, model_type: str, input_dim: int | None = None):
        called["load"] = {
            "model_path": model_path,
            "model_type": model_type,
            "input_dim": input_dim,
        }
        return object()

    def _fake_run_explainability(
        *,
        estimator: object,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: str,
        output_dir: str,
        background_samples: int = 100,
        task_type: str = "regression",
    ) -> None:
        called["run_explainability"] = {
            "estimator": estimator,
            "X_train_shape": tuple(X_train.shape),
            "X_test_shape": tuple(X_test.shape),
            "y_test_len": int(len(y_test)),
            "model_type": model_type,
            "output_dir": output_dir,
            "background_samples": int(background_samples),
            "task_type": task_type,
        }

    monkeypatch.setattr(cli.training_api, "load", _fake_load)
    monkeypatch.setattr(cli.training_api, "run_explainability", _fake_run_explainability)

    rc = cli.main(
        [
            "explain",
            "--train-path",
            str(train_path),
            "--test-path",
            str(test_path),
            "--target-column",
            "target",
            "--model-path",
            str(tmp_path / "model.pkl"),
            "--model-type",
            "random_forest",
            "--task-type",
            "regression",
            "--output-dir",
            str(tmp_path / "explain"),
            "--background-samples",
            "10",
        ]
    )
    assert rc == 0
    assert called["load"] == {
        "model_path": str(tmp_path / "model.pkl"),
        "model_type": "random_forest",
        "input_dim": 2,
    }
    assert called["run_explainability"] == {
        "estimator": called["run_explainability"]["estimator"],
        "X_train_shape": (24, 2),
        "X_test_shape": (12, 2),
        "y_test_len": 12,
        "model_type": "random_forest",
        "output_dir": str(tmp_path / "explain"),
        "background_samples": 10,
        "task_type": "regression",
    }
