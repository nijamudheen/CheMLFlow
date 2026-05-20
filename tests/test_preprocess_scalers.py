import numpy as np
import pandas as pd
import pytest

from MLModels import data_preprocessing


def test_fit_preprocessor_supports_standard_scaler():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    preprocessor = data_preprocessing.fit_preprocessor(
        X,
        variance_threshold=0.0,
        corr_threshold=1.0,
        clip_range=(-1e10, 1e10),
        scaler="standard",
    )
    transformed = data_preprocessing.transform_preprocessor(X, preprocessor)

    assert preprocessor["scaler_name"] == "standard"
    assert np.allclose(transformed.mean(axis=0).to_numpy(), np.zeros(2), atol=1e-7)


def test_fit_preprocessor_supports_none_scaler():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    preprocessor = data_preprocessing.fit_preprocessor(
        X,
        variance_threshold=0.0,
        corr_threshold=1.0,
        clip_range=(-1e10, 1e10),
        scaler="none",
    )
    transformed = data_preprocessing.transform_preprocessor(X, preprocessor)

    assert preprocessor["scaler_name"] == "none"
    assert transformed.equals(X)


def test_fit_preprocessor_supports_minmax_scaler():
    X_train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )
    X_eval = pd.DataFrame(
        {
            "a": [0.0, 4.0],
            "b": [0.0, 40.0],
        }
    )

    preprocessor = data_preprocessing.fit_preprocessor(
        X_train,
        variance_threshold=0.0,
        corr_threshold=1.0,
        clip_range=(-1e10, 1e10),
        scaler="minmax",
    )
    transformed = data_preprocessing.transform_preprocessor(X_eval, preprocessor)

    assert preprocessor["scaler_name"] == "minmax"
    assert np.allclose(transformed.min(axis=0).to_numpy(), np.zeros(2), atol=1e-7)
    assert np.allclose(transformed.max(axis=0).to_numpy(), np.ones(2), atol=1e-7)


def test_fit_preprocessor_rejects_unknown_scaler():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="preprocess.scaler"):
        data_preprocessing.fit_preprocessor(
            X,
            variance_threshold=0.0,
            corr_threshold=1.0,
            clip_range=(-1e10, 1e10),
            scaler="banana",
        )
