import numpy as np

from heart_risk.data import FEATURES, TARGET, clean
from heart_risk.features import build_preprocessor


def _features(raw_df):
    return clean(raw_df)[FEATURES].astype("float64")


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def test_preprocessor_output_shape(raw_df):
    X = _features(raw_df)
    assert build_preprocessor().fit_transform(X).shape == (len(X), 23)


def test_preprocessor_removes_missing_values(raw_df):
    X = _features(raw_df)
    assert X.isna().any().any()
    Xt = _dense(build_preprocessor().fit_transform(X))
    assert not np.isnan(Xt).any()


def test_preprocessor_ignores_unseen_categories(raw_df):
    X = _features(raw_df)
    prep = build_preprocessor().fit(X)
    new = X.iloc[:1].copy()
    new["thal"] = 99.0
    assert prep.transform(new).shape == (1, 22)


def test_numeric_columns_are_scaled(raw_df):
    Xt = _dense(build_preprocessor().fit_transform(_features(raw_df)))
    assert abs(Xt[:, 0].mean()) < 1e-6  # first column is age (numeric block first)
    assert TARGET not in FEATURES
