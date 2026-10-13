import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart_risk.data import FEATURES, TARGET, clean
from heart_risk.features import build_preprocessor
from heart_risk.predict import predict_one


@pytest.fixture
def model_file(raw_df, tmp_path):
    df = clean(raw_df)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(df[FEATURES].astype("float64"), df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def record(raw_df):
    return clean(raw_df)[FEATURES].iloc[2].to_dict()


def test_predict_one_returns_expected_fields(model_file, record):
    out = predict_one(record, path=model_file)
    assert set(out) == {"prediction", "label", "confidence", "probability_disease"}
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert 0.0 <= out["probability_disease"] <= 1.0


def test_predict_one_rejects_missing_features(model_file, record):
    record.pop("age")
    with pytest.raises(ValueError):
        predict_one(record, path=model_file)


def test_predict_one_accepts_missing_ca_and_thal(model_file, record):
    record["ca"] = None
    record["thal"] = None
    assert predict_one(record, path=model_file)["prediction"] in (0, 1)
