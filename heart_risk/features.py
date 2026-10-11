"""Preprocessing pipeline shared by training and the serving API."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_risk.data import FEATURES

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # ca: vessel count
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]  # category codes

assert set(NUMERIC + BINARY + NOMINAL) == set(FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and encode. Fitted on training data only, then reused at serving."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    nominal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("nom", nominal, NOMINAL),
        ("bin", "passthrough", BINARY),
    ])