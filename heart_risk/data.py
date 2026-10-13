"""Download and clean the UCI Heart Disease dataset."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
CLEAN_PATH = Path("data/processed/heart_disease_clean.csv")
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"


def download_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Fetch the dataset from the UCI repository (id=45) and save it as CSV."""
    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=45)
    df = heart.data.features.copy()
    df["num"] = heart.data.targets["num"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target (0 = no disease, 1 = disease) and drop duplicates.

    Missing values (ca, thal) are kept here and imputed inside the model
    pipeline, so imputation statistics are learned from training data only.
    """
    df = df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    return df[FEATURES + [TARGET]]


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    raw = download_raw()
    cleaned = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print("shape:", cleaned.shape)
    print("missing values:\n", cleaned.isna().sum()[cleaned.isna().sum() > 0])
    print("class balance:\n", cleaned[TARGET].value_counts())
