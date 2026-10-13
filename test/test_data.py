import pandas as pd

from heart_risk.data import FEATURES, TARGET, clean


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns


def test_clean_keeps_expected_columns(raw_df):
    assert list(clean(raw_df).columns) == FEATURES + [TARGET]


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_clean_keeps_missing_values_for_the_pipeline(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1
