import pandas as pd
from data_prep import build_feature_dataframe

def test_build_feature_dataframe_handles_short_dataframe():
    df = pd.DataFrame({
        "open": [1, 1],
        "high": [1, 1],
        "low": [1, 1],
        "close": [1, 1],
        "volume": [1, 1],
    })
    result = build_feature_dataframe(df, "SYM")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
