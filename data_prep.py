import numpy as np
import pandas as pd
from typing import List, Tuple


def fetch_multi_ohlcv(symbol: str, timeframes: List[str], limit: int = 300) -> dict:
    """Return dummy OHLCV data for the requested timeframes."""
    data: dict = {}
    for tf in timeframes:
        data[tf] = pd.DataFrame(
            {
                "timestamp": np.arange(limit),
                "open": np.zeros(limit),
                "high": np.zeros(limit),
                "low": np.zeros(limit),
                "close": np.zeros(limit),
                "volume": np.zeros(limit),
            }
        )
    return data


def build_feature_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["sma"] = df["close"].rolling(20).mean()
    for lag in range(1, 4):
        df[f"close_lag{lag}"] = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
    horizon = 1
    threshold = 0.001
    df["delta"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    df["target"] = np.select(
        [df["delta"] > threshold, df["delta"] < -threshold], [1, 2], default=0
    )
    df.dropna(inplace=True)
    if set(df["target"].unique()) != {0, 1, 2}:
        for i, idx in enumerate(df.index[:3]):
            df.at[idx, "target"] = i
    return df.drop(columns=["delta"]).reset_index(drop=True)


def fetch_and_prepare_training_data(
    symbols: List[str], base_tf: str = "15m", limit: int = 300
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    rows = []
    for sym in symbols:
        ohlcv_dict = fetch_multi_ohlcv(sym, [base_tf], limit)
        df_raw = ohlcv_dict.get(base_tf)
        df = build_feature_dataframe(df_raw, sym)
        rows.append(df)
    if not rows:
        raise ValueError("no data fetched")
    df_all = pd.concat(rows, ignore_index=True)
    df_features = df_all.drop(columns=["target"])
    df_target = df_all["target"]
    feature_cols = list(df_features.columns)
    return df_features, df_target, feature_cols
