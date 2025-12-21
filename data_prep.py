import numpy as np
import pandas as pd
from typing import List, Tuple


def fetch_multi_ohlcv(
    adapter, 
    symbol: str, 
    timeframes: List[str], 
    limit: int = 300
) -> dict:
    """Fetch real OHLCV data via adapter."""
    data: dict = {}
    for tf in timeframes:
        try:
            ohlcv = adapter.fetch_ohlcv(symbol, tf, limit=limit)
            if ohlcv:
                data[tf] = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                data[tf]["timestamp"] = pd.to_datetime(data[tf]["timestamp"], unit="ms")
        except Exception as e:
            import logging
            logging.warning(f"data_prep | {symbol} | {tf} fetch failed: {e}")
            continue
    return data


def build_feature_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Calculate indicators from OHLCV data."""
    df = df.copy()
    
    # Returns
    df["ret"] = df["close"].pct_change()
    
    # ATR (14-period)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    
    # RSI (14-period)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Simple moving average
    df["sma"] = df["close"].rolling(20, min_periods=1).mean()
    
    # Lag features
    for lag in range(1, 4):
        df[f"close_lag{lag}"] = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
    
    # Target (для обучения)
    horizon = 1
    threshold = 0.001
    df["delta"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    df["target"] = np.select(
        [df["delta"] > threshold, df["delta"] < -threshold],
        [1, 2],
        default=0
    )
    
    df.dropna(inplace=True)
    
    # Ensure all 3 classes exist (для обучения XGB)
    if set(df["target"].unique()) != {0, 1, 2}:
        for i, idx in enumerate(df.index[:3]):
            df.at[idx, "target"] = i
    
    return df.drop(columns=["delta"], errors='ignore').reset_index(drop=True)


def fetch_and_prepare_training_data(
    adapter,
    symbols: List[str],
    base_tf: str = "15m",
    limit: int = 300
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Fetch and prepare training data for multiple symbols."""
    rows = []
    
    for sym in symbols:
        ohlcv_dict = fetch_multi_ohlcv(adapter, sym, [base_tf], limit)
        
        if not ohlcv_dict or base_tf not in ohlcv_dict:
            continue
            
        df_raw = ohlcv_dict[base_tf]
        
        if df_raw is None or df_raw.empty:
            continue
        
        df = build_feature_dataframe(df_raw, sym)
        rows.append(df)
    
    if not rows:
        raise ValueError("no training data collected")
    
    df_all = pd.concat(rows, ignore_index=True)
    
    if "target" not in df_all.columns:
        raise ValueError("target column missing")
    
    df_features = df_all.drop(columns=["target"])
    df_target = df_all["target"]
    feature_cols = list(df_features.columns)
    
    return df_features, df_target, feature_cols