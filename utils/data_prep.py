import pandas as pd
import numpy as np
import logging
from typing import Any, Iterable


def _to_dataframe(obj: Any) -> pd.DataFrame:
    """
    Нормализует входные данные OHLCV: принимает DataFrame, Series, list[list] или
    list[dict], возвращает pd.DataFrame с колонками
    [timestamp, open, high, low, close, volume]. Если вход пустой или
    некорректный, возвращает пустой DataFrame.
    """
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    try:
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            rows = list(obj)
            if not rows:
                return pd.DataFrame()
            if isinstance(rows[0], Iterable) and not isinstance(rows[0], (str, bytes)) and len(rows[0]) >= 6:
                cols = ["timestamp", "open", "high", "low", "close", "volume"]
                return pd.DataFrame(rows, columns=cols)
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def build_feature_dataframe(df_ohlcv: pd.DataFrame, symbol: str, thr: float = 0.002) -> pd.DataFrame:
    """
    Построить набор простых признаков и целевую метку для данного символа.
    df_ohlcv должен иметь колонки [timestamp, open, high, low, close, volume].
    """
    df = _to_dataframe(df_ohlcv)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = df["ret_1"].clip(lower=0).rolling(14).mean() / (
        df["ret_1"].abs().rolling(14).mean() + 1e-9
    )

    df["ema_fast"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp20 = typical.rolling(20).mean()
    mean_dev = (typical - sma_tp20).abs().rolling(20).mean()
    df["cci"] = (typical - sma_tp20) / (0.015 * (mean_dev.replace(0, np.nan)))
    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"].fillna(0)).cumsum()

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr14 = tr.rolling(14).mean()
    plus_dm = (df["high"].diff()).clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr14.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)

    # ===  КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильный таргет - смотрим ВПЕРЁД на 5 свечей ===
    # Расчёт будущей доходности (через 5 свечей от текущей)
    future_ret_5 = df["close"].shift(-5) / df["close"] - 1
    # Target: 1 (long) если будущая доходность > thr, 2 (short) если < -thr, иначе 0 (hold)
    target = np.where(future_ret_5 > thr, 1, np.where(future_ret_5 < -thr, 2, 0))
    feats = df[
        [
            "ret_1",
            "ret_5",
            "sma_10",
            "sma_50",
            "ema_fast",
            "ema_slow",
            "rsi_14",
            "adx",
            "plus_di",
            "minus_di",
            "volume_ratio",
            "cci",
            "obv",
        ]
    ].copy()
    feats["symbol_cat"] = hash(symbol) % 17
    feats["target"] = target
    feats.dropna(inplace=True)
    return feats


def fetch_and_prepare_training_data(adapter, symbols: list[str], base_tf: str = "15m", limit: int = 400):
    """
    Собирает OHLCV через adapter.fetch_ohlcv для каждого символа,
    строит фичи и целевую метку. Возвращает (df_features, df_target, feature_cols).
    Если нет данных ни по одному символу — поднимает ValueError.
    """
    try:
        from exchange_adapter import fetch_ohlcv as _fetch_ohlcv  # type: ignore
    except Exception:
        _fetch_ohlcv = None
    rows = []
    for sym in symbols:
        try:
            df_raw = (
                _fetch_ohlcv(sym, base_tf, limit=limit)
                if _fetch_ohlcv is not None
                else adapter.fetch_ohlcv(sym, base_tf, limit=limit)
            )
            feat = build_feature_dataframe(df_raw, sym)
            if len(feat) > 0:
                rows.append(feat)
        except Exception as e:
            logging.warning(
                "data_prep | %s | failed to fetch/build features: %s (type=%s)",
                sym,
                e,
                type(df_raw).__name__ if "df_raw" in locals() else "unknown",
            )
    if not rows:
        raise ValueError("data_prep | no training data collected: all rows empty")
    df_all = pd.concat(rows, ignore_index=True)
    df_features = df_all.drop(columns=["target"])
    df_target = df_all["target"]
    feature_cols = list(df_features.columns)
    return df_features, df_target, feature_cols
