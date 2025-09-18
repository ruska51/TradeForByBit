"""Simple rule-based fallback trading signals."""
from __future__ import annotations

import pandas as pd
import logging
from logging_utils import log
from risk_management import confirm_trend

from exchange_adapter import ExchangeAdapter

ADAPTER: ExchangeAdapter | None = None


def _get_adapter() -> ExchangeAdapter:
    global ADAPTER
    if ADAPTER is None:
        from main import ADAPTER as MAIN_ADAPTER  # type: ignore
        ADAPTER = MAIN_ADAPTER
    return ADAPTER


def ema_crossover(df: pd.DataFrame) -> str:
    """Return 'long' if EMA9>EMA21, 'short' if EMA9<EMA21, else 'hold'."""
    ema_fast = df['close'].ewm(span=9).mean()
    ema_slow = df['close'].ewm(span=21).mean()
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "long"
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "short"
    return "hold"


def fallback_signal(
    symbol: str,
    timeframe: str = "15m",
    limit: int = 200,
    prob_threshold: float = 0.55,
) -> str:
    """Fetch data and produce a rule-based fallback signal.

    A position is opened only when the heuristic probability of the signal is
    above ``prob_threshold`` **and** the trend aligns across the 5m, 15m, 30m and
    1h timeframes.  This keeps the original behaviour for tests while providing
    a slightly more realistic entry filter for real trading.
    """

    try:
        base = _get_adapter().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = {
            timeframe: base,
            "5m": _get_adapter().fetch_ohlcv(symbol, timeframe="5m", limit=limit),
            "30m": _get_adapter().fetch_ohlcv(symbol, timeframe="30m", limit=limit),
            "1h": _get_adapter().fetch_ohlcv(symbol, timeframe="1h", limit=limit),
        }
    except Exception as e:  # pragma: no cover - network errors
        log(logging.ERROR, "fallback", symbol, f"fetch failed: {e}")
        return "hold"

    frames = {k: pd.DataFrame(v, columns=["ts", "open", "high", "low", "close", "volume"]) for k, v in data.items()}
    df = frames[timeframe]
    signal = ema_crossover(df)

    # quick probability proxy: distance between fast and slow EMA relative to price
    ema_fast = df['close'].ewm(span=9).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=21).mean().iloc[-1]
    prob = min(1.0, abs(ema_fast - ema_slow) / max(df['close'].iloc[-1], 1e-9))

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]

    if prob < prob_threshold:
        return "hold"

    side = "LONG" if signal == "long" else "SHORT" if signal == "short" else None
    if side and confirm_trend(frames, side):
        if signal == "long" and 30 < rsi_val < 70:
            return "long"
        if signal == "short" and 30 < rsi_val < 70:
            return "short"
    return "hold"
