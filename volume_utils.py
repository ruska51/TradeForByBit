import logging
import pandas as pd
import numpy as np
from typing import Dict

# [ANCHOR:VOL_CONSTS]
VOL_WINDOW = 20
VOL_RATIO_MAX = 10.0

# [ANCHOR:VOL_HELPERS]
def _safe_vol_ratio(series: pd.Series | None, window: int) -> float | None:
    """Return volume ratio or ``None`` if data insufficient."""
    try:
        if series is None:
            return None
        series = series.dropna()
        if len(series) < window:
            return None
        mean_volume = series.rolling(window).mean().iloc[-1]
        # [ANCHOR:VOL_RATIO_FIX]
        avg = max(1e-9, float(mean_volume)) if np.isfinite(mean_volume) else 1e-9
        curr = float(series.iloc[-1])
        vol_ratio = curr / avg
        vol_ratio = max(0.08, min(vol_ratio, VOL_RATIO_MAX))
        return vol_ratio
    except Exception:
        return None

# [ANCHOR:VOL_REASON_HELPER]
def _volume_reason(vol_ratio: float | None, min_ratio: float) -> str | None:
    """Return ``vol_missing``/``vol_low`` depending on ``vol_ratio``."""
    if vol_ratio is None:
        return "vol_missing"
    if vol_ratio < min_ratio:
        return "vol_low"
    return None

# [ANCHOR:VOL_API]
_VOL_CACHE: Dict[str, float] = {}
_ATR_CACHE: Dict[str, float] = {}


def reset_caches() -> None:
    """Clear cached ATR and volume ratios."""
    _VOL_CACHE.clear()
    _ATR_CACHE.clear()


def safe_vol_ratio(series: pd.Series | None, window: int, key: str | None = None) -> float:
    """Return smoothed volume ratio with caching.

    When ``series`` lacks sufficient data the last cached ratio for ``key`` is
    returned, defaulting to ``1.0``.  For valid ratios an exponential moving
    average is applied to dampen noise which helps the trading loop avoid
    getting stuck in ``vol_missing`` states.
    """

    ratio = _safe_vol_ratio(series, window)
    if key is not None:
        prev = _VOL_CACHE.get(key)
        if ratio is None:
            ratio = prev if prev is not None else 1.0
        else:
            if prev is not None:
                ratio = 0.7 * prev + 0.3 * ratio
            _VOL_CACHE[key] = ratio
    if ratio is None:
        logging.info("volume | safe_vol_ratio | insufficient data; defaulting to 1.0")
        return 1.0
    return ratio


def safe_atr(series: pd.Series | None, key: str | None = None) -> float | None:
    """Return smoothed ATR value with caching."""

    val: float | None = None
    try:
        if series is not None:
            series = series.dropna()
            if not series.empty:
                val = float(series.iloc[-1])
    except Exception:
        val = None
    if key is not None:
        prev = _ATR_CACHE.get(key)
        if val is None:
            val = prev
        else:
            if prev is not None:
                val = 0.7 * prev + 0.3 * val
            _ATR_CACHE[key] = val
    return val


def volume_reason(series: pd.Series | None, min_ratio: float, window: int) -> str | None:
    """Return volume reason based on series and thresholds."""
    return _volume_reason(_safe_vol_ratio(series, window), min_ratio)
