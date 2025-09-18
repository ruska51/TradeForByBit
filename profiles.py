"""Parameter presets for different trading risk profiles."""
from __future__ import annotations

import os
from dataclasses import replace
from main import StrategyParams


def load_profile(name: str, overrides: dict[str, float] | None = None) -> StrategyParams:
    """Return a ``StrategyParams`` instance for the given profile name.

    Parameters can be tweaked either by providing a mapping of ``overrides``
    or by defining environment variables (``PROBA_FILTER``, ``ADX_THRESHOLD``,
    ``RSI_OVERBOUGHT`` and ``RSI_OVERSOLD``). The values are applied on top of
    the preset profile without mutating the original ``PROFILES`` entry.
    """

    params = PROFILES[name]
    if overrides:
        params = replace(
            params, **{k: overrides[k] for k in overrides if hasattr(params, k)}
        )
    env_keys = ("PROBA_FILTER", "ADX_THRESHOLD", "RSI_OVERBOUGHT", "RSI_OVERSOLD")
    env_overrides = {k: os.getenv(k) for k in env_keys}
    env_overrides = {k: float(v) for k, v in env_overrides.items() if v is not None}
    if env_overrides:
        params = replace(params, **env_overrides)
    return params


PROFILES: dict[str, StrategyParams] = {
    "conservative": StrategyParams(
        THRESHOLD=0.002,
        SL_PCT=0.015,
        TP_PCT=0.03,
        HORIZON=2,
        PROBA_FILTER=0.65,
        ADX_THRESHOLD=20,
        RSI_OVERBOUGHT=70,
        RSI_OVERSOLD=30,
    ),
    "aggressive": StrategyParams(
        THRESHOLD=0.0005,
        SL_PCT=0.03,
        TP_PCT=0.06,
        HORIZON=2,
        PROBA_FILTER=0.5,
        ADX_THRESHOLD=12,
        RSI_OVERBOUGHT=80,
        RSI_OVERSOLD=20,
    ),
}
