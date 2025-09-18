"""Backtest metric calculations used for performance evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_metrics(equity: pd.Series) -> dict[str, float]:
    """Return Sharpe ratio, drawdown and other metrics for an equity curve."""
    returns = equity.pct_change(fill_method=None).dropna()
    if returns.empty:
        return {
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "profit_pct": float("nan"),
            "recovery_factor": float("nan"),
            "winrate": float("nan"),
        }
    std = returns.std()
    if std == 0 or np.isnan(std):
        sharpe = float("nan")
    else:
        sharpe = (returns.mean() / std) * np.sqrt(365)
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).min()
    profit_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    recovery_factor = profit_pct / abs(drawdown * 100) if drawdown != 0 else float("nan")
    winrate = (returns > 0).mean() if len(returns) else float("nan")
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
        "profit_pct": float(profit_pct),
        "recovery_factor": float(recovery_factor),
        "winrate": float(winrate),
    }
