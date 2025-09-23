import os
import pandas as pd
import numpy as np

from logging_utils import ensure_report_schema
from utils.csv_utils import read_csv_safe


def _load_trades(trades_log: str) -> pd.DataFrame:
    """Return DataFrame from *trades_log* or empty DataFrame on error."""
    if not os.path.exists(trades_log):
        return pd.DataFrame()
    try:
        return read_csv_safe(trades_log)
    except Exception:
        return pd.DataFrame()


# [ANCHOR:REPORTS_BUILDERS]
def build_profit_report(trades_log: str, profit_report: str) -> None:
    """Aggregate trade metrics from ``trades_log`` into ``profit_report``.

    The resulting CSV contains per-trade statistics derived solely from the
    trade log: entry/exit timestamps, symbol, ``pnl_net`` and cumulative
    performance metrics (``cum_pnl``, ``winrate``, ``avg_win``/``avg_loss``,
    daily ``sharpe`` and running drawdown ``dd``).
    """

    ensure_report_schema(
        profit_report,
        [
            "trade_id",
            "timestamp_entry",
            "timestamp_exit",
            "symbol",
            "pnl_net",
            "cum_pnl",
            "winrate",
            "avg_win",
            "avg_loss",
            "sharpe",
            "dd",
        ],
    )

    if os.path.exists(profit_report):
        try:
            os.remove(profit_report)
        except OSError:
            pass

    df = _load_trades(trades_log)
    if df.empty:
        empty = pd.DataFrame(
            columns=
            [
                "trade_id",
                "timestamp_entry",
                "timestamp_exit",
                "symbol",
                "pnl_net",
                "cum_pnl",
                "winrate",
                "avg_win",
                "avg_loss",
                "sharpe",
                "dd",
            ]
        )
        empty.to_csv(profit_report, index=False)
        return

    ts_exit_col = None
    for name in ("timestamp_exit", "timestamp_close", "exit_time"):
        if name in df.columns:
            ts_exit_col = name
            break
    ts_entry_col = None
    for name in ("timestamp_entry", "entry_time", "timestamp"):
        if name in df.columns:
            ts_entry_col = name
            break
    profit_col = "pnl_net" if "pnl_net" in df.columns else "profit" if "profit" in df.columns else None

    if not ts_exit_col or not ts_entry_col or not profit_col or "symbol" not in df.columns:
        return

    df[ts_exit_col] = pd.to_datetime(df[ts_exit_col], utc=True, errors="coerce")
    df[ts_entry_col] = pd.to_datetime(df[ts_entry_col], utc=True, errors="coerce")

    df.dropna(subset=[ts_exit_col, ts_entry_col, "symbol", profit_col], inplace=True)
    if df.empty:
        return

    df.sort_values(ts_exit_col, inplace=True)

    pnl = df[profit_col].astype(float)

    df_report = pd.DataFrame(
        {
            "trade_id": df["trade_id"] if "trade_id" in df.columns else range(len(df)),
            "timestamp_entry": df[ts_entry_col],
            "timestamp_exit": df[ts_exit_col],
            "symbol": df["symbol"],
            "pnl_net": pnl.round(2),
        }
    )

    if "trade_id" in df_report.columns:
        df_report["trade_id"] = df_report["trade_id"].astype(str)

    df_report["cum_pnl"] = pnl.cumsum().round(2)

    wins = (pnl > 0).astype(float)
    df_report["winrate"] = wins.expanding().mean().round(4)
    df_report["avg_win"] = (
        pnl.where(pnl > 0).expanding().mean().fillna(0).round(2)
    )
    df_report["avg_loss"] = (
        pnl.where(pnl <= 0).expanding().mean().fillna(0).round(2)
    )

    dates = df_report["timestamp_exit"].dt.floor("D")
    daily_returns = df_report.groupby(dates)["pnl_net"].sum().astype(float)
    sharpe_series = (
        daily_returns.expanding().mean()
        / daily_returns.expanding().std()
    ).fillna(0) * np.sqrt(365)
    df_report["sharpe"] = dates.map(sharpe_series).round(3)

    peak = df_report["cum_pnl"].cummax()
    df_report["dd"] = (df_report["cum_pnl"] - peak).round(2)

    for col in ("timestamp_entry", "timestamp_exit"):
        df_report[col] = (
            df_report[col]
            .dt.round("ms")
            .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            .str[:-3]
            + "Z"
        )

    df_report.to_csv(profit_report, index=False)


def build_equity_curve(trades_log: str, equity_curve: str) -> None:
    """Construct cumulative equity solely from ``trades_log``."""

    ensure_report_schema(equity_curve, ["timestamp", "equity"])

    if os.path.exists(equity_curve):
        try:
            os.remove(equity_curve)
        except OSError:
            pass

    df = _load_trades(trades_log)
    if df.empty:
        pd.DataFrame(columns=["timestamp", "equity"]).to_csv(equity_curve, index=False)
        return

    ts_exit_col = None
    for name in ("timestamp_exit", "timestamp_close", "exit_time"):
        if name in df.columns:
            ts_exit_col = name
            break
    profit_col = "pnl_net" if "pnl_net" in df.columns else "profit" if "profit" in df.columns else None
    if not ts_exit_col or not profit_col:
        return

    df[ts_exit_col] = pd.to_datetime(df[ts_exit_col], utc=True, errors="coerce")
    df.dropna(subset=[ts_exit_col, profit_col], inplace=True)
    if df.empty:
        return

    df.sort_values(ts_exit_col, inplace=True)

    equity = df[profit_col].astype(float).cumsum().round(2)
    df_curve = pd.DataFrame(
        {
            "timestamp": df[ts_exit_col]
            .dt.round("ms")
            .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            .str[:-3]
            + "Z",
            "equity": equity,
        }
    )
    df_curve.to_csv(equity_curve, index=False)
