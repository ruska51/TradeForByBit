import os
from datetime import datetime

import numpy as np
import pandas as pd

from logging_utils import ensure_report_schema


def _resolve_timestamp_column(df: pd.DataFrame) -> pd.Series:
    for col in ("timestamp_exit", "timestamp_close", "timestamp"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce")
    raise ValueError("no timestamp column found")


def build_profit_report(trades_log_path: str, out_path: str) -> pd.DataFrame:
    ensure_report_schema(
        out_path,
        [
            "timestamp",
            "symbol",
            "pnl_net",
            "cum_pnl",
            "winrate",
            "avg_win",
            "avg_loss",
            "sharpe",
            "max_dd",
        ],
    )
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    df = pd.read_csv(trades_log_path) if pd.io.common.file_exists(trades_log_path) else pd.DataFrame()
    if df.empty:
        out = pd.DataFrame(columns=[
            "timestamp",
            "symbol",
            "pnl_net",
            "cum_pnl",
            "winrate",
            "avg_win",
            "avg_loss",
            "sharpe",
            "max_dd",
        ])
        out.to_csv(out_path, index=False)
        return out

    df = df.copy()
    if "trade_id" in df.columns:
        df["trade_id"] = df["trade_id"].astype(str)
    df["timestamp"] = _resolve_timestamp_column(df)
    df = df.sort_values("timestamp")
    df["pnl_net"] = df.get("pnl_net", df.get("profit", 0.0))
    df["cum_pnl"] = df["pnl_net"].cumsum()
    df["win"] = df["pnl_net"] > 0
    df["winrate"] = df["win"].expanding().mean()
    wins = df["pnl_net"].where(df["pnl_net"] > 0)
    losses = df["pnl_net"].where(df["pnl_net"] <= 0)
    df["avg_win"] = wins.expanding().mean()
    df["avg_loss"] = losses.expanding().mean()
    daily = df.groupby(df["timestamp"].dt.date)["pnl_net"].sum()
    if len(daily) > 1 and daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(len(daily))
    else:
        sharpe = 0.0
    equity = df["cum_pnl"]
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd = drawdown.min()
    df["sharpe"] = sharpe
    df["max_dd"] = max_dd
    out = df[[
        "timestamp",
        "symbol",
        "pnl_net",
        "cum_pnl",
        "winrate",
        "avg_win",
        "avg_loss",
        "sharpe",
        "max_dd",
    ]]
    out.to_csv(out_path, index=False)
    return out


def build_equity_curve(trades_log_path: str, out_path: str) -> pd.DataFrame:
    ensure_report_schema(out_path, ["timestamp", "equity"])
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    df = pd.read_csv(trades_log_path) if pd.io.common.file_exists(trades_log_path) else pd.DataFrame()
    if df.empty:
        out = pd.DataFrame(columns=["timestamp", "equity"])
        out.to_csv(out_path, index=False)
        return out
    df = df.copy()
    df["timestamp"] = _resolve_timestamp_column(df)
    df = df.sort_values("timestamp")
    pnl = df.get("pnl_net", df.get("profit", 0.0))
    equity = pnl.cumsum()
    out = pd.DataFrame({"timestamp": df["timestamp"], "equity": equity})
    out.to_csv(out_path, index=False)
    return out
