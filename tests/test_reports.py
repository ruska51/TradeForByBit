from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from reporting import build_profit_report, build_equity_curve
from utils.csv_utils import read_csv_safe


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_profit_and_equity_reports(tmp_path):
    trades_path = tmp_path / "trades_log.csv"
    profit_path = tmp_path / "profit_report.csv"
    equity_path = tmp_path / "equity_curve.csv"

    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp_close": _iso(base),
            "entry_time": _iso(base - timedelta(hours=1)),
            "symbol": "BTC/USDT",
            "profit": 10.0,
        },
        {
            "timestamp_close": _iso(base + timedelta(hours=1)),
            "entry_time": _iso(base - timedelta(minutes=30)),
            "symbol": "ETH/USDT",
            "profit": -5.0,
        },
        {
            "timestamp_close": _iso(base + timedelta(hours=2)),
            "entry_time": _iso(base + timedelta(minutes=30)),
            "symbol": "XRP/USDT",
            "profit": 15.0,
        },
    ]
    pd.DataFrame(rows).to_csv(trades_path, index=False)

    build_profit_report(str(trades_path), str(profit_path))
    build_equity_curve(str(trades_path), str(equity_path))

    df_profit = read_csv_safe(profit_path)
    df_equity = read_csv_safe(equity_path)

    assert df_profit["pnl_net"].round(2).tolist() == [10.0, -5.0, 15.0]
    assert df_profit["cum_pnl"].round(2).tolist() == [10.0, 5.0, 20.0]
    assert pytest.approx(df_profit["winrate"].iloc[-1], rel=1e-3) == 2 / 3
    assert df_equity["equity"].round(2).tolist() == [10.0, 5.0, 20.0]
    assert pd.to_datetime(df_equity["timestamp"]).is_monotonic_increasing
