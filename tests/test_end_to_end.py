from __future__ import annotations

from datetime import datetime, timedelta, timezone
import types
import sys

import pytest

from logging_utils import log_entry, log_exit_from_order, _ENTRY_CACHE
from reporting import build_profit_report, build_equity_curve
from risk_management import should_activate_trailing, trail_levels
from memory_utils import memory_manager
from utils.csv_utils import read_csv_safe


def _iso(base: datetime, offset_min: int) -> str:
    dt = base + timedelta(minutes=offset_min)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def test_dry_cycle_builds_reports(tmp_path, monkeypatch):
    entries_log = tmp_path / "entries.csv"
    trades_log = tmp_path / "trades.csv"
    profit_report = tmp_path / "profit.csv"
    equity_curve = tmp_path / "equity.csv"

    # stub minimal main module for logging_utils
    stub = types.SimpleNamespace(open_trade_ctx={}, register_trade_result=None)
    monkeypatch.setitem(sys.modules, "main", stub)

    # redirect memory manager to temp file to avoid polluting repo
    memory_manager.path = str(tmp_path / "mem.json")
    memory_manager.data = []

    _ENTRY_CACHE.clear()

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = [
        ("BTC/USDT", 100.0, {"type": "TAKE_PROFIT_MARKET", "avgPrice": 110.0}, {}),
        ("ETH/USDT", 200.0, {"type": "STOP_MARKET", "avgPrice": 190.0}, {}),
        ("XRP/USDT", 50.0, {"type": "STOP_MARKET", "avgPrice": 55.0}, {"trailing_profit_used": True}),
        ("LTC/USDT", 80.0, {"type": "MARKET", "avgPrice": 78.0}, {"exit_type_hint": "TIME"}),
    ]

    trail_vals: list[float] = []
    for idx, (sym, entry_price, exit_order, extra_ctx) in enumerate(trades):
        entry_ctx = {
            "entry_price": entry_price,
            "qty": 1.0,
            "side": "LONG",
            "entry_time": _iso(base, idx),
        }
        trade_id = log_entry(sym, entry_ctx, str(entries_log))
        stub.open_trade_ctx[sym] = {**entry_ctx, "trade_id": trade_id, **extra_ctx}
        assert log_exit_from_order(sym, exit_order, 0.0, str(trades_log))

        if sym == "XRP/USDT":
            # check trailing monotonicity
            state: dict = {}
            current_sl = entry_price - 1.0
            tick = 0.01
            atr = 1.0
            for last in [entry_price + 1, entry_price + 2, entry_price + 3]:
                assert should_activate_trailing("LONG", entry_price, last, 1.0, atr)
                new_sl, be_flag = trail_levels(
                    "LONG",
                    entry_price,
                    last,
                    atr,
                    tick,
                    state.get("breakeven_done", False),
                    current_sl,
                    sym,
                )
                state["breakeven_done"] = be_flag
                current_sl = float(new_sl)
                trail_vals.append(current_sl)

    assert trail_vals == sorted(trail_vals)

    df_entries = read_csv_safe(entries_log)
    df_trades = read_csv_safe(trades_log)
    df_trades.rename(
        columns={"timestamp_exit": "timestamp_close", "timestamp_entry": "entry_time"},
        inplace=True,
    )
    df_trades.to_csv(trades_log, index=False)

    assert df_entries["trade_id"].is_unique
    assert df_trades["trade_id"].is_unique
    assert len(df_entries) == len(trades) == len(df_trades)

    allowed = {"TP", "SL", "TRAIL_STOP", "TIME"}
    assert set(df_trades["exit_type"]) == allowed

    build_profit_report(str(trades_log), str(profit_report))
    build_equity_curve(str(trades_log), str(equity_curve))

    df_profit = read_csv_safe(profit_report)
    df_equity = read_csv_safe(equity_curve)

    assert len(df_profit) == len(trades)
    total_pnl = df_trades["pnl_net"].sum()
    assert pytest.approx(total_pnl) == df_profit["pnl_net"].sum()
    assert pytest.approx(total_pnl) == df_equity["equity"].iloc[-1]
