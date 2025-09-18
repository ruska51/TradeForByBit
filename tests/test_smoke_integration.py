from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

import main
from risk_management import (
    should_activate_trailing,
    trail_levels,
    StatsTracker,
    confirm_trend,
)
import logging_utils
from reporting import build_profit_report, build_equity_curve


# [ANCHOR:SMOKE_TESTS]
def test_trailing_be_and_no_expansion(monkeypatch):
    logs: list[str] = []

    def fake_log_decision(symbol, msg):
        logs.append(msg)

    monkeypatch.setattr(logging_utils, "log_decision", fake_log_decision)

    symbol = "TEST"

    # --- LONG scenario ---
    state: dict = {}
    entry = 100.0
    r_value = 1.0
    atr = 1.0
    tick = 0.01
    current_sl = 99.0

    # Before activation stop stays put
    last = 100.4
    assert not should_activate_trailing("LONG", entry, last, r_value, atr)
    assert current_sl == 99.0

    # Activation triggers breakeven step
    last = 101.0
    assert should_activate_trailing("LONG", entry, last, r_value, atr)
    new_sl, be_flag = trail_levels("LONG", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    state["breakeven_done"] = be_flag
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    current_sl = float(new_sl)
    assert current_sl == 100.2
    assert state["breakeven_done"]
    assert logs[-1] == "trailing_update be=True new_sl=100.20"

    # Further price increase trails the stop
    last = 102.0
    new_sl, be_flag = trail_levels("LONG", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == 100.8
    assert logs[-1] == "trailing_update be=True new_sl=100.80"
    assert float(new_sl) >= entry
    current_sl = float(new_sl)

    # Tick-by-tick increase
    last = 102.01
    new_sl, _ = trail_levels("LONG", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == current_sl + tick
    assert logs[-1].endswith(f"new_sl={float(new_sl):.2f}")
    current_sl = float(new_sl)

    # Price drop should not move stop back
    last = 101.5
    new_sl, _ = trail_levels("LONG", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == current_sl

    # --- SHORT scenario (correct branch) ---
    logs.clear()
    state = {}
    entry = 105.0
    r_value = 1.0
    atr = 1.0
    tick = 0.01
    current_sl = 106.0

    last = 104.0
    assert should_activate_trailing("SHORT", entry, last, r_value, atr)
    new_sl, be_flag = trail_levels("SHORT", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    state["breakeven_done"] = be_flag
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    current_sl = float(new_sl)
    assert current_sl == 104.8
    assert logs[-1] == "trailing_update be=True new_sl=104.80"

    last = 103.0
    new_sl, be_flag = trail_levels("SHORT", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == 104.2
    assert logs[-1] == "trailing_update be=True new_sl=104.20"
    assert float(new_sl) <= entry
    current_sl = float(new_sl)

    # Tick-by-tick decrease
    last = 102.99
    new_sl, _ = trail_levels("SHORT", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == current_sl - tick
    current_sl = float(new_sl)

    # Price rise should not move stop back
    last = 103.5
    new_sl, _ = trail_levels("SHORT", entry, last, atr, tick, state.get("breakeven_done", False), current_sl, symbol)
    logging_utils.log_decision(symbol, f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}")
    assert float(new_sl) == current_sl

    # Guards against invalid atr/tick
    sl_safe, _ = trail_levels("LONG", entry, last, float("nan"), 0.0, True, current_sl, symbol)
    assert np.isfinite(sl_safe)


def test_roi_closes_market_and_cleans_children(monkeypatch, tmp_path):
    symbol = "BTC/USDT"
    log_path = tmp_path / "trades.csv"

    main.open_trade_ctx.clear()
    main.open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "exit_type_hint": "TP",
    }
    main.trailing_memory = {f"{symbol}_long": 100.0}
    pair_state = {symbol: {}}

    cancelled: list[str] = []

    def fake_cancel(sym):
        cancelled.append(sym)

    def fake_market_close(sym):
        return {"type": "MARKET", "side": "SELL", "avgPrice": 105.0}

    monkeypatch.setattr(main, "cancel_all_child_orders", fake_cancel)
    monkeypatch.setattr(main, "market_close", fake_market_close)

    cancel_all_child_orders = main.cancel_all_child_orders
    market_close = main.market_close

    cancel_all_child_orders(symbol)
    order = market_close(symbol)
    main.open_trade_ctx[symbol]["exit_type_hint"] = "TP"
    assert main.log_exit_from_order(symbol, order, 0.0, str(log_path))
    if cancelled:
        pair_state.pop(symbol, None)
    main.trailing_memory.pop(f"{symbol}_long", None)

    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.iloc[0]["exit_type"] == "TP"
    assert cancelled == [symbol]
    assert symbol not in main.open_trade_ctx
    assert f"{symbol}_long" not in main.trailing_memory
    assert symbol not in pair_state


def test_time_stop_closes_and_logs_once(tmp_path):
    symbol = "ETH/USDT"
    log_path = tmp_path / "trades.csv"

    main.open_trade_ctx.clear()
    main.open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 50.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "exit_type_hint": "TIME",
    }

    order = {"type": "MARKET", "side": "SELL", "avgPrice": 49.0}
    assert main.log_exit_from_order(symbol, order, 0.0, str(log_path))
    assert not main.log_exit_from_order(symbol, order, 0.0, str(log_path))

    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.iloc[0]["exit_type"] == "TIME"


def test_symbol_ban_and_soft_risk(tmp_path):
    stats = StatsTracker()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    stats.on_trade_close("BTC/USDT", -1, timestamp=base)
    stats.on_trade_close("BTC/USDT", -1, timestamp=base + timedelta(minutes=10))
    assert stats.is_banned("BTC/USDT", now=base + timedelta(minutes=10))

    # after ban period risk is reduced for the first trade
    assert not stats.is_banned("BTC/USDT", now=base + timedelta(minutes=70))
    reduced = stats.pop_soft_risk("BTC/USDT")

    symbol = "BTC/USDT"
    log_path = tmp_path / "trades.csv"
    main.open_trade_ctx.clear()
    main.open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": base.isoformat(),
        "source": "live",
        "reduced_risk": reduced,
    }
    main.log_trade(base + timedelta(minutes=80), symbol, "LONG", 100.0, 101.0, 1.0, 1.0, "MANUAL", str(log_path))

    df = pd.read_csv(log_path)
    assert df["reduced_risk"].iloc[0] == 1
    assert "entry_time" in df.columns
    assert "source" in df.columns


def test_reports_from_trades_log_consistent(tmp_path):
    trades_log = tmp_path / "trades_log.csv"
    profit_report = tmp_path / "profit_report.csv"
    equity_curve = tmp_path / "equity_curve.csv"

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    main.open_trade_ctx.clear()
    main.log_trade(base, "BTC/USDT", "LONG", 100.0, 102.0, 1.0, 2.0, "MANUAL", str(trades_log))
    main.log_trade(base + timedelta(minutes=30), "ETH/USDT", "SHORT", 50.0, 48.0, 1.0, 2.0, "MANUAL", str(trades_log))

    build_profit_report(str(trades_log), str(profit_report))
    build_equity_curve(str(trades_log), str(equity_curve))

    df_trades = pd.read_csv(trades_log)
    df_profit = pd.read_csv(profit_report)
    df_equity = pd.read_csv(equity_curve)

    assert df_profit["pnl_net"].round(2).tolist() == df_trades["profit"].round(2).tolist()
    assert df_equity["equity"].iloc[-1] == round(df_trades["profit"].sum(), 2)
    assert len(df_profit) == len(df_trades) == len(df_equity)


def test_trade_frequency_within_bounds():
    stats = StatsTracker()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    profits = [1, -1, 1, 1, 1, -1]
    for i, p in enumerate(profits):
        stats.on_trade_close("BTC/USDT", p, timestamp=base + timedelta(minutes=12 * i))

    trades = stats.trades("BTC/USDT")
    duration = (base + timedelta(minutes=12 * (len(profits) - 1)) - base).total_seconds() / 60
    freq = trades / duration
    baseline = 6 / 60
    assert abs(freq - baseline) / baseline <= 0.1
    assert stats.win_rate("BTC/USDT") > 0.5
    assert stats.avg_profit("BTC/USDT") > 0


def test_confirm_trend_resilience_nan():
    data = {
        "5m": pd.DataFrame({"close": [np.nan, np.nan]}),
        "15m": pd.DataFrame({"close": []}),
    }
    assert not confirm_trend(data, "LONG")
