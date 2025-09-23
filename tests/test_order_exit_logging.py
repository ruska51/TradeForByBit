from datetime import datetime, timezone
import sys
import types

import pytest

from logging_utils import log_exit_from_order
from utils.csv_utils import read_csv_safe


_processed_order_ids: set[str] = set()


@pytest.fixture
def main_ctx(monkeypatch):
    stub = types.SimpleNamespace(open_trade_ctx={}, register_trade_result=None)
    monkeypatch.setitem(sys.modules, 'main', stub)
    return stub.open_trade_ctx


def _extract_commission(order) -> float:
    fee = order.get("fee") or order.get("info", {}).get("fee") or order.get("info", {}).get("cum_fee")
    if isinstance(fee, dict):
        return float(fee.get("cost") or fee.get("value") or 0.0)
    if fee is not None:
        return float(fee)
    return 0.0


def test_log_exit_from_stop_order(tmp_path, main_ctx):
    path = tmp_path / "trades.csv"
    symbol = "BTC/USDT"
    open_trade_ctx = main_ctx
    open_trade_ctx.clear()
    open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "trade_id": "t1",
    }
    order = {"type": "STOP_MARKET", "avgPrice": 95.0}
    assert log_exit_from_order(symbol, order, 0.0006, str(path))
    df = read_csv_safe(path)
    assert df.iloc[0]["exit_type"] == "SL"
    assert df.iloc[0]["trade_id"] == "t1"
    assert "timestamp_entry" in df.columns
    # profit = (95-100)*1 - 0.0006*(100+95)*1 = -5 - 0.117 = -5.117
    assert abs(df.iloc[0]["profit"] + 5.117) < 1e-6
    assert not df.iloc[0]["trail_triggered"]
    assert not df.iloc[0]["time_stop_triggered"]
    assert symbol not in open_trade_ctx


def test_log_exit_trailing_stop(tmp_path, main_ctx):
    path = tmp_path / "trades.csv"
    symbol = "BTC/USDT"
    open_trade_ctx = main_ctx
    open_trade_ctx.clear()
    open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "trailing_profit_used": True,
        "trade_id": "t2",
    }
    order = {"type": "STOP_MARKET", "avgPrice": 102.0}
    assert log_exit_from_order(symbol, order, 0.0006, str(path))
    df = read_csv_safe(path)
    assert df.iloc[0]["exit_type"] == "TRAIL_STOP"
    assert df.iloc[0]["trail_triggered"]
    assert not df.iloc[0]["time_stop_triggered"]
    assert symbol not in open_trade_ctx


def test_log_exit_take_profit(tmp_path, main_ctx):
    path = tmp_path / "trades.csv"
    symbol = "BNB/USDT"
    open_trade_ctx = main_ctx
    open_trade_ctx.clear()
    open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "trade_id": "t3",
    }
    order = {"type": "TAKE_PROFIT", "avgPrice": 105.0}
    assert log_exit_from_order(symbol, order, 0.0006, str(path))
    df = read_csv_safe(path)
    assert df.iloc[0]["exit_type"] == "TP"


def test_log_exit_time_stop(tmp_path, main_ctx):
    path = tmp_path / "trades.csv"
    symbol = "ADA/USDT"
    open_trade_ctx = main_ctx
    open_trade_ctx.clear()
    open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "trade_id": "t4",
        "exit_type_hint": "TIME",
    }
    order = {"type": "MARKET", "avgPrice": 99.0}
    assert log_exit_from_order(symbol, order, 0.0006, str(path))
    df = read_csv_safe(path)
    assert df.iloc[0]["exit_type"] == "TIME"


def test_sweep_idempotent(tmp_path, main_ctx):
    path = tmp_path / "trades.csv"
    symbol = "ETH/USDT"
    open_trade_ctx = main_ctx
    open_trade_ctx.clear()
    _processed_order_ids.clear()
    open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }
    order = {"id": "1", "type": "STOP_MARKET", "avgPrice": 95.0}

    def sweep_once():
        oid = str(order.get("id"))
        if oid in _processed_order_ids:
            return False
        commission = _extract_commission(order)
        handled = log_exit_from_order(symbol, order, commission, str(path))
        if handled:
            _processed_order_ids.add(oid)
        return handled

    assert sweep_once()
    assert not sweep_once()
    df = read_csv_safe(path)
    assert len(df) == 1

