import importlib
import sys
import types

import pytest


def _make_dummy_exchange():
    class DummyExchange:
        id = "bybit"

        def price_to_precision(self, symbol, price):
            return price

        def amount_to_precision(self, symbol, amount):
            return amount

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_open_orders(self, *args, **kwargs):
            return []

    return DummyExchange()


@pytest.fixture
def main_module(monkeypatch):
    dummy_adapter_module = types.ModuleType("exchange_adapter")

    class DummyExchangeAdapter:
        def __init__(self, *_, **kwargs):
            config = dict(kwargs.get("config") or {})
            self.config = config
            self.sandbox = bool(config.get("sandbox"))
            self.futures = bool(config.get("futures"))
            self.exchange_id = str(config.get("exchange_id", "bybit"))
            self.x = _make_dummy_exchange()
            self._markets = {
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "BNB/USDT",
            }

        def load_markets(self, *_args, **_kwargs):
            return self._markets

        def load_markets_safe(self, *_args, **_kwargs):
            return self._markets

    dummy_adapter_module.ExchangeAdapter = DummyExchangeAdapter
    dummy_adapter_module.AdapterOHLCVUnavailable = type(
        "AdapterOHLCVUnavailable", (Exception,), {}
    )
    dummy_adapter_module.set_valid_leverage = lambda *args, **kwargs: None
    dummy_adapter_module.safe_fetch_closed_orders = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "exchange_adapter", dummy_adapter_module)
    # ensure ``main`` is reloaded with the dummy adapter module
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "exchange", _make_dummy_exchange(), raising=False)
    return main


def test_place_protected_exit_spot_omits_trigger_direction(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()
    monkeypatch.setattr(main, "exchange", exchange, raising=False)
    monkeypatch.setattr(main, "adjust_price_to_percent_filter", lambda _symbol, price: price)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "spot")

    captured_params = {}

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_params.update(params or {})
        return "order-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    result = main.place_protected_exit(
        "BTC/USDT",
        "STOP_MARKET",
        "sell",
        1.0,
        100.0,
    )

    assert result == "order-id"
    assert "triggerDirection" not in captured_params
    assert "triggerBy" not in captured_params


def test_place_protected_exit_derivative_sets_trigger_direction(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()
    monkeypatch.setattr(main, "exchange", exchange, raising=False)
    monkeypatch.setattr(main, "adjust_price_to_percent_filter", lambda _symbol, price: price)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "linear")

    captured_params = {}

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_params.update(params or {})
        return "order-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    result = main.place_protected_exit(
        "BTC/USDT",
        "STOP_MARKET",
        "sell",
        1.0,
        100.0,
    )

    assert result == "order-id"
    assert captured_params["triggerDirection"] == 2
    assert captured_params["triggerBy"] == "LastPrice"


def test_ensure_exit_orders_spot_long_omits_trigger_direction(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "spot")

    captured_calls = []

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_calls.append({"order_kind": order_kind, "params": dict(params or {})})
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert captured_calls, "Expected at least one safe_create_order call"
    for call in captured_calls:
        params = call["params"]
        assert "triggerDirection" not in params
        assert "triggerBy" not in params


def test_ensure_exit_orders_derivative_long_sets_trigger_direction(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "linear")

    captured_calls = []

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_calls.append({"order_kind": order_kind, "params": dict(params or {})})
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert captured_calls, "Expected at least one safe_create_order call"
    for call in captured_calls:
        params = call["params"]
        assert params["triggerBy"] == "LastPrice"
        if call["order_kind"] == "STOP_MARKET":
            assert params["triggerDirection"] == 2
        elif call["order_kind"] == "TAKE_PROFIT_MARKET":
            assert params["triggerDirection"] == 1
