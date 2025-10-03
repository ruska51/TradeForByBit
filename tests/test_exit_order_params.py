import importlib
import logging
import sys
import types

import pytest

from logging_utils import detect_market_category, _with_bybit_order_params


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


def test_bybit_spot_category_respected_with_futures_adapter():
    class DummyBybitExchange:
        id = "bybit"

        def __init__(self):
            self.futures = True
            self.adapter = types.SimpleNamespace(futures=True)
            self.markets = {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "base": "BTC",
                    "quote": "USDT",
                    "spot": True,
                    "info": {"category": "spot"},
                }
            }

        def market(self, symbol):
            return self.markets.get(symbol)

    exchange = DummyBybitExchange()

    assert detect_market_category(exchange, "BTC/USDT") == "spot"

    params, category = _with_bybit_order_params(exchange, "BTC/USDT", {})

    assert category == "spot"
    assert params["category"] == "spot"
    assert "positionIdx" not in params


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
    assert "tpSlMode" not in captured_params


@pytest.mark.parametrize(
    "side, order_type, expected_direction",
    [
        ("sell", "STOP_MARKET", "falling"),
        ("sell", "TAKE_PROFIT_MARKET", "rising"),
        ("buy", "STOP_MARKET", "rising"),
        ("buy", "TAKE_PROFIT_MARKET", "falling"),
    ],
)
def test_place_protected_exit_derivative_sets_trigger_direction(
    monkeypatch, main_module, side, order_type, expected_direction
):
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
        order_type,
        side,
        1.0,
        100.0,
    )

    assert result == "order-id"
    assert (
        captured_params["triggerDirection"]
        == main.BYBIT_TRIGGER_DIRECTIONS[expected_direction]
    )
    assert captured_params["triggerBy"] == "LastPrice"
    assert captured_params["tpSlMode"] == "Full"


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


def test_get_max_position_qty_spot_uses_amount_limit(monkeypatch, main_module):
    main = main_module

    class SpotExchange:
        id = "binance"

        def __init__(self):
            self.fetch_leverage_tiers_called = False
            self.leverage_bracket_called = False
            self.markets = {
                "BTC/USDT": {
                    "id": "BTCUSDT",
                    "symbol": "BTC/USDT",
                    "spot": True,
                    "limits": {"amount": {"max": 5.0}},
                }
            }

        def market(self, symbol):
            return self.markets[symbol]

        def amount_to_precision(self, symbol, amount):
            return amount

        def fetch_leverage_tiers(self, symbols):
            self.fetch_leverage_tiers_called = True
            return {}

        def fapiPrivate_get_leverageBracket(self, params):
            self.leverage_bracket_called = True
            return []

    exchange = SpotExchange()

    monkeypatch.setattr(main, "exchange", exchange, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "spot")

    qty = main.get_max_position_qty("BTC/USDT", leverage=2, price=100.0)

    assert qty == 5.0
    assert not exchange.fetch_leverage_tiers_called
    assert not exchange.leverage_bracket_called


def test_ensure_exit_orders_derivative_long_sets_trigger_direction(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {"BTC/USDT": {"entry_price": 100.0}}, raising=False)
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
        assert params["triggerPrice"] in {95.0, 105.0}
        assert params["closeOnTrigger"] is True
        assert params["closePosition"] is True
        assert params["reduceOnly"] is True
        assert params["orderType"] == "Market"
        assert params["category"] == "linear"
        if call["order_kind"] == "STOP_MARKET":
            assert (
                params["triggerDirection"]
                == main.BYBIT_TRIGGER_DIRECTIONS["falling"]
            )
            assert params["slTriggerBy"] == "LastPrice"
            assert params["slOrderType"] == "Market"
            assert params["stopPrice"] == params["triggerPrice"]
            assert params["tpSlMode"] == "Full"
        elif call["order_kind"] == "TAKE_PROFIT_MARKET":
            assert (
                params["triggerDirection"]
                == main.BYBIT_TRIGGER_DIRECTIONS["rising"]
            )
            assert params["tpTriggerBy"] == "LastPrice"
            assert params["tpOrderType"] == "Market"
            assert params["stopPrice"] == params["triggerPrice"]
            assert params["priceProtect"] is True
            assert params["tpSlMode"] == "Full"


def test_ensure_exit_orders_derivative_short_sets_trigger_direction(monkeypatch, main_module):
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
        "short",
        1.0,
        sl_price=105.0,
        tp_price=95.0,
    )

    assert captured_calls, "Expected at least one safe_create_order call"
    for call in captured_calls:
        params = call["params"]
        assert params["triggerBy"] == "LastPrice"
        if call["order_kind"] == "STOP_MARKET":
            assert (
                params["triggerDirection"]
                == main.BYBIT_TRIGGER_DIRECTIONS["rising"]
            )
            assert params["tpSlMode"] == "Full"
        elif call["order_kind"] == "TAKE_PROFIT_MARKET":
            assert (
                params["triggerDirection"]
                == main.BYBIT_TRIGGER_DIRECTIONS["falling"]
            )
            assert params["tpSlMode"] == "Full"


def test_ensure_exit_orders_adjusts_trigger_direction_to_price(monkeypatch, main_module):
    main = main_module

    exchange = _make_dummy_exchange()

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(
        main,
        "open_trade_ctx",
        {"BTC/USDT": {"entry_price": 100.0}},
        raising=False,
    )
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
        sl_price=105.0,
        tp_price=None,
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["order_kind"] == "STOP_MARKET"
    params = captured_calls[0]["params"]
    assert params["triggerPrice"] == 105.0
    assert (
        params["triggerDirection"]
        == main.BYBIT_TRIGGER_DIRECTIONS["rising"]
    )
    assert params["tpSlMode"] == "Full"


def test_bybit_dual_market_prefers_linear_for_futures(monkeypatch, main_module):
    main = main_module

    class DualMarketExchange:
        id = "bybit"

        def __init__(self):
            self.options = {"defaultType": "linear"}
            self.params = {"category": "linear"}
            self.futures = True
            self._fetch_calls = []
            self.markets = {
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "id": "ETHUSDT",
                    "base": "ETH",
                    "quote": "USDT",
                    "spot": True,
                    "info": {"category": "spot"},
                },
                "ETH/USDT:USDT": {
                    "symbol": "ETH/USDT:USDT",
                    "id": "ETHUSDT",
                    "base": "ETH",
                    "quote": "USDT",
                    "linear": True,
                    "swap": True,
                    "settle": "USDT",
                    "info": {"category": "linear"},
                },
            }
            self.markets_by_id = {
                "ETHUSDT": self.markets["ETH/USDT:USDT"],
            }
            self.adapter = types.SimpleNamespace(
                futures=True,
                params={"category": "linear"},
                options={"defaultType": "linear"},
            )

        def price_to_precision(self, _symbol, price):
            return price

        def amount_to_precision(self, _symbol, amount):
            return amount

        def fetch_ticker(self, symbol):
            return {"last": 100.0, "symbol": symbol}

        def fetch_open_orders(self, *args, **kwargs):
            self._fetch_calls.append((args, kwargs))
            return []

        def market(self, symbol):
            return self.markets.get(symbol) or self.markets_by_id.get(symbol)

    exchange = DualMarketExchange()

    assert detect_market_category(exchange, "ETH/USDT") == "linear"

    monkeypatch.setattr(main, "exchange", exchange, raising=False)
    monkeypatch.setattr(main, "adjust_price_to_percent_filter", lambda _symbol, price: price)

    captured_orders = []

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_orders.append(
            {
                "order_kind": order_kind,
                "side": side,
                "symbol": symbol,
                "params": dict(params or {}),
            }
        )
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    place_result = main.place_protected_exit(
        "ETH/USDT",
        "STOP_MARKET",
        "sell",
        1.0,
        100.0,
    )

    assert place_result == "STOP_MARKET-id"
    assert captured_orders
    stop_params = captured_orders[-1]["params"]
    assert stop_params["triggerDirection"] == main.BYBIT_TRIGGER_DIRECTIONS["falling"]
    assert stop_params["triggerBy"] == "LastPrice"

    captured_orders.clear()

    adapter = types.SimpleNamespace(client=exchange)
    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)

    main.ensure_exit_orders(
        adapter,
        "ETH/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert captured_orders, "Expected exit orders for dual market futures setup"
    for call in captured_orders:
        params = call["params"]
        assert params["triggerDirection"] in main.BYBIT_TRIGGER_DIRECTIONS.values()
        assert params["triggerBy"] == "LastPrice"


def test_ensure_exit_orders_blocks_after_fetch_failure(
    monkeypatch, main_module, caplog
):
    main = main_module

    exchange = _make_dummy_exchange()
    attempts = {"count": 0}

    def flaky_fetch_open_orders(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary outage")
        return []

    monkeypatch.setattr(exchange, "fetch_open_orders", flaky_fetch_open_orders)

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "linear")
    monkeypatch.setattr(main, "exit_orders_fetch_guard", {}, raising=False)

    captured_calls = []

    def fake_safe_create_order(
        _exchange, symbol, order_kind, side, qty, price, params
    ):
        captured_calls.append(
            {
                "order_kind": order_kind,
                "params": dict(params or {}),
            }
        )
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    caplog.set_level(logging.WARNING)

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert not captured_calls
    state = main.exit_orders_fetch_guard.get("BTC/USDT")
    assert state and state["blocked"] and state["warned"]

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "fetch_open_orders failed" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert not captured_calls

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert captured_calls, "Expected orders after fetch snapshot restored"
    state = main.exit_orders_fetch_guard.get("BTC/USDT")
    assert state and not state["blocked"] and not state["warned"]
    assert attempts["count"] == 3


def test_ensure_exit_orders_loads_markets_before_normalization(
    monkeypatch, main_module, caplog
):
    main = main_module

    class DummyExchange:
        id = "bybit"

        def __init__(self):
            self.markets = {}
            self.markets_by_id = {}
            self.load_calls = 0

        def load_markets(self):
            self.load_calls += 1
            market_meta = {
                "symbol": "BTC/USDT:USDT",
                "base": "BTC",
                "quote": "USDT",
                "linear": True,
            }
            self.markets = {"BTC/USDT:USDT": market_meta}
            self.markets_by_id = {"BTCUSDT": market_meta}
            return self.markets

        def market(self, symbol):
            return self.markets.get(symbol) or self.markets_by_id.get(symbol.replace("/", ""))

        def price_to_precision(self, symbol, price):
            return price

        def amount_to_precision(self, symbol, amount):
            assert symbol == "BTC/USDT:USDT"
            return amount

        def fetch_open_orders(self, symbol, *args, **kwargs):
            if symbol != "BTC/USDT:USDT":
                logging.warning("Illegal category for %s", symbol)
            return []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    exchange = DummyExchange()
    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "linear")
    monkeypatch.setattr(main, "exit_orders_fetch_guard", {}, raising=False)

    captured_symbols: list[str] = []

    def fake_safe_create_order(
        _exchange, symbol, order_kind, side, qty, price, params
    ):
        captured_symbols.append(symbol)
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    caplog.set_level(logging.WARNING)

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert exchange.load_calls >= 1
    assert captured_symbols, "Expected safe_create_order invocations"
    assert all(symbol == "BTC/USDT:USDT" for symbol in captured_symbols)
    assert "Illegal category" not in caplog.text
