import builtins
import csv
import importlib
import sys
from typing import Any

import pytest
from logging_utils import (
    safe_create_order,
    place_conditional_exit,
    flush_symbol_logs,
    safe_set_leverage,
    setup_logger,
    log_entry,
    _ENTRY_CACHE,
    ensure_trades_csv_header,
    ensure_report_schema,
    TRADES_CSV_HEADER,
    LOG_ENTRY_FIELDS,
    detect_market_category,
    _normalize_bybit_symbol,
    _with_bybit_order_params,
    _market_category_from_meta,
)

class DummyExchange:
    def create_order(self, *args, **kwargs):
        return {"id": "1", "status": "filled"}
    def set_leverage(self, leverage, symbol):
        return True

class FailExchange:
    def create_order(self, *args, **kwargs):
        raise RuntimeError("fail")
    def set_leverage(self, leverage, symbol):
        raise RuntimeError("bad")


@pytest.fixture
def bybit_spot_and_linear_exchange():
    class Exchange:
        id = "bybit"
        futures = True

        def __init__(self):
            self.markets = {
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "base": "ETH",
                    "quote": "USDT",
                    "spot": True,
                    "info": {"category": "spot"},
                },
                "ETH/USDT:USDT": {
                    "symbol": "ETH/USDT:USDT",
                    "base": "ETH",
                    "quote": "USDT",
                    "linear": True,
                    "info": {"category": "linear"},
                },
            }
            self.markets_by_id = {}

        def market(self, symbol):
            meta = self.markets.get(symbol)
            if meta is None:
                raise KeyError(symbol)
            return meta

    return Exchange()


def test_safe_create_order_success(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    order_id, err = safe_create_order(DummyExchange(), "BTC/USDT", "market", "buy", 1)
    assert order_id == "1"
    assert err is None
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "Order BUY" in logged


def test_safe_create_order_error(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    order_id, err = safe_create_order(FailExchange(), "ETH/USDT", "market", "buy", 1)
    assert order_id is None
    assert err is not None
    flush_symbol_logs("ETH/USDT")
    logged = caplog.text
    assert "Order failed" in logged


def test_safe_set_leverage(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    assert safe_set_leverage(DummyExchange(), "LTC/USDT", 20)
    flush_symbol_logs("LTC/USDT")
    logged = caplog.text
    assert "Leverage set to 20x" in logged


def test_safe_set_leverage_error(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    assert not safe_set_leverage(FailExchange(), "DOGE/USDT", 20)
    flush_symbol_logs("DOGE/USDT")
    logged = caplog.text
    assert "Failed to set leverage" in logged


def test_log_entry_idempotent_and_defaults(tmp_path):
    _ENTRY_CACHE.clear()
    log_file = tmp_path / "entries.csv"
    ctx = {
        "entry_price": 100.0,
        "qty": 1.0,
        "reason": "test",
        "side": "LONG",
        "sl_price": 95.0,
        "tp_price": 110.0,
    }
    t1 = log_entry("BTC/USDT", ctx, str(log_file))
    t2 = log_entry("BTC/USDT", ctx, str(log_file))
    assert t1 == t2
    with open(log_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "live"
    assert row["reduced_risk"] == "False"
    assert row["trade_id"] == t1
    assert row["sl"] == "95.0"
    assert row["tp"] == "110.0"


class PercentFilterBase:
    markets = True

    def market(self, symbol):
        return {
            "info": {
                "filters": [
                    {
                        "filterType": "PERCENT_PRICE",
                        "multiplierUp": "1.05",
                        "multiplierDown": "0.95",
                    }
                ]
            }
        }

    def fetch_ticker(self, symbol):
        return {"ask": 100.0, "bid": 99.0}

    def price_to_precision(self, symbol, price):
        return str(price)


class RetryExchange(PercentFilterBase):
    def __init__(self):
        self.calls = []

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append((order_type, price))
        if len(self.calls) == 1:
            raise RuntimeError("{\"code\":-4131}")
        return {"id": "1", "status": "FILLED", "price": price, "filled": qty}


class FallbackExchange(PercentFilterBase):
    def __init__(self):
        self.calls = []

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append((order_type, price))
        if order_type == "limit":
            raise RuntimeError('{"retCode":30208}')
        return {"id": "2", "status": "FILLED", "price": price, "filled": qty}


class BybitExitExchange:
    id = "bybit"

    def __init__(self):
        self.calls = []
        self.markets = {
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "precision": {"amount": 3},
                "limits": {
                    "amount": {"min": 0.001, "step": 0.001},
                    "cost": {"min": 5.0},
                },
                "info": {
                    "filters": [
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.001",
                            "stepSize": "0.001",
                        }
                    ]
                },
            }
        }
        self.markets_by_id = self.markets

    def market(self, symbol):
        return self.markets.get(symbol, {})

    def fetch_ticker(self, symbol):
        return {"ask": 101.0, "bid": 100.0}

    def fetch_open_orders(self, symbol, params=None):
        return []

    def cancel_order(self, order_id, symbol, params=None):
        self.calls.append(("cancel", order_id, params))
        return {"id": order_id, "status": "canceled"}

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.3f}"

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append((order_type, side, qty, price, params))
        return {"id": "3", "status": "FILLED", "filled": qty}


class BybitSpotMetaExchange:
    id = "bybit"

    def __init__(self):
        self.markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "spot": True,
                "info": {"category": "spot"},
            }
        }
        self.markets_by_id = {}

    def market(self, symbol):
        return self.markets.get(symbol)

    def load_markets(self):
        return self.markets


class BybitLinearMetaExchange:
    id = "bybit"

    def __init__(self):
        self.markets = {
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "linear": True,
                "info": {"category": "linear"},
            }
        }
        self.markets_by_id = {}

    def market(self, symbol):
        return self.markets.get(symbol)

    def load_markets(self):
        return self.markets


class LazyLoadBybitLinearExchange:
    id = "bybit"

    def __init__(self):
        self.markets = {}
        self.markets_by_id = {}
        self.calls = []
        self.load_calls = 0

    def load_markets(self):
        self.load_calls += 1
        market = {
            "symbol": "ETH/USDT:USDT",
            "precision": {"amount": 3},
            "limits": {
                "amount": {"min": 0.001, "step": 0.001},
                "cost": {"min": 5.0},
            },
            "info": {
                "filters": [
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.001",
                        "stepSize": "0.001",
                    }
                ],
                "category": "linear",
            },
        }
        self.markets = {market["symbol"]: market}
        self.markets_by_id = self.markets
        return self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def fetch_ticker(self, symbol):
        return {"ask": 101.0, "bid": 100.0}

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.3f}"

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append(
            {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params,
            }
        )
        return {"id": "42", "status": "FILLED", "filled": qty}


def test_with_bybit_order_params_spot():
    exchange = BybitSpotMetaExchange()
    params, category = _with_bybit_order_params(exchange, "BTC/USDT", {})
    assert category == "spot"
    assert params == {"category": "spot"}


def test_with_bybit_order_params_linear_defaults_position_idx():
    exchange = BybitLinearMetaExchange()
    params, category = _with_bybit_order_params(exchange, "ETH/USDT", {})
    assert category == "linear"
    assert params["category"] == "linear"
    assert params["positionIdx"] == 0


def test_with_bybit_order_params_normalizes_swap_category(monkeypatch):
    class _SwapExchange:
        id = "bybit"
        markets = {}

    exchange = _SwapExchange()

    monkeypatch.setattr(
        "logging_utils.detect_market_category", lambda *_args, **_kwargs: "swap"
    )

    params, resolved = _with_bybit_order_params(exchange, "ETH/USDT", {})

    assert resolved == "linear"
    assert params.get("category") == "linear"
    assert all(
        not (isinstance(value, str) and "swap" in value.lower())
        for value in params.values()
    )


def test_with_bybit_order_params_sets_tp_sl_mode_for_exit_hints(monkeypatch):
    class _HintExchange:
        id = "bybit"
        markets = {}

    exchange = _HintExchange()

    monkeypatch.setattr(
        "logging_utils.detect_market_category", lambda *_args, **_kwargs: "spot"
    )

    params, resolved = _with_bybit_order_params(
        exchange,
        "ETH/USDT",
        {
            "reduceOnly": True,
            "closeOnTrigger": True,
            "slOrderType": "Market",
        },
    )

    assert resolved == "linear"
    assert params["category"] == "linear"
    assert params["tpSlMode"] == "Full"


class FakeBybitLinearExchange:
    id = "bybit"

    def __init__(self):
        self.markets = {
            "ETH/USDT:USDT": {
                "symbol": "ETH/USDT:USDT",
                "id": "ETHUSDT",
                "info": {"category": "linear"},
                "linear": True,
            }
        }
        self.markets_by_id = self.markets
        self.params = {"category": "linear"}

    def market(self, symbol):
        result = self.markets.get(symbol)
        if result is None:
            raise KeyError(symbol)
        return result


class _StubBybitExitExchange:
    id = "bybit"

    def __init__(self):
        self.markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "base": "BTC",
                "quote": "USDT",
                "info": {"category": "linear", "filters": []},
            }
        }
        self.markets_by_id = self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def fetch_open_orders(self, symbol, *args, **kwargs):
        return []

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.3f}"


def _import_main(monkeypatch):
    class _AdapterStub:
        def __init__(self, *args, **kwargs):
            self.x = _StubBybitExitExchange()

        def load_markets(self):  # pragma: no cover - trivial stub
            return None

    monkeypatch.setattr("exchange_adapter.ExchangeAdapter", _AdapterStub)
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_safe_create_order_percent_filter_retry(caplog):
    setup_logger()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    ex = RetryExchange()
    order_id, err = safe_create_order(ex, "BTC/USDT", "limit", "buy", 1, 100)
    assert order_id == "1"
    assert err is None
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "order_price_band_retry" in logged
    assert ex.calls[0][0] == "limit"
    assert ex.calls[1][0] == "market"


def test_safe_create_order_percent_filter_market_fallback(caplog):
    setup_logger()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    ex = FallbackExchange()
    order_id, err = safe_create_order(ex, "BTC/USDT", "limit", "buy", 1, 100)
    assert order_id == "2"
    assert err is None
    assert ex.calls[-1][0] == "market"
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "order_market_fallback" in logged


def test_place_conditional_exit_normalized(caplog):
    setup_logger()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    ex = BybitExitExchange()
    order_id, err = place_conditional_exit(
        ex,
        "ETH/USDT",
        "buy",
        1000.0,
        0.02,
        is_tp=False,
    )
    assert order_id == "3"
    assert err is None
    assert ex.calls, "Expected create_order to be invoked"
    otype, side, qty, price, params = ex.calls[-1]
    assert otype == "market"
    assert side == "sell"
    assert qty is None
    assert price is None
    assert params["tpSlMode"] == "Full"
    assert params["triggerBy"] == "LastPrice"
    assert params["orderType"] == "Market"
    assert params["triggerDirection"] == "descending"


def test_safe_create_order_loads_markets_before_symbol_normalization():
    exchange = LazyLoadBybitLinearExchange()
    params = {"category": "linear"}

    order_id, err = safe_create_order(
        exchange, "ETH/USDT", "limit", "buy", 1.0, 100.0, params
    )

    assert order_id == "42"
    assert err is None
    assert exchange.load_calls == 1
    assert exchange.calls, "Expected create_order to be invoked"
    last_call = exchange.calls[-1]
    assert last_call["symbol"] == "ETH/USDT:USDT"
    assert last_call["params"]["category"] == "linear"


def test_normalize_bybit_symbol_linear_contract_fallback():
    class DummyExchange:
        id = "bybit"

        def __init__(self):
            self.markets = {}
            self.markets_by_id = {}

    exchange = DummyExchange()
    normalized = _normalize_bybit_symbol(exchange, "ETH/USDT", "linear")
    assert normalized == "ETH/USDT:USDT"


def test_detect_market_category_linear_mapping():
    exchange = FakeBybitLinearExchange()
    assert detect_market_category(exchange, "ETH/USDT") == "linear"


def test_detect_market_category_prefers_linear_with_futures_hint(bybit_spot_and_linear_exchange):
    exchange = bybit_spot_and_linear_exchange
    assert detect_market_category(exchange, "ETH/USDT") == "linear"


def test_with_bybit_order_params_infers_linear_category(bybit_spot_and_linear_exchange):
    exchange = bybit_spot_and_linear_exchange
    params, resolved = _with_bybit_order_params(exchange, "ETH/USDT", {})
    assert resolved == "linear"
    assert params.get("category") == "linear"


def test_normalize_bybit_category_aliases():
    meta = {
        "symbol": "ETH/USDT:USDT",
        "info": {"contractType": "LinearPerpetual"},
        "linear": True,
    }
    assert _market_category_from_meta(meta) == "linear"


def test_detect_market_category_loads_markets_when_missing():
    class LazyExchange:
        id = "bybit"

        def __init__(self):
            self.markets = {}
            self.markets_by_id = {}
            self.load_calls = 0

        def load_markets(self):
            self.load_calls += 1
            meta = {
                "symbol": "ETH/USDT:USDT",
                "info": {"category": "linear"},
                "base": "ETH",
                "quote": "USDT",
                "linear": True,
            }
            self.markets = {"ETH/USDT:USDT": meta}
            self.markets_by_id = {"ETHUSDT": meta}
            return self.markets

        def market(self, symbol):
            if symbol not in self.markets:
                raise KeyError(symbol)
            return self.markets[symbol]

    exchange = LazyExchange()
    assert detect_market_category(exchange, "ETH/USDT") == "linear"
    assert exchange.load_calls == 1


def test_ensure_trades_csv_header_migrates(tmp_path):
    path = tmp_path / "trades.csv"
    # simulate old header missing ``trade_id`` as the first column
    old_header = LOG_ENTRY_FIELDS[1:]
    with open(path, "w", newline="") as f:
        f.write(",".join(old_header) + "\n")
        f.write(",".join(str(i) for i in range(len(old_header))) + "\n")

    ensure_trades_csv_header(str(path))
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == TRADES_CSV_HEADER

    backup = path.with_name("trades_misaligned.csv")
    assert backup.exists()
    with open(backup) as f:
        legacy_header = f.readline().strip().split(",")
    assert legacy_header == old_header


def test_ensure_trades_csv_header_creates(tmp_path):
    path = tmp_path / "trades.csv"
    ensure_trades_csv_header(str(path))
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == TRADES_CSV_HEADER


def test_ensure_report_schema_resets(tmp_path):
    path = tmp_path / "pair_report.csv"
    with open(path, "w", newline="") as f:
        f.write("foo,bar\n1,2\n")

    ensure_report_schema(str(path), ["symbol", "winrate"])

    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == ["symbol", "winrate"]

    backup_new = tmp_path / "pair_report_misaligned.csv"
    backup_legacy = tmp_path / "pair_report_legacy.csv"
    assert backup_new.exists() or backup_legacy.exists()


def test_setup_logger_writes_file(tmp_path):
    log_path = tmp_path / "run.log"
    setup_logger(str(log_path))
    import logging

    logging.info("hello world")
    with open(log_path) as f:
        data = f.read()
    assert "hello world" in data


def test_setup_logger_creates_directory(tmp_path):
    log_path = tmp_path / "nested" / "run.log"
    setup_logger(str(log_path))
    assert log_path.exists()


def test_ensure_exit_orders_trigger_direction_long(monkeypatch):
    main = _import_main(monkeypatch)
    exchange = _StubBybitExitExchange()

    class Adapter:
        client = exchange

    recorded: list[dict[str, Any]] = []

    def _recording_place_conditional_exit(
        _exchange, symbol, side_open, base_price, pct, *, is_tp
    ):
        recorded.append(
            {
                "symbol": symbol,
                "side_open": side_open,
                "base_price": base_price,
                "pct": pct,
                "is_tp": is_tp,
            }
        )
        return ("tp" if is_tp else "sl"), None

    monkeypatch.setattr(main, "place_conditional_exit", _recording_place_conditional_exit)
    main.open_trade_ctx.pop("BTC/USDT", None)
    main.ensure_exit_orders(Adapter(), "BTC/USDT", "long", 1.0, 99.0, 101.0)

    assert recorded, "Expected conditional exits"
    assert {call["is_tp"] for call in recorded} == {False, True}
    assert all(call["side_open"] == "buy" for call in recorded)


def test_ensure_exit_orders_trigger_direction_short(monkeypatch):
    main = _import_main(monkeypatch)
    exchange = _StubBybitExitExchange()

    class Adapter:
        client = exchange

    recorded: list[dict[str, Any]] = []

    def _recording_place_conditional_exit(
        _exchange, symbol, side_open, base_price, pct, *, is_tp
    ):
        recorded.append(
            {
                "symbol": symbol,
                "side_open": side_open,
                "base_price": base_price,
                "pct": pct,
                "is_tp": is_tp,
            }
        )
        return ("tp" if is_tp else "sl"), None

    monkeypatch.setattr(main, "place_conditional_exit", _recording_place_conditional_exit)
    main.open_trade_ctx.pop("BTC/USDT", None)
    main.ensure_exit_orders(Adapter(), "BTC/USDT", "short", 1.0, 101.0, 99.0)

    assert recorded, "Expected conditional exits"
    assert {call["is_tp"] for call in recorded} == {False, True}
    assert all(call["side_open"] == "sell" for call in recorded)
