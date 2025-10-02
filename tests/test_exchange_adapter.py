import csv
import types
import pytest

import exchange_adapter
from exchange_adapter import ExchangeAdapter, AdapterOHLCVUnavailable


def test_ccxt_activation_and_markets(monkeypatch):
    class DummyBybit:
        def __init__(self, params=None):
            self.has = {"fetchOHLCV": True}
            
        def set_sandbox_mode(self, mode):
            self.sandbox = mode

        def load_markets(self, reload=True, /):
            return {"BTC/USDT": {}}

        def fetch_ohlcv(self, symbol, timeframe, limit):
            return [[1, 1, 1, 1, 1, 1]]

    mod = types.SimpleNamespace(bybit=DummyBybit)
    monkeypatch.setattr(exchange_adapter, "_ccxt", mod)

    calls = {"cnt": 0}
    original = ExchangeAdapter.load_markets_safe

    def wrapper(self):
        calls["cnt"] += 1
        return original(self)

    monkeypatch.setattr(ExchangeAdapter, "load_markets_safe", wrapper)

    ad = ExchangeAdapter(config={"EXCHANGE_BACKEND": "ccxt", "exchange_id": "bybit"})
    assert calls["cnt"] == 1  # called during __init__
    assert ad.backend == "ccxt"
    assert ad.ccxt_id == "bybit"
    assert getattr(ad.x, "markets", {})
    assert ad.fetch_ohlcv("BTC/USDT", "1m") == [[1, 1, 1, 1, 1, 1]]


def test_ccxt_fallback_when_fetch_missing(monkeypatch):
    class Bad:
        def __init__(self, params=None):
            self.has = {}

        def set_sandbox_mode(self, mode):
            pass

        def load_markets(self, reload=True, /):
            return {"BTC/USDT": {}}

    class Good:
        def __init__(self, params=None):
            self.has = {"fetchOHLCV": True}

        def set_sandbox_mode(self, mode):
            pass

        def load_markets(self, reload=True, /):
            return {"BTC/USDT": {}}

        def fetch_ohlcv(self, symbol, timeframe, limit):
            return [[1]]

    mod = types.SimpleNamespace(bybit=Bad, binance=Good)
    monkeypatch.setattr(exchange_adapter, "_ccxt", mod)

    with pytest.raises(exchange_adapter.AdapterInitError):
        ExchangeAdapter(config={"EXCHANGE_BACKEND": "ccxt", "exchange_id": "bybit"})


def test_multi_ohlcv_partial_fail():
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.backend = "ccxt"
    ad.last_warn_at = {}

    def fake_fetch(self, symbol, tf, limit=500):
        if tf == "1m":
            return [[1, 1, 1, 1, 1, 1]]
        raise AdapterOHLCVUnavailable("boom")

    ad.fetch_ohlcv = types.MethodType(fake_fetch, ad)

    data = ad.fetch_multi_ohlcv("BTC/USDT", ["1m", "5m"], limit=10)
    assert "1m" in data and "5m" not in data


def test_fetch_ohlcv_argument_checks():
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.x = types.SimpleNamespace(
        fetch_ohlcv=lambda *a, **k: [[1]],
        markets={"BTC/USDT": {}},
        last_request_url="",
        last_http_status_code=200,
    )
    ad.sandbox = False
    ad.load_markets_safe = lambda: True
    with pytest.raises(AdapterOHLCVUnavailable):
        ad.fetch_ohlcv("", "1m")
    with pytest.raises(AdapterOHLCVUnavailable):
        ad.fetch_ohlcv("BTC/USDT", "")


def test_fetch_ohlcv_csv_fallback(tmp_path):
    path = tmp_path / "BTCUSDT_1m.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0, 1, 2, 3, 4, 5])

    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.config = {"csv_dir": str(tmp_path)}
    ad.sandbox = False
    ad.x = types.SimpleNamespace(
        fetch_ohlcv=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        markets={"BTC/USDT": {}},
        last_request_url="",
        last_http_status_code=0,
    )
    ad.load_markets_safe = lambda: True

    data = ad.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    assert data == [[0, 1.0, 2.0, 3.0, 4.0, 5.0]]


def test_exchange_options_match(monkeypatch):
    class DummyBybit:
        def __init__(self, params=None):
            self.has = {"fetchOHLCV": True}
            self.options = params.get("options", {})

        def set_sandbox_mode(self, mode):
            self.sandbox = mode

        def load_markets(self, reload=True, /):
            return {"BTC/USDT": {}}

        def fetch_ohlcv(self, symbol, timeframe, limit):
            return [[1]]

    mod = types.SimpleNamespace(bybit=DummyBybit)
    monkeypatch.setattr(exchange_adapter, "_ccxt", mod)

    ad = ExchangeAdapter(config={"EXCHANGE_BACKEND": "ccxt", "futures": True, "exchange_id": "bybit"})
    assert ad.x.options.get("defaultType") == "swap"
    assert ad.x.options.get("defaultSubType") in (None, "linear", "inverse")


def test_fetch_open_orders_normalizes_linear_symbol():
    class DummyExchange:
        def __init__(self):
            self.last_call = None
            self.markets = {
                "XXX/USDT:USDT": {
                    "symbol": "XXX/USDT:USDT",
                    "info": {"category": "linear"},
                    "type": "swap",
                    "linear": True,
                }
            }

        def market(self, symbol):
            if symbol == "XXX/USDT":
                return self.markets["XXX/USDT:USDT"]
            return self.markets.get(symbol)

        def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
            self.last_call = (symbol, params or {})
            return []

    adapter = ExchangeAdapter.__new__(ExchangeAdapter)
    adapter.x = DummyExchange()
    adapter.config = {"exchange_id": "bybit"}
    adapter.exchange_id = "bybit"
    adapter.futures = True
    adapter.sandbox = False

    count, ids = adapter.fetch_open_orders("XXX/USDT")

    assert (count, ids) == (0, [])
    assert adapter.x.last_call is not None
    symbol_used, params_used = adapter.x.last_call
    assert symbol_used == "XXX/USDT:USDT"
    assert params_used.get("category") == "linear"

