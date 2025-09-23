"""Tests for asset_scanner using a dummy adapter."""

import asyncio
import types
import sys
import pytest

import ccxt_stub as ccxt
import types
import sys

import asset_scanner

sys.modules["ccxt"] = ccxt
sys.modules.pop("asset_scanner", None)


class DummyExchangeSync:
    def fetch_markets(self):
        return [
            {
                "symbol": "AAA/USDT",
                "quote": "USDT",
                "base": "AAA",
                "contract": True,
                "linear": True,
                "info": {"status": "TRADING", "quoteVolume": "200000"},
            },
            {
                "symbol": "BBB/USDT",
                "quote": "USDT",
                "base": "BBBUP",
                "contract": True,
                "linear": True,
                "info": {"status": "TRADING", "quoteVolume": "500000"},
            },
            {
                "symbol": "CCC/USDT",
                "quote": "USDT",
                "base": "CCC",
                "contract": True,
                "linear": True,
                "info": {"status": "BREAK", "quoteVolume": "150000"},
            },
            {
                "symbol": "DDD/USDT",
                "quote": "USDT",
                "base": "DDD",
                "contract": True,
                "linear": True,
                "info": {"status": "TRADING", "quoteVolume": "90000"},
            },
        ]

    def load_markets(self):
        markets = self.fetch_markets()
        return {m["symbol"]: m for m in markets}

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
        prices = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        return [[i, 0, 0, 0, prices[i - 1], 0] for i in range(1, 11)]

    def fetch_ticker(self, symbol):
        return {"quoteVolume": "200000", "last": "1"}

    def close(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode


class DummyAdapter:
    def __init__(self):
        self.exchange = DummyExchangeSync()
        self.config = {}
        self.sandbox = False

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


dummy_main = types.SimpleNamespace(ADAPTER=DummyAdapter())

from asset_scanner import scan_markets, scan_symbols  # pylint: disable=wrong-import-position


class DummyExchange(DummyExchangeSync):
    async def fetch_markets(self):
        return super().fetch_markets()

    async def load_markets(self):
        return super().load_markets()

    async def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
        return super().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def fetch_ticker(self, symbol):
        return super().fetch_ticker(symbol)

    async def close(self):
        pass


def test_scan_markets(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "main", dummy_main)
    monkeypatch.setattr(ccxt, "bybit", lambda params=None: DummyExchangeSync())
    asset_scanner.ADAPTER = None
    asset_scanner.SKIPPED_SYMBOLS.clear()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    symbols = scan_markets(volume_threshold=100000, limit=10)
    assert symbols == ["AAA/USDT"]
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("scan | AAA/USDT" in msg for msg in messages)


def test_scan_markets_uses_timeframe_fallback(monkeypatch, caplog):
    import importlib
    import sys

    sys.modules.pop("asset_scanner", None)
    scanner_mod = importlib.import_module("asset_scanner")

    def run_scan_markets(*args, **kwargs):
        return scanner_mod.scan_markets(*args, **kwargs)

    class MissingDailyAdapter(DummyAdapter):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
            self.calls.append(timeframe)
            if timeframe == "1d":
                raise RuntimeError("unsupported timeframe")
            return super().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    adapter = MissingDailyAdapter()
    monkeypatch.setitem(
        sys.modules,
        "main",
        types.SimpleNamespace(ADAPTER=adapter),
    )
    monkeypatch.setattr(ccxt, "bybit", lambda params=None: DummyExchangeSync())
    scanner_mod.ADAPTER = adapter
    monkeypatch.setattr(scanner_mod, "_get_adapter", lambda: adapter)
    scanner_mod.SKIPPED_SYMBOLS.clear()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    symbols = run_scan_markets(volume_threshold=100000, limit=10)
    assert symbols == ["AAA/USDT"]
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("falling back to 4h" in msg for msg in messages), f"messages={messages!r} calls={adapter.calls!r}"
    assert "1d" in adapter.calls
    assert any(tf != "1d" for tf in adapter.calls)


def test_scan_symbols(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "main", dummy_main)
    monkeypatch.setattr(ccxt, "bybit", lambda *args, **kwargs: DummyExchangeSync())
    try:
        import ccxt.pro as ccxtpro  # type: ignore
        monkeypatch.setattr(ccxtpro, "bybit", lambda *args, **kwargs: DummyExchange())
    except Exception:
        pass
    asset_scanner.ADAPTER = None
    asset_scanner.SKIPPED_SYMBOLS.clear()
    caplog.set_level("INFO")
    symbols = asyncio.run(scan_symbols(min_volume=100000, limit=10))
    assert "AAA/USDT" in symbols
    logged = caplog.text
    assert "ROI=" in logged


def test_scan_symbols_top_n(monkeypatch):
    monkeypatch.setitem(sys.modules, "main", dummy_main)
    class ExchangeA(DummyExchangeSync):
        def fetch_markets(self):
            return [
                {
                    "symbol": "AAA/USDT",
                    "quote": "USDT",
                    "base": "AAA",
                    "contract": True,
                    "linear": True,
                    "info": {"status": "TRADING", "quoteVolume": "200000"},
                },
                {
                    "symbol": "EEE/USDT",
                    "quote": "USDT",
                    "base": "EEE",
                    "contract": True,
                    "linear": True,
                    "info": {"status": "TRADING", "quoteVolume": "200000"},
                },
            ]

        def load_markets(self):
            markets = self.fetch_markets()
            return {m["symbol"]: m for m in markets}

        def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
            if symbol == "AAA/USDT":
                prices = [1 + i for i in range(10)]
            else:
                prices = [1 + 0.1 * i for i in range(10)]
            return [[i + 1, 0, 0, 0, prices[i], 0] for i in range(10)]

    class ExchangeAAsync(ExchangeA):
        async def fetch_markets(self):
            return super().fetch_markets()

        async def load_markets(self):
            return super().load_markets()

        async def fetch_ohlcv(self, symbol, timeframe="1d", limit=90):
            return super().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        async def fetch_ticker(self, symbol):
            return {"quoteVolume": "200000", "last": "1"}

        async def close(self):
            pass

    monkeypatch.setattr(ccxt, "bybit", lambda *args, **kwargs: ExchangeA())
    try:
        import ccxt.pro as ccxtpro  # type: ignore

        monkeypatch.setattr(ccxtpro, "bybit", lambda *args, **kwargs: ExchangeAAsync())
    except Exception:
        pass
    asset_scanner.ADAPTER = None
    asset_scanner.SKIPPED_SYMBOLS.clear()
    symbols = asyncio.run(scan_symbols(min_volume=100000, limit=10, top_n=1))
    assert symbols == ["AAA/USDT"]
