from exchange_adapter import ExchangeAdapter
from symbol_utils import filter_supported_symbols


class CCXTDummy:
    def load_markets(self, reload=False):
        self.markets = {"ETH/USDT": {}, "BTC/USDT": {}}
        return self.markets


class SDKDummy:
    def futures_exchange_info(self):  # pragma: no cover - simple
        return {"symbols": [{"symbol": "ETHUSDT"}]}


def test_market_filter_ccxt(monkeypatch):
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.backend = "ccxt"
    ad.x = CCXTDummy()
    ad.last_warn_at = {}
    markets_cache = {}
    supported, removed, degraded = filter_supported_symbols(
        ad, ["ETH/USDT", "XRP/USDT"], markets_cache
    )
    assert supported == ["ETH/USDT"]
    assert removed == ["XRP/USDT"]
    assert degraded is False


def test_market_filter_sdk(monkeypatch):
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.backend = "binance_sdk"
    ad.sdk = SDKDummy()
    ad.futures = True
    ad.last_warn_at = {}
    markets_cache = {}
    supported, removed, _ = filter_supported_symbols(
        ad, ["ETH/USDT", "XRP/USDT"], markets_cache
    )
    assert supported == ["ETH/USDT"]
    assert removed == ["XRP/USDT"]


def test_market_filter_failure(monkeypatch):
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.backend = "ccxt"
    ad.x = CCXTDummy()
    ad.last_warn_at = {}

    def bad_load():
        raise Exception("boom")

    monkeypatch.setattr(ad, "load_markets", bad_load)
    markets_cache = {}
    supported, removed, degraded = filter_supported_symbols(
        ad, ["ETH/USDT", "XRP/USDT"], markets_cache
    )
    assert supported == ["ETH/USDT", "XRP/USDT"]
    assert removed == []
    assert degraded is False
