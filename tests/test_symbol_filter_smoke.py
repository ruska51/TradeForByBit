import logging

from symbol_utils import filter_supported_symbols


class DummyAdapter:
    _markets_available = False

    def supports_symbol(self, symbol):
        return True


def test_markets_unavailable_skip_filter(caplog):
    markets_cache = {}
    symbols = ["AAA/USDT", "BBB/USDT"]
    with caplog.at_level(logging.WARNING):
        supported, removed, degraded = filter_supported_symbols(DummyAdapter(), symbols, markets_cache)
    assert supported == symbols
    assert removed == []
    assert degraded is False
    assert "markets unavailable" in caplog.text
