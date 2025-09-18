def test_normalize_and_degraded():
    from symbol_utils import filter_supported_symbols

    class DummyAdapter:
        _markets_available = True

        def supports_symbol(self, symbol):
            return symbol == "ETH/USDT"

    markets_cache = {}
    symbols = ["ETH/USDT", "SOL/USDT"]
    supported, removed, degraded = filter_supported_symbols(DummyAdapter(), symbols, markets_cache)
    assert supported == ["ETH/USDT"]
    assert removed == ["SOL/USDT"]
    assert degraded is False
