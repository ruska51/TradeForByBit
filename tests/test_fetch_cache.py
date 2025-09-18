import pandas as pd
import main

def test_fetch_ohlcv_cycle_cache(monkeypatch):
    calls = []
    now_ms = 1700000000000

    def fake_fetch(symbol, timeframe, limit):
        calls.append(1)
        return [[now_ms + i * 60000, 1, 1, 1, 1, 1] for i in range(limit)]

    monkeypatch.setattr(main.ADAPTER, "fetch_ohlcv", fake_fetch, raising=False)
    monkeypatch.setattr(main, "calculate_indicators", lambda df: df)
    monkeypatch.setattr(main.time, "time", lambda: 0)

    df1 = main.fetch_ohlcv("BTC/USDT", "5m", limit=10)
    df2 = main.fetch_ohlcv("BTC/USDT", "5m", limit=10)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(df1, df2)
