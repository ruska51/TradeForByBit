import numpy as np
import pandas as pd
import pytest

import main


def test_trade_continues_without_long_tf(tmp_path, monkeypatch):
    symbol = "BTC/USDT"
    main.symbols = [symbol]
    main.reserve_symbols = []
    main.risk_state = {}
    main.open_trade_ctx.clear()
    main.ENABLE_REPORTS_BUILDER = False
    main.risk_config["trades_path"] = str(tmp_path / "trades.csv")

    class DummyStats:
        def is_banned(self, s):
            return False

        def pop_soft_risk(self, s):
            return 0

    main.stats = DummyStats()

    class DummyLimiter:
        def can_trade(self, s, eq):
            return True

    class DummyCool:
        def can_trade(self, s, bar):
            return True

    main.limiter = DummyLimiter()
    main.cool = DummyCool()

    class DummyModel:
        classes_ = [0, 1, 2]

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    class DummyScaler:
        def transform(self, X):
            return X

    monkeypatch.setattr(main, "GLOBAL_MODEL", DummyModel())
    monkeypatch.setattr(main, "GLOBAL_SCALER", DummyScaler())
    monkeypatch.setattr(main, "GLOBAL_FEATURES", None)
    monkeypatch.setattr(main, "_maybe_retrain_global", lambda: None)

    def fake_fetch_multi_ohlcv(symbol, tfs, limit=300, warn=True):
        periods = 100
        idx = pd.date_range("2024-01-01", periods=periods, freq="15T")
        df = pd.DataFrame(
            {
                "timestamp_15m": idx,
                "open_15m": np.ones(periods),
                "high_15m": np.ones(periods),
                "low_15m": np.ones(periods),
                "close_15m": np.linspace(1, 1.1, periods),
                "volume_15m": np.ones(periods),
            }
        )
        df.attrs["reduced"] = True
        return df

    monkeypatch.setattr(main, "fetch_multi_ohlcv", fake_fetch_multi_ohlcv)

    def fake_fetch_ohlcv(symbol, tf="15m", limit=100):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=limit, freq="15T"),
                "open": [1] * limit,
                "high": [1] * limit,
                "low": [1] * limit,
                "close": [1] * limit,
                "volume": [1] * limit,
            }
        )

    monkeypatch.setattr(main, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(main, "save_candle_chart", lambda *a, **k: None)
    monkeypatch.setattr(main, "detect_pattern_image", lambda *a, **k: {"pattern_name": "none", "confidence": 0.0})

    async def fake_detect_pattern(symbol, df):
        return {"pattern_name": "none", "source": "none", "confidence": 0.0}

    monkeypatch.setattr(main, "detect_pattern", fake_detect_pattern)
    monkeypatch.setattr(main, "record_pattern", lambda *a, **k: None)

    monkeypatch.setattr(main, "update_pair_stats", lambda *a, **k: {})
    monkeypatch.setattr(main, "adjust_state_by_stats", lambda state, stats_dict, cfg: state)
    monkeypatch.setattr(main, "save_pair_report", lambda *a, **k: None)
    monkeypatch.setattr(main, "save_risk_state", lambda *a, **k: None)
    monkeypatch.setattr(main, "update_dynamic_thresholds", lambda: None)
    monkeypatch.setattr(main, "flush_cycle_logs", lambda: None)
    monkeypatch.setattr(main, "record_error", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_trades_csv_header", lambda *a, **k: None)

    class DummyAdapter:
        def fetch_open_orders(self, symbol=None):
            return (0, [])

        def cancel_open_orders(self, symbol):
            return (0, [])

    main.ADAPTER = DummyAdapter()

    class DummyExchange:
        def fetch_closed_orders(self, sym, since=None, limit=5):
            return []

    main.exchange = DummyExchange()

    monkeypatch.setattr(main, "get_symbol_params", lambda s: main.StrategyParams.from_dict(main.DEFAULT_PARAMS))
    monkeypatch.setattr(main, "apply_params", lambda params: None)

    called = {}

    def fake_pattern_self_test(df):
        called["hit"] = True
        raise SystemExit

    monkeypatch.setattr(main, "pattern_self_test", fake_pattern_self_test)

    with pytest.raises(SystemExit):
        main.run_bot()

    assert called.get("hit")
