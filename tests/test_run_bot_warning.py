import types
import sys
import pytest
import ccxt_stub as ccxt

sys.modules["ccxt"] = ccxt


def test_run_bot_loop_warn_once(monkeypatch, caplog):
    class DummyExchange:
        def __init__(self, *a, **k):
            pass

        def set_sandbox_mode(self, mode):
            pass

        def load_markets(self, reload=False):
            return {}

    monkeypatch.setattr(ccxt, "binance", DummyExchange)
    import importlib
    sys.modules.pop("main", None)
    import main
    importlib.reload(main)

    monkeypatch.setattr(main.ADAPTER, "fetch_ohlcv", lambda *a, **k: None)
    class DummyModel:
        classes_ = [0, 1, 2]

        def predict(self, X):
            return [0] * len(X)

    class DummyScaler:
        def transform(self, X):
            return X

    monkeypatch.setattr(main, "GLOBAL_MODEL", DummyModel())
    monkeypatch.setattr(main, "GLOBAL_SCALER", DummyScaler())
    monkeypatch.setattr(main, "GLOBAL_FEATURES", ["close_15m"])
    monkeypatch.setattr(main, "_maybe_retrain_global", lambda: None)

    def fake_run_bot():
        main.backtest("BTC/USDT")
        raise SystemExit

    monkeypatch.setattr(main, "run_bot", fake_run_bot)

    caplog.set_level("WARNING")
    with pytest.raises(SystemExit):
        main.run_bot_loop()

    warns = [
        r for r in caplog.records if "no OHLCV for required timeframes" in r.getMessage()
    ]
    assert len(warns) == 1

