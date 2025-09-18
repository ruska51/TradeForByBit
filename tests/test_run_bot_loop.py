import importlib
import numpy as np
import pytest
import joblib


def test_run_bot_loop_invokes_predict(monkeypatch):
    import model_utils
    import ccxt_stub as ccxt
    import sys
    sys.modules["ccxt"] = ccxt

    # Force fallback model with a tiny synthetic dataset to avoid network calls
    monkeypatch.setattr(model_utils, "_check_sklearn_install", lambda: (False, "no"))

    class DummyScaler:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    def dummy_dataset():
        return np.zeros((5, 1)), np.zeros(5), DummyScaler(), ["f"]

    monkeypatch.setattr(model_utils, "build_recent_dataset", dummy_dataset)
    monkeypatch.setattr(joblib, "dump", lambda *a, **k: None)
    import os
    monkeypatch.setattr(os.path, "exists", lambda *a, **k: False)

    class DummyExchange:
        def __init__(self, *a, **k):
            self.markets = {}
            self.markets_by_id = {}

        def set_sandbox_mode(self, mode):
            pass

        def load_markets(self, reload=False):
            self.markets = {}
            self.markets_by_id = {}
            return self.markets

    monkeypatch.setattr(ccxt, "binance", DummyExchange)

    sys.modules.pop("main", None)
    import main
    importlib.reload(main)

    called = {}

    def fake_run_bot():
        main.GLOBAL_MODEL.predict([[0]])
        called["predict"] = True
        raise SystemExit

    monkeypatch.setattr(main, "run_bot", fake_run_bot)

    with pytest.raises(SystemExit):
        main.run_bot_loop()

    assert called.get("predict")
