import numpy as np
from sklearn.datasets import make_classification

import numpy as np
from sklearn.datasets import make_classification

from model_utils import load_global_model, make_xgb_classifier
import pandas as pd
import joblib


def test_make_xgb_classifier_multiclass():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    clf = make_xgb_classifier(3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)) <= {0, 1, 2}


def test_train_optuna_model(tmp_path, monkeypatch):
    import ccxt_stub as ccxt
    import importlib
    import sys
    sys.modules["ccxt"] = ccxt
    sys.modules.pop("main", None)

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

    monkeypatch.setattr(ccxt, "bybit", DummyExchange)
    import main
    importlib.reload(main)

    # synthetic OHLCV data with minimal required columns
    n = 50
    data = {}
    for tf in main.timeframes:
        data[f"timestamp_{tf}"] = pd.date_range("2020-01-01", periods=n, freq="15min")
        data[f"close_{tf}"] = np.random.rand(n)
        data[f"open_{tf}"] = np.random.rand(n)
        data[f"high_{tf}"] = np.random.rand(n)
        data[f"low_{tf}"] = np.random.rand(n)
        data[f"volume_{tf}"] = np.random.rand(n)
    df = pd.DataFrame(data)

    monkeypatch.setattr(main, "fetch_multi_ohlcv", lambda *a, **k: df.copy())

    saved = {}
    orig_dump = joblib.dump

    def fake_dump(obj, path):
        saved["path"] = tmp_path / "model.joblib"
        orig_dump(obj, saved["path"])

    monkeypatch.setattr(main.joblib, "dump", fake_dump)

    main.train_optuna_model("TEST", n_trials=1)
    assert "path" in saved and saved["path"].exists()


def test_load_global_model_fallback(monkeypatch):
    """If joblib.load raises ModuleNotFoundError a fallback model is returned."""

    import os
    import types

    # Force check to pass but make joblib.load fail
    monkeypatch.setattr("model_utils._check_sklearn_install", lambda: (True, "ok"))

    def bad_load(*args, **kwargs):
        raise ModuleNotFoundError("sklearn broken")

    monkeypatch.setattr(joblib, "load", bad_load)
    monkeypatch.setattr(joblib, "dump", lambda *a, **k: None)
    monkeypatch.setattr(os.path, "exists", lambda *a, **k: False)

    class DummyScaler:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    def dummy_dataset():
        return np.zeros((5, 1)), np.zeros(5), DummyScaler(), ["f"]

    monkeypatch.setattr("model_utils.build_recent_dataset", dummy_dataset)

    model, scaler, features = load_global_model()
    from xgboost import XGBClassifier

    assert isinstance(model, XGBClassifier)
    assert features == ["f"]
