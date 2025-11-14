import builtins
import types

import exchange_adapter
from exchange_adapter import ExchangeAdapter


def test_backend_forces_ccxt(monkeypatch):
    attempts = {"n": 0}

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - used in test
        if name.startswith("binance"):
            attempts["n"] += 1
            raise ImportError("no binance")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("EXCHANGE_BACKEND", "auto")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(exchange_adapter.ExchangeAdapter, "_activate_ccxt", lambda self: setattr(self, "backend", "ccxt"))

    ad1 = ExchangeAdapter()
    ad2 = ExchangeAdapter()

    assert ad1.backend == "ccxt"
    assert ad2.backend == "ccxt"
    # binance import should never be attempted
    assert attempts["n"] == 0
