import types
import sys
import ccxt_stub as ccxt
import pandas as pd

from exchange_adapter import ExchangeAdapter


class DummyAdapter:
    def __init__(self):
        self.data = None
        self.err = None

    def fetch_ohlcv(self, *args, **kwargs):
        if self.err:
            raise self.err
        return self.data


dummy_adapter = DummyAdapter()
_orig_main = sys.modules.get("main")
sys.modules["main"] = types.SimpleNamespace(ADAPTER=dummy_adapter)
from fallback import fallback_signal  # pylint: disable=wrong-import-position
if _orig_main is not None:
    sys.modules["main"] = _orig_main
else:
    del sys.modules["main"]


def test_fallback_signal_success():
    dummy_adapter.data = [[i, 0, 0, 0, float(i), 0] for i in range(1, 51)]
    dummy_adapter.err = None
    signal = fallback_signal("BTC/USDT")
    assert signal in {"long", "short", "hold"}


def test_fallback_signal_error():
    dummy_adapter.data = None
    dummy_adapter.err = ccxt.NetworkError("boom")
    assert fallback_signal("BTC/USDT") == "hold"
