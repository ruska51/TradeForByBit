import ast
import types
from pathlib import Path

from exchange_adapter import ExchangeAdapter


def _make_adapter(obj):
    ad = ExchangeAdapter.__new__(ExchangeAdapter)
    ad.backend = "ccxt"
    ad.x = obj
    ad.sdk = None
    ad.config = {}
    ad.futures = True
    ad.last_warn_at = {}
    ad.markets_loaded_at = 0
    ad.sandbox = False
    return ad


def test_cancel_open_orders_absent_methods():
    class X:
        pass

    ad = _make_adapter(X())
    assert ad.cancel_open_orders("BTC/USDT") == (0, [])


def test_cancel_returns_tuple_even_on_errors():
    class X:
        def fetch_open_orders(self, symbol=None):
            raise RuntimeError("boom")

        def cancel_all_orders(self, symbol=None):
            raise RuntimeError("boom")

    ad = _make_adapter(X())
    assert ad.cancel_open_orders("BTC/USDT") == (0, [])


def test_cancel_ccxt_flow_counts_prefetch():
    class X:
        def fetch_open_orders(self, symbol=None):
            return [{"id": 1}, {"id": 2}, {"id": 3}]

        def cancel_all_orders(self, symbol=None):
            return None

    ad = _make_adapter(X())
    assert ad.cancel_open_orders("BTC/USDT") == (3, [1, 2, 3])


def test_cancel_stale_orders_none_guard():
    src = Path("main.py").read_text()
    tree = ast.parse(src)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cancel_stale_orders")
    code = compile(ast.Module([func], []), "main.py", "exec")
    env: dict = {}
    adapter = types.SimpleNamespace(cancel_open_orders=lambda symbol: (None, []))
    log = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, error=lambda *a, **k: None)
    exec(code, {"ADAPTER": adapter, "logging": log}, env)
    assert env["cancel_stale_orders"]("BTC/USDT") == 0

