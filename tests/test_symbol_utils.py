import logging
from types import SimpleNamespace

import pytest

from symbol_utils import normalize_symbol_for_exchange


class DummyExchange:
    def __init__(self, markets=None, markets_by_id=None):
        self.markets = markets or {}
        self.markets_by_id = markets_by_id or {}

    def load_markets(self, reload=True):
        pass


def test_normalize_symbol_for_exchange_direct():
    markets_cache = {"loaded": False, "ts": 0.0, "by_name": set(), "by_id": set()}
    ex = DummyExchange(markets={"ETH/USDT": {}}, markets_by_id={})
    assert normalize_symbol_for_exchange(ex, "ETH/USDT", markets_cache) == "ETH/USDT"


def test_normalize_symbol_for_exchange_suffix():
    markets_cache = {"loaded": False, "ts": 0.0, "by_name": set(), "by_id": set()}
    ex = DummyExchange(markets={"ETH/USDT:USDT": {}}, markets_by_id={})
    assert normalize_symbol_for_exchange(ex, "ETH/USDT", markets_cache) == "ETH/USDT:USDT"


def test_normalize_symbol_for_exchange_id_form():
    markets_cache = {"loaded": False, "ts": 0.0, "by_name": set(), "by_id": set()}
    ex = DummyExchange(markets={}, markets_by_id={"ETHUSDT": {"symbol": "ETH/USDT"}})
    assert normalize_symbol_for_exchange(ex, "ETH/USDT", markets_cache) == "ETH/USDT"

