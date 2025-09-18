import asyncio
from types import SimpleNamespace

import pytest

import risk_management


def test_calc_position_size_clamps_min_qty():
    class DummyExchange:
        def market(self, symbol):
            return {"limits": {"amount": {"min": 0.01}}}

    adapter = SimpleNamespace(exchange=DummyExchange())
    qty = risk_management.calc_position_size(
        100,
        0.0001,
        1000,
        0.01,
        adapter,
        "BTC/USDT",
        0.001,
    )
    assert qty == pytest.approx(0.01)
