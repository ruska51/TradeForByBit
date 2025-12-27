#!/usr/bin/env python3
"""
Тест функции _round_qty для проверки правильности округления количества.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Mock exchange class
class MockExchange:
    def __init__(self):
        self.markets_data = {
            "BTC/USDT": {
                "info": {
                    "lotSizeFilter": {
                        "minOrderQty": "0.001",
                        "qtyStep": "0.001"
                    }
                },
                "precision": {"amount": 3, "price": 2}
            },
            "ETH/USDT": {
                "info": {
                    "lotSizeFilter": {
                        "minOrderQty": "0.01",
                        "qtyStep": "0.01"
                    }
                },
                "precision": {"amount": 2, "price": 2}
            },
            "SOL/USDT": {
                "info": {
                    "lotSizeFilter": {
                        "minOrderQty": "0.1",
                        "qtyStep": "0.1"
                    }
                },
                "precision": {"amount": 1, "price": 2}
            }
        }

    def market(self, symbol):
        return self.markets_data.get(symbol, {})

    def amount_to_precision(self, symbol, amount):
        market = self.markets_data.get(symbol, {})
        precision = market.get("precision", {}).get("amount", 0)
        if precision:
            return round(float(amount), precision)
        return float(amount)

# Import the function to test
sys.path.insert(0, '/c/Users/fishf/Documents/GitHub/TradeForByBit')
from logging_utils import _round_qty

def test_round_qty():
    """Тест округления qty для разных символов."""

    exchange = MockExchange()

    test_cases = [
        # (symbol, input_qty, expected_output, description)
        ("BTC/USDT", 0.015293, 0.015, "BTC should round to 0.001 step"),
        ("BTC/USDT", 0.0149, 0.015, "BTC should round up to nearest 0.001"),
        ("BTC/USDT", 0.0001, 0.001, "BTC below min should raise to min_qty"),

        ("ETH/USDT", 0.4147, 0.41, "ETH should round to 0.01 step"),
        ("ETH/USDT", 0.415, 0.42, "ETH should round up to 0.42"),
        ("ETH/USDT", 0.005, 0.01, "ETH below min should raise to min_qty"),

        ("SOL/USDT", 2.06, 2.1, "SOL should round to 0.1 step"),
        ("SOL/USDT", 2.04, 2.0, "SOL should round down to 2.0"),
        ("SOL/USDT", 0.05, 0.1, "SOL below min should raise to min_qty"),
    ]

    print("\n" + "="*80)
    print("ТЕСТ ОКРУГЛЕНИЯ КОЛИЧЕСТВА (_round_qty)")
    print("="*80 + "\n")

    passed = 0
    failed = 0

    for symbol, input_qty, expected, description in test_cases:
        result = _round_qty(exchange, symbol, input_qty)
        status = "✓ PASS" if abs(result - expected) < 0.0001 else "✗ FAIL"

        if status == "✓ PASS":
            passed += 1
        else:
            failed += 1

        print(f"{status} | {symbol:10} | {input_qty:.6f} -> {result:.6f} (expected {expected:.6f})")
        print(f"      {description}")
        print()

    print("="*80)
    print(f"Результаты: {passed} пройдено, {failed} провалено")
    print("="*80 + "\n")

    return failed == 0

if __name__ == "__main__":
    success = test_round_qty()
    sys.exit(0 if success else 1)
