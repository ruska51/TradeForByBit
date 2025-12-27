#!/usr/bin/env python3
"""
Тест функции _normalize_bybit_symbol.
Проверяет конверсию форматов символов для spot/linear контрактов.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from logging_utils import _normalize_bybit_symbol


class MockBybitExchange:
    """Mock Bybit exchange с правильным id."""

    def __init__(self):
        self.id = "bybit"  # КРИТИЧНО: должен содержать "bybit"!
        self.orders_created = []


def test_symbol_normalization():
    """Тест нормализации символов."""
    print("\n" + "="*80)
    print("ТЕСТ: НОРМАЛИЗАЦИЯ СИМВОЛОВ BYBIT")
    print("="*80)

    exchange = MockBybitExchange()

    tests = [
        # (input_symbol, category, expected_output, description)
        ("SOL/USDT", "linear", "SOLUSDT", "Linear: убрать слеш из SOL/USDT"),
        ("ADA/USDT", "linear", "ADAUSDT", "Linear: убрать слеш из ADA/USDT"),
        ("BTC/USDT", "linear", "BTCUSDT", "Linear: убрать слеш из BTC/USDT"),
        ("ETHUSDT", "linear", "ETHUSDT", "Linear: уже без слеша - оставить как есть"),

        ("BTCUSDT", "spot", "BTC/USDT", "Spot: добавить слеш в BTCUSDT"),
        ("SOLUSDT", "spot", "SOL/USDT", "Spot: добавить слеш в SOLUSDT"),
        ("ADA/USDT", "spot", "ADA/USDT", "Spot: уже со слешем - оставить как есть"),
    ]

    passed = 0
    failed = 0

    for input_symbol, category, expected, description in tests:
        result = _normalize_bybit_symbol(exchange, input_symbol, category)

        if result == expected:
            passed += 1
            print(f"  [PASS] {description}")
            print(f"         Input: '{input_symbol}' + category='{category}' -> '{result}'")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Input: '{input_symbol}' + category='{category}'")
            print(f"         Expected: '{expected}', Got: '{result}'")

    print(f"\n{'='*80}")
    print(f"Результат: {passed} OK, {failed} FAIL")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = test_symbol_normalization()
    sys.exit(0 if success else 1)
