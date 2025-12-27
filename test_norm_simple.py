#!/usr/bin/env python3
"""Simple direct test of _normalize_bybit_symbol."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from logging_utils import _normalize_bybit_symbol, _is_bybit_exchange


class MockBybitExchange:
    def __init__(self):
        self.id = "bybit"


# Test 1: Check if exchange is recognized
ex = MockBybitExchange()
is_bybit = _is_bybit_exchange(ex)
print(f"Test 1: _is_bybit_exchange = {is_bybit}")
if not is_bybit:
    print("  [FAIL] Exchange not recognized as Bybit!")
    sys.exit(1)

# Test 2: Normalize SOL/USDT with linear category
result = _normalize_bybit_symbol(ex, 'SOL/USDT', 'linear')
print(f"Test 2: _normalize_bybit_symbol(ex, 'SOL/USDT', 'linear') = '{result}'")
if result == 'SOL/USDT:USDT':
    print("  [PASS]")
else:
    print(f"  [FAIL] Expected 'SOL/USDT:USDT', got '{result}'")
    sys.exit(1)

# Test 3: Normalize BTCUSDT with spot category
result = _normalize_bybit_symbol(ex, 'BTCUSDT', 'spot')
print(f"Test 3: _normalize_bybit_symbol(ex, 'BTCUSDT', 'spot') = '{result}'")
if result == 'BTC/USDT':
    print("  [PASS]")
else:
    print(f"  [FAIL] Expected 'BTC/USDT', got '{result}'")
    sys.exit(1)

print("\nAll tests PASSED!")
