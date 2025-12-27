#!/usr/bin/env python3
"""Тест функции has_open_position."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from credentials import API_KEY, API_SECRET
import ccxt

exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'swap'},
})

exchange.set_sandbox_mode(True)

from logging_utils import has_open_position

print("="*80)
print("ТЕСТ has_open_position")
print("="*80)

# Test with different symbol formats
test_symbols = [
    "SUI/USDT",
    "SUI/USDT:USDT",
    "SUIUSDT",
]

for symbol in test_symbols:
    print(f"\nTesting symbol: {symbol}")

    try:
        qty_signed, qty_abs = has_open_position(exchange, symbol, "linear")
        print(f"  Result: qty_signed={qty_signed}, qty_abs={qty_abs}")

        if qty_abs > 0:
            print(f"  [OK] Position found!")
        else:
            print(f"  [FAIL] No position (but we know SUI position exists!)")

    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "="*80)
print("TESTING fetch_positions DIRECTLY:")
print("="*80)

# Try different formats directly
formats_to_try = [
    ("SUI/USDT", "fetch_positions(['SUI/USDT'])"),
    ("SUI/USDT:USDT", "fetch_positions(['SUI/USDT:USDT'])"),
    (None, "fetch_positions() - all positions"),
]

for symbol_arg, description in formats_to_try:
    print(f"\n{description}:")

    try:
        if symbol_arg:
            positions = exchange.fetch_positions([symbol_arg], params={"category": "linear"})
        else:
            positions = exchange.fetch_positions(params={"category": "linear"})

        print(f"  Found {len(positions)} positions")

        for pos in positions:
            if pos.get('contracts', 0) > 0 or pos.get('contractSize', 0) > 0:
                print(f"    Symbol: {pos.get('symbol')}, Contracts: {pos.get('contracts')}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
