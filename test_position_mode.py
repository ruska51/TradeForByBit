#!/usr/bin/env python3
"""Тест проверки реального position mode на бирже."""

import sys
import os

sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

# Load API keys
from credentials import API_KEY, API_SECRET

import ccxt

# Create exchange
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {
        'defaultType': 'swap',
    }
})

exchange.set_sandbox_mode(True)

print("="*80)
print("ПРОВЕРКА POSITION MODE")
print("="*80)

# Check account position mode
try:
    response = exchange.private_get_v5_account_info()
    print(f"\nAccount info response:")
    print(f"  {response}")
except Exception as e:
    print(f"Error getting account info: {e}")

# Check XRP position - try different symbol formats
test_symbols = [
    ("XRPUSDT", "linear"),
    ("XRP/USDT", "linear"),
    ("XRP/USDT:USDT", "linear"),
]

for symbol, category in test_symbols:
    print(f"\nTrying symbol: {symbol}, category: {category}")

    try:
        response = exchange.private_get_v5_position_list({
            "category": category,
            "symbol": symbol,
        })

        result = response.get("result", {})
        positions = result.get("list", [])

        if not positions:
            print("  No positions found")
        else:
            print(f"  [OK] FOUND {len(positions)} position(s)!")
            for pos in positions:
                print(f"\n    Position:")
                print(f"      symbol: {pos.get('symbol')}")
                print(f"      side: {pos.get('side')}")
                print(f"      size: {pos.get('size')}")
                print(f"      positionIdx: {pos.get('positionIdx')}")  # КРИТИЧНО!
                print(f"      positionMode: {pos.get('positionMode', 'N/A')}")
            break  # Found it!

    except Exception as e:
        print(f"  [FAIL] Error: {e}")

# Check position mode setting
print(f"\n{'='*80}")
print("CHECKING POSITION MODE SETTING:")
print("="*80)

try:
    # Get position mode for linear
    response = exchange.private_get_v5_position_switch_mode({
        "category": "linear",
    })

    print(f"\nSwitch mode response:")
    print(f"  {response}")

except Exception as e:
    print(f"Error getting switch mode: {e}")

print("\n" + "="*80)
print("ВЫВОД:")
print("="*80)
print("Если positionIdx=0 -> ONE-WAY mode (используй positionIdx=0)")
print("Если positionIdx=1 или 2 -> HEDGE mode (используй positionIdx=1 для Long)")
print("="*80)
