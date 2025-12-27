#!/usr/bin/env python3
"""Полный тест исправлений."""

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

from logging_utils import _force_hedge_mode_check, _normalize_bybit_symbol

print("="*80)
print("ПОЛНЫЙ ТЕСТ ИСПРАВЛЕНИЙ")
print("="*80)

# Test 1: Symbol normalization
print("\nТест 1: Нормализация символов")
print("-"*80)

symbol = "XRP/USDT"
category = "linear"

norm = _normalize_bybit_symbol(exchange, symbol, category)
print(f"  Input: {symbol}, category={category}")
print(f"  Output: {norm}")
print(f"  Expected: XRP/USDT:USDT")
print(f"  Result: {'[OK]' if norm == 'XRP/USDT:USDT' else '[FAIL]'}")

# Test 2: Hedge mode check
print("\nТест 2: Проверка position mode")
print("-"*80)

is_hedge = _force_hedge_mode_check(exchange, norm, category)
print(f"  Symbol: {norm}")
print(f"  is_hedge: {is_hedge}")
print(f"  Expected: False (ONE-WAY mode)")
print(f"  Result: {'[OK]' if not is_hedge else '[FAIL - should be False!]'}")

# Test 3: Check actual position
print("\nТест 3: Проверка реальной позиции")
print("-"*80)

# Bybit API format
api_symbol = "XRPUSDT"

try:
    response = exchange.private_get_v5_position_list({
        "category": "linear",
        "symbol": api_symbol,
    })

    positions = response.get("result", {}).get("list", [])

    if positions:
        for pos in positions:
            print(f"  Symbol: {pos.get('symbol')}")
            print(f"  positionIdx: {pos.get('positionIdx')}")
            print(f"  Side: {pos.get('side')}")
            print(f"  Size: {pos.get('size')}")

            idx = pos.get('positionIdx')
            if idx == 0 or idx == "0":
                print(f"  [OK] ONE-WAY mode confirmed (positionIdx=0)")
            else:
                print(f"  [FAIL] Expected positionIdx=0, got {idx}")
    else:
        print("  No position found")

except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*80)
print("РЕЗЮМЕ:")
print("="*80)
print("1. Символы должны конвертироваться в XRP/USDT:USDT для CCXT")
print("2. _force_hedge_mode_check должен вернуть False (ONE-WAY)")
print("3. positionIdx в реальной позиции должен быть 0")
print("="*80)
