#!/usr/bin/env python3
"""Проверить TRX позицию напрямую через API."""

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

print("="*80)
print("ПРОВЕРКА TRX ПОЗИЦИИ")
print("="*80)

# Direct API call
response = exchange.private_get_v5_position_list({
    "category": "linear",
    "symbol": "TRXUSDT",
})

positions = response.get("result", {}).get("list", [])

if positions:
    for pos in positions:
        print(f"\nSymbol: {pos.get('symbol')}")
        print(f"Side: {pos.get('side')}")
        print(f"Size: {pos.get('size')}")
        print(f"positionIdx: {pos.get('positionIdx')}")  # КРИТИЧНО!
        print(f"Entry price: {pos.get('avgPrice')}")

        idx = pos.get('positionIdx')
        if str(idx) == '0':
            print("\n[INFO] Позиция в ONE-WAY mode (positionIdx=0)")
            print("  Для закрытия нужно использовать positionIdx=0")
        elif str(idx) in ('1', '2'):
            print(f"\n[ERROR] Позиция в HEDGE mode (positionIdx={idx})!")
            print("  Но биржа в ONE-WAY mode - конфликт!")
            print("  Возможно позиция была открыта с неправильным positionIdx")
else:
    print("\nПозиция TRX не найдена")

print("="*80)
