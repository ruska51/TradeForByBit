#!/usr/bin/env python3
"""Проверить открытые ордера TRX."""

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
print("ОТКРЫТЫЕ ОРДЕРА TRX")
print("="*80)

try:
    orders = exchange.fetch_open_orders("TRX/USDT:USDT", params={"category": "linear"})

    if orders:
        print(f"\nНайдено {len(orders)} открытых ордеров:")

        for order in orders:
            print(f"\nOrder ID: {order.get('id')}")
            print(f"  Time: {order.get('datetime')}")
            print(f"  Side: {order.get('side')}")
            print(f"  Type: {order.get('type')}")
            print(f"  Amount: {order.get('amount')}")
            print(f"  Status: {order.get('status')}")

            info = order.get('info', {})
            print(f"  positionIdx: {info.get('positionIdx')}")
            print(f"  reduceOnly: {info.get('reduceOnly')}")

            if order.get('id') == '59a78158-b61c-41cb-8c74-c5a576b0a4bf':
                print("  [!!!] ЭТО НАШ ОРДЕР НА ЗАКРЫТИЕ - ОН ОТКРЫТ!")
    else:
        print("\nНет открытых ордеров")

except Exception as e:
    print(f"Error: {e}")

print("="*80)
