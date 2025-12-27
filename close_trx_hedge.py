#!/usr/bin/env python3
"""Закрыть TRX позицию с positionIdx=2 (HEDGE mode)."""

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
print("ЗАКРЫТИЕ TRX ПОЗИЦИИ С positionIdx=2")
print("="*80)

symbol = "TRX/USDT:USDT"
size = 2232.0

print(f"\nЗакрываю {symbol} (positionIdx=2, Short)")

try:
    # Close Short position (positionIdx=2) by buying
    order = exchange.create_order(
        symbol,
        'market',
        'buy',  # Покупаем чтобы закрыть Short
        size,
        None,
        {
            'category': 'linear',
            'positionIdx': 2,  # HEDGE mode Short
            'reduceOnly': True,
        }
    )

    print(f"  [OK] Закрыто! Order ID: {order.get('id')}")

except Exception as e:
    print(f"  [ERROR] {e}")

print("="*80)
