#!/usr/bin/env python3
"""Закрыть все позиции для тестирования."""

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
print("ЗАКРЫТИЕ ВСЕХ ПОЗИЦИЙ")
print("="*80)

# Get all positions
positions = exchange.fetch_positions(params={"category": "linear"})

for pos in positions:
    if pos.get('contracts', 0) > 0 or pos.get('contractSize', 0) > 0:
        symbol = pos.get('symbol')
        side = pos.get('side')
        size = pos.get('contracts')

        print(f"\nЗакрываю {symbol}: side={side}, size={size}")

        try:
            # Close position by placing opposite market order
            close_side = 'sell' if side == 'long' else 'buy'

            order = exchange.create_order(
                symbol,
                'market',
                close_side,
                size,
                None,
                {
                    'category': 'linear',
                    'positionIdx': 0,  # ONE-WAY mode
                    'reduceOnly': True,
                }
            )

            print(f"  [OK] Закрыто! Order ID: {order.get('id')}")

        except Exception as e:
            print(f"  [ERROR] {e}")

print("\n" + "="*80)
print("Готово! Все позиции закрыты.")
print("="*80)
