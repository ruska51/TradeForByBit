#!/usr/bin/env python3
"""Проверка conditional orders на бирже."""

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
print("ПРОВЕРКА CONDITIONAL ORDERS НА БИРЖЕ")
print("="*80)

# Get all positions
positions = exchange.fetch_positions(params={"category": "linear"})

symbols_with_positions = []
for pos in positions:
    if pos.get('contracts', 0) > 0 or pos.get('contractSize', 0) > 0:
        symbol = pos.get('symbol')
        symbols_with_positions.append(symbol)
        print(f"\nПозиция: {symbol}")
        print(f"  Size: {pos.get('contracts')}")
        print(f"  Side: {pos.get('side')}")
        print(f"  StopLoss: {pos.get('stopLoss')}")
        print(f"  TakeProfit: {pos.get('takeProfit')}")

if not symbols_with_positions:
    print("\nНет открытых позиций!")
    sys.exit(0)

# Check conditional orders for each position
print("\n" + "="*80)
print("CONDITIONAL ORDERS:")
print("="*80)

for symbol in symbols_with_positions:
    print(f"\n{symbol}:")

    try:
        # Fetch open orders (includes conditional orders)
        orders = exchange.fetch_open_orders(symbol, params={"category": "linear"})

        if orders:
            print(f"  Найдено {len(orders)} открытых ордеров:")
            for order in orders:
                print(f"    ID: {order.get('id')}")
                print(f"    Type: {order.get('type')}")
                print(f"    Side: {order.get('side')}")
                print(f"    Amount: {order.get('amount')}")
                print(f"    Trigger Price: {order.get('triggerPrice')}")

                # Check params
                info = order.get('info', {})
                position_idx = info.get('positionIdx')
                print(f"    positionIdx: {position_idx}")

                if position_idx is None or position_idx == '':
                    print(f"    [WARNING] positionIdx MISSING!")
                elif str(position_idx) == '0':
                    print(f"    [OK] positionIdx=0 (ONE-WAY mode)")
                else:
                    print(f"    [INFO] positionIdx={position_idx}")
                print()
        else:
            print("  Нет открытых ордеров")

    except Exception as e:
        print(f"  Error: {e}")

print("="*80)
print("ВЫВОД:")
print("="*80)
print("Если conditional orders существуют и имеют positionIdx=0,")
print("то исправление сработало правильно!")
print("="*80)
