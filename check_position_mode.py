#!/usr/bin/env python3
"""Проверить режим позиций на бирже."""

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
print("ПРОВЕРКА РЕЖИМА ПОЗИЦИЙ")
print("="*80)

# Check account settings for position mode
try:
    # Bybit API endpoint to get account info
    response = exchange.private_get_v5_account_info()

    print("\nAccount Info:")
    print(response)

except Exception as e:
    print(f"Error getting account info: {e}")

# Alternative: check position mode through position list
print("\n" + "="*80)
print("ПРОВЕРКА ЧЕРЕЗ ПОЗИЦИИ:")
print("="*80)

try:
    response = exchange.private_get_v5_position_list({
        "category": "linear",
        "settleCoin": "USDT",  # Required parameter
    })

    positions = response.get("result", {}).get("list", [])

    if positions:
        print(f"\nНайдено {len(positions)} позиций (включая пустые)")

        hedge_positions = []
        oneway_positions = []

        for pos in positions:
            idx = str(pos.get('positionIdx', ''))
            size = float(pos.get('size', 0))

            if size > 0:  # Only non-empty positions
                print(f"\n{pos.get('symbol')}: positionIdx={idx}, size={size}, side={pos.get('side')}")

                if idx in ('1', '2'):
                    hedge_positions.append(pos)
                elif idx == '0':
                    oneway_positions.append(pos)

        print("\n" + "-"*80)
        if hedge_positions:
            print(f"[HEDGE MODE] Найдено {len(hedge_positions)} позиций с positionIdx=1/2")
            print("ВЫВОД: Биржа в HEDGE MODE или были открыты позиции в HEDGE MODE")
        if oneway_positions:
            print(f"[ONE-WAY MODE] Найдено {len(oneway_positions)} позиций с positionIdx=0")
            print("ВЫВОД: Биржа в ONE-WAY MODE или были открыты позиции в ONE-WAY MODE")

        if hedge_positions and oneway_positions:
            print("\n[КОНФЛИКТ!] На бирже есть и HEDGE и ONE-WAY позиции!")
            print("Это невозможно в теории - биржа должна быть в одном режиме!")

    else:
        print("\nНет позиций")

except Exception as e:
    print(f"Error: {e}")

print("="*80)
