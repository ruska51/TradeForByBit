#!/usr/bin/env python3
"""Переключить в HEDGE mode и закрыть TRX позицию."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from credentials import API_KEY, API_SECRET
import ccxt
import time

exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'swap'},
})

exchange.set_sandbox_mode(True)

print("="*80)
print("ПЕРЕКЛЮЧЕНИЕ В HEDGE MODE И ЗАКРЫТИЕ ПОЗИЦИЙ")
print("="*80)

# Step 1: Try to switch to HEDGE mode
print("\n[1] Переключение в HEDGE mode...")
try:
    # Bybit V5 API для переключения position mode
    response = exchange.private_post_v5_position_switch_mode({
        "category": "linear",
        "mode": 3,  # 0=MergedSingle (ONE-WAY), 3=BothSide (HEDGE)
        "symbol": "TRXUSDT",
        "coin": "USDT",
    })

    print(f"  Response: {response}")
    time.sleep(1)

except Exception as e:
    print(f"  Переключение не удалось (возможно уже в HEDGE): {e}")

# Step 2: Close TRX position with positionIdx=2
print("\n[2] Закрытие TRX позиции (positionIdx=2)...")
try:
    order = exchange.create_order(
        "TRX/USDT:USDT",
        'market',
        'buy',  # Close Short
        2232.0,
        None,
        {
            'category': 'linear',
            'positionIdx': 2,  # HEDGE mode Short
            'reduceOnly': True,
        }
    )

    print(f"  [OK] TRX закрыт! Order ID: {order.get('id')}")
    time.sleep(2)

except Exception as e:
    print(f"  [ERROR] {e}")

# Step 3: Switch back to ONE-WAY mode
print("\n[3] Переключение обратно в ONE-WAY mode...")
try:
    response = exchange.private_post_v5_position_switch_mode({
        "category": "linear",
        "mode": 0,  # ONE-WAY
        "symbol": "TRXUSDT",
        "coin": "USDT",
    })

    print(f"  Response: {response}")
    time.sleep(1)

except Exception as e:
    print(f"  Переключение не удалось: {e}")

# Step 4: Close APT position with positionIdx=0
print("\n[4] Закрытие APT позиции (positionIdx=0)...")
try:
    order = exchange.create_order(
        "APT/USDT:USDT",
        'market',
        'sell',  # Close Long
        693.58,
        None,
        {
            'category': 'linear',
            'positionIdx': 0,  # ONE-WAY mode
            'reduceOnly': True,
        }
    )

    print(f"  [OK] APT закрыт! Order ID: {order.get('id')}")

except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "="*80)
print("Готово!")
print("="*80)
