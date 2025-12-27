#!/usr/bin/env python3
"""
Проверка исправления - calc_sl_tp должен возвращать РАЗНЫЕ значения для SL и TP
даже при неправильном tick_size
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import risk_management

# Реальные параметры из логов для SUI/USDT
price = 3.8631
atr_val = 0.16027857
mode_params = {"sl_mult": 1.2, "tp_mult": 3.0}
side = "long"
tick_size = 0.99976977  # НЕПРАВИЛЬНЫЙ tick_size из лога!

print("=== ТЕСТ С НЕПРАВИЛЬНЫМ tick_size ===")
print(f"price: {price}")
print(f"atr: {atr_val}")
print(f"tick_size: {tick_size} (НЕПРАВИЛЬНЫЙ!)")

tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
    price, atr_val, mode_params, side, tick_size=tick_size
)

print(f"\nРЕЗУЛЬТАТ:")
print(f"TP price: {tp_price:.8f}")
print(f"SL price: {sl_price:.8f}")
print(f"SL %: {sl_pct:.4f}")

if tp_price == sl_price:
    print(f"\n❌ ОШИБКА: TP и SL ОДИНАКОВЫЕ: {tp_price:.8f}")
    sys.exit(1)
elif tp_price > price and sl_price < price:
    print(f"\n✅ УСПЕХ: TP={tp_price:.4f} > price={price:.4f} > SL={sl_price:.4f}")
    print("SL и TP РАЗНЫЕ и правильно расположены для LONG позиции!")
else:
    print(f"\n❌ ОШИБКА: Неправильное расположение")
    print(f"  TP={tp_price:.4f} (должен быть > {price:.4f})")
    print(f"  SL={sl_price:.4f} (должен быть < {price:.4f})")
    sys.exit(1)
