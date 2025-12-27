"""
Тест для проверки функции calc_sl_tp.
"""

import risk_management

# Тест для LONG позиции
price = 3.9127
atr_val = 0.05
mode_params = {"sl_mult": 2.0, "tp_mult": 3.0}
side = "long"

tp, sl, sl_pct = risk_management.calc_sl_tp(price, atr_val, mode_params, side)

print(f"LONG позиция:")
print(f"  Цена входа: {price}")
print(f"  TP (должен быть ВЫШЕ): {tp:.4f}")
print(f"  SL (должен быть НИЖЕ): {sl:.4f}")
print(f"  SL %: {sl_pct:.4f}")
print()

# Проверка
assert tp > price, f"TP должен быть выше цены! tp={tp}, price={price}"
assert sl < price, f"SL должен быть ниже цены! sl={sl}, price={price}"

print("✅ LONG тест ПРОЙДЕН")
print()

# Тест для SHORT позиции
side = "short"

tp, sl, sl_pct = risk_management.calc_sl_tp(price, atr_val, mode_params, side)

print(f"SHORT позиция:")
print(f"  Цена входа: {price}")
print(f"  TP (должен быть НИЖЕ): {tp:.4f}")
print(f"  SL (должен быть ВЫШЕ): {sl:.4f}")
print(f"  SL %: {sl_pct:.4f}")
print()

# Проверка
assert tp < price, f"TP должен быть ниже цены! tp={tp}, price={price}"
assert sl > price, f"SL должен быть выше цены! sl={sl}, price={price}"

print("✅ SHORT тест ПРОЙДЕН")
print()
print("="*60)
print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ - calc_sl_tp работает правильно!")
print("="*60)
