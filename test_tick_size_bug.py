#!/usr/bin/env python3
"""
Тест для проверки проблемы с tick_size
"""

price = 3.8631
sl_price = 3.6708  # Правильное значение для long
tp_price = 4.3439  # Правильное значение для long
tick_size = 0.99976977  # ИЗ ЛОГА!!!

print(f"BEFORE rounding:")
print(f"  price: {price}")
print(f"  sl_price: {sl_price}")
print(f"  tp_price: {tp_price}")
print(f"  tick_size: {tick_size}")

# Применяем округление как в коде
sl_price_rounded = round(tick_size * round(sl_price / tick_size), 8)
tp_price_rounded = round(tick_size * round(tp_price / tick_size), 8)

print(f"\nAFTER rounding (КАК В КОДЕ):")
print(f"  sl_price: {sl_price_rounded}")
print(f"  tp_price: {tp_price_rounded}")

print(f"\nПРОМЕЖУТОЧНЫЕ ЗНАЧЕНИЯ:")
print(f"  sl_price / tick_size = {sl_price / tick_size}")
print(f"  round(sl_price / tick_size) = {round(sl_price / tick_size)}")
print(f"  tick_size * round(sl_price / tick_size) = {tick_size * round(sl_price / tick_size)}")

print(f"\n  tp_price / tick_size = {tp_price / tick_size}")
print(f"  round(tp_price / tick_size) = {round(tp_price / tick_size)}")
print(f"  tick_size * round(tp_price / tick_size) = {tick_size * round(tp_price / tick_size)}")

if sl_price_rounded == tp_price_rounded:
    print(f"\nERROR: После округления SL и TP ОДИНАКОВЫЕ: {sl_price_rounded}!")
