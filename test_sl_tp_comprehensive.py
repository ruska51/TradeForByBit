#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексный тест для проверки правильности расчета SL/TP
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import risk_management

def test_sltp_various_scenarios():
    """Тест различных сценариев расчета SL/TP"""

    scenarios = [
        # (name, price, atr, sl_mult, tp_mult, side, tick_size)
        ("SUI/USDT LONG с неправильным tick_size", 3.8631, 0.16027857, 1.2, 3.0, "long", 0.99976977),
        ("SUI/USDT LONG с правильным tick_size", 3.8631, 0.16027857, 1.2, 3.0, "long", 0.0001),
        ("BTC/USDT LONG", 63000, 1000, 2.0, 3.0, "long", 0.1),
        ("ETH/USDT SHORT", 3500, 50, 2.0, 3.0, "short", 0.01),
        ("ADA/USDT LONG малый ATR", 0.35, 0.005, 2.0, 3.0, "long", 0.0001),
    ]

    all_passed = True

    for name, price, atr, sl_mult, tp_mult, side, tick_size in scenarios:
        print(f"\n{'='*60}")
        print(f"ТЕСТ: {name}")
        print(f"{'='*60}")
        print(f"  Цена: {price}")
        print(f"  ATR: {atr}")
        print(f"  Множители: SL={sl_mult}, TP={tp_mult}")
        print(f"  Сторона: {side}")
        print(f"  tick_size: {tick_size}")

        mode_params = {"sl_mult": sl_mult, "tp_mult": tp_mult}
        tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
            price, atr, mode_params, side, tick_size=tick_size
        )

        print(f"\n  РЕЗУЛЬТАТ:")
        print(f"    TP: {tp_price:.8f}")
        print(f"    SL: {sl_price:.8f}")
        print(f"    SL %: {sl_pct:.4f}")

        # Проверки
        passed = True
        errors = []

        # 1. SL и TP не должны быть одинаковыми
        if abs(tp_price - sl_price) < 0.0001:
            errors.append(f"TP и SL ОДИНАКОВЫЕ: {tp_price:.8f}")
            passed = False

        # 2. Для LONG: TP > price > SL
        if side.lower() == "long":
            if tp_price <= price:
                errors.append(f"TP ({tp_price:.4f}) должен быть > price ({price:.4f})")
                passed = False
            if sl_price >= price:
                errors.append(f"SL ({sl_price:.4f}) должен быть < price ({price:.4f})")
                passed = False

        # 3. Для SHORT: SL > price > TP
        if side.lower() == "short":
            if sl_price <= price:
                errors.append(f"SL ({sl_price:.4f}) должен быть > price ({price:.4f})")
                passed = False
            if tp_price >= price:
                errors.append(f"TP ({tp_price:.4f}) должен быть < price ({price:.4f})")
                passed = False

        # 4. Значения должны быть положительными
        if tp_price <= 0 or sl_price <= 0:
            errors.append(f"TP или SL отрицательные или нулевые")
            passed = False

        if passed:
            print(f"\n  УСПЕХ [OK]")
        else:
            print(f"\n  ОШИБКА:")
            for error in errors:
                print(f"    - {error}")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return 0
    else:
        print("НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ!")
        return 1

if __name__ == "__main__":
    sys.exit(test_sltp_various_scenarios())
