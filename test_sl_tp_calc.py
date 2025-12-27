#!/usr/bin/env python3
"""
Тест для проверки расчета SL/TP
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import risk_management

def test_calc_sl_tp_long():
    """Тест расчета SL/TP для LONG позиции"""
    price = 3.8631
    atr_val = 0.05
    mode_params = {"sl_mult": 2.0, "tp_mult": 3.0}
    side = "long"
    tick_size = 0.0001

    tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
        price, atr_val, mode_params, side, tick_size=tick_size
    )

    print(f"=== TEST LONG POSITION ===")
    print(f"Entry price: {price:.4f}")
    print(f"ATR: {atr_val:.4f}")
    print(f"SL mult: {mode_params['sl_mult']}, TP mult: {mode_params['tp_mult']}")
    print(f"\nRESULTS:")
    print(f"TP price: {tp_price:.4f} (should be ABOVE {price:.4f})")
    print(f"SL price: {sl_price:.4f} (should be BELOW {price:.4f})")
    print(f"SL %: {sl_pct:.4f}")

    # Проверки
    assert tp_price > price, f"ERROR: TP {tp_price} должен быть ВЫШЕ цены входа {price} для LONG"
    assert sl_price < price, f"ERROR: SL {sl_price} должен быть НИЖЕ цены входа {price} для LONG"

    print("\n✅ LONG position: OK")
    return tp_price, sl_price


def test_calc_sl_tp_short():
    """Тест расчета SL/TP для SHORT позиции"""
    price = 3.8631
    atr_val = 0.05
    mode_params = {"sl_mult": 2.0, "tp_mult": 3.0}
    side = "short"
    tick_size = 0.0001

    tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
        price, atr_val, mode_params, side, tick_size=tick_size
    )

    print(f"\n=== TEST SHORT POSITION ===")
    print(f"Entry price: {price:.4f}")
    print(f"ATR: {atr_val:.4f}")
    print(f"SL mult: {mode_params['sl_mult']}, TP mult: {mode_params['tp_mult']}")
    print(f"\nRESULTS:")
    print(f"TP price: {tp_price:.4f} (should be BELOW {price:.4f})")
    print(f"SL price: {sl_price:.4f} (should be ABOVE {price:.4f})")
    print(f"SL %: {sl_pct:.4f}")

    # Проверки
    assert tp_price < price, f"ERROR: TP {tp_price} должен быть НИЖЕ цены входа {price} для SHORT"
    assert sl_price > price, f"ERROR: SL {sl_price} должен быть ВЫШЕ цены входа {price} для SHORT"

    print("\n✅ SHORT position: OK")
    return tp_price, sl_price


if __name__ == "__main__":
    try:
        test_calc_sl_tp_long()
        test_calc_sl_tp_short()
        print("\n" + "="*50)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        print("="*50)
    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
