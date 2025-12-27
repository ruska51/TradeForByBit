#!/usr/bin/env python3
"""
Тест для проверки расчета SL/TP для SUI/USDT с реальными параметрами из логов
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import risk_management

def test_sui_calc():
    """Тест с реальными параметрами из логов для SUI/USDT"""
    price = 3.8631
    atr_val = 0.16027857

    # Вычисляем sl_pct и tp_pct как в коде
    atr_pct = atr_val / price  # 0.16027857 / 3.8631 = 0.0415
    print(f"ATR %: {atr_pct:.4f}")

    # sl_mult и tp_mult - давайте проверим разные значения
    for sl_mult in [2.0, 1.2]:
        for tp_mult in [3.0, 4.0]:
            mode_params = {"sl_mult": sl_mult, "tp_mult": tp_mult}
            side = "long"
            tick_size = 0.0001

            tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
                price, atr_val, mode_params, side, tick_size=tick_size
            )

            print(f"\n=== sl_mult={sl_mult}, tp_mult={tp_mult} ===")
            print(f"Entry price: {price:.4f}")
            print(f"TP price: {tp_price:.4f} ({'ABOVE' if tp_price > price else 'BELOW'} price)")
            print(f"SL price: {sl_price:.4f} ({'ABOVE' if sl_price > price else 'BELOW'} price)")
            print(f"SL %: {sl_pct:.4f}")

            # Проверки
            if tp_price > price and sl_price < price:
                print("OK for LONG")
            else:
                print(f"ERROR! tp={tp_price}, sl={sl_price}, price={price}")
                if tp_price == sl_price:
                    print("  ERROR: TP and SL are EQUAL!")

if __name__ == "__main__":
    test_sui_calc()
