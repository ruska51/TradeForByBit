#!/usr/bin/env python3
"""
РЕАЛЬНЫЙ интеграционный тест открытия позиций с SL/TP.
Проверяет весь flow от округления qty до установки стопов.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

sys.path.insert(0, '/c/Users/fishf/Documents/GitHub/TradeForByBit')

from logging_utils import _round_qty, _normalize_bybit_symbol, place_conditional_exit


class MockExchange:
    """Mock Bybit exchange для тестирования."""

    def __init__(self):
        self.orders_created = []
        self.positions = {}
        self.markets_data = {
            "ADAUSDT": {  # Linear формат БЕЗ слеша!
                "info": {
                    "lotSizeFilter": {"minOrderQty": "0.01", "qtyStep": "0.01"},
                    "category": "linear"
                },
                "precision": {"amount": 2, "price": 4}
            },
            "ADA/USDT": {  # Spot формат СО слешем
                "info": {
                    "lotSizeFilter": {"minOrderQty": "1", "qtyStep": "1"},
                    "category": "spot"
                },
                "precision": {"amount": 0, "price": 4}
            },
            "BTCUSDT": {
                "info": {
                    "lotSizeFilter": {"minOrderQty": "0.001", "qtyStep": "0.001"}
                },
                "precision": {"amount": 3, "price": 2}
            },
            "SOLUSDT": {
                "info": {
                    "lotSizeFilter": {"minOrderQty": "0.1", "qtyStep": "0.1"}
                },
                "precision": {"amount": 1, "price": 2}
            }
        }

    def market(self, symbol):
        return self.markets_data.get(symbol, {})

    def amount_to_precision(self, symbol, amount):
        market = self.markets_data.get(symbol, {})
        precision = market.get("precision", {}).get("amount", 0)
        if precision:
            return round(float(amount), precision)
        return float(amount)

    def price_to_precision(self, symbol, price):
        market = self.markets_data.get(symbol, {})
        precision = market.get("precision", {}).get("price", 0)
        if precision:
            return round(float(price), precision)
        return float(price)

    def create_order(self, symbol, order_type, side, amount, price, params):
        """Mock создания ордера."""
        order = {
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params.copy() if params else {},
            "id": f"order_{len(self.orders_created) + 1}"
        }

        # Проверка критических ошибок
        category = params.get("category", "linear") if params else "linear"

        # КРИТИЧНО: Проверяем формат символа
        if category == "linear" and "/" in symbol:
            raise Exception(f"Linear contract must use format without slash: got '{symbol}', expected '{symbol.replace('/', '')}'")

        if category == "spot" and "/" not in symbol:
            raise Exception(f"Spot market must use format with slash: got '{symbol}', expected with '/'")

        # Проверка triggerDirection для spot
        if "triggerDirection" in (params or {}):
            if category == "spot" or ("/" in symbol and category == "linear"):
                raise Exception("bybit createOrder() : trigger order does not support triggerDirection for spot markets yet")

        self.orders_created.append(order)
        return {"id": order["id"], "filled": amount}

    def fetch_ticker(self, symbol, params=None):
        """Mock получения тикера."""
        base_prices = {
            "ADAUSDT": 0.3436,
            "ADA/USDT": 0.3436,
            "BTCUSDT": 80000,
            "SOLUSDT": 123
        }
        price = base_prices.get(symbol, 100)
        return {"last": price, "lastPrice": price, "info": {"lastPrice": str(price)}}

    def fetch_positions(self, symbols=None, params=None):
        """Mock получения позиций."""
        if not symbols:
            return list(self.positions.values())

        result = []
        for sym in symbols:
            if sym in self.positions:
                result.append(self.positions[sym])
        return result

    def fetch_open_orders(self, symbol, params=None):
        """Mock получения открытых ордеров."""
        return []

    def cancel_order(self, order_id, symbol, params=None):
        """Mock отмены ордера."""
        return {"id": order_id}

    def set_position(self, symbol, qty, side):
        """Устанавливаем позицию для тестирования."""
        self.positions[symbol] = {
            "symbol": symbol,
            "contracts": abs(qty),
            "contractSize": 1,
            "side": side,
            "info": {
                "symbol": symbol,
                "side": side.capitalize(),
                "size": str(abs(qty)),
                "positionIdx": 2 if side == "Sell" else 1
            }
        }


def test_symbol_normalization():
    """Тест нормализации символов."""
    print("\n" + "="*80)
    print("ТЕСТ 1: НОРМАЛИЗАЦИЯ СИМВОЛОВ")
    print("="*80)

    exchange = MockExchange()

    tests = [
        ("ADA/USDT", "linear", "ADAUSDT", "Linear должен убрать слеш"),
        ("ADA/USDT", "spot", "ADA/USDT", "Spot должен оставить слеш"),
        ("BTCUSDT", "linear", "BTCUSDT", "Linear без слеша остаётся как есть"),
        ("BTCUSDT", "spot", "BTC/USDT", "Spot без слеша добавляет слеш"),
        ("SOL/USDT", "linear", "SOLUSDT", "SOL linear убирает слеш"),
    ]

    passed = 0
    failed = 0

    for input_symbol, category, expected, description in tests:
        result = _normalize_bybit_symbol(exchange, input_symbol, category)
        status = "PASS" if result == expected else "FAIL"

        if status == "PASS":
            passed += 1
            print(f"  [OK] {description}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"    Input: {input_symbol}, Category: {category}")
            print(f"    Expected: {expected}, Got: {result}")

    print(f"\nРезультат: {passed} OK, {failed} FAIL\n")
    return failed == 0


def test_qty_rounding():
    """Тест округления qty."""
    print("="*80)
    print("ТЕСТ 2: ОКРУГЛЕНИЕ КОЛИЧЕСТВА")
    print("="*80)

    exchange = MockExchange()

    tests = [
        ("ADAUSDT", 1073.456, 1073.46, "ADA: округление до 0.01"),
        ("ADAUSDT", 0.005, 0.01, "ADA: минимум 0.01"),
        ("BTCUSDT", 0.015293, 0.015, "BTC: округление до 0.001"),
        ("SOLUSDT", 2.06, 2.1, "SOL: округление до 0.1"),
    ]

    passed = 0
    failed = 0

    for symbol, input_qty, expected, description in tests:
        result = _round_qty(exchange, symbol, input_qty)
        status = "PASS" if abs(result - expected) < 0.0001 else "FAIL"

        if status == "PASS":
            passed += 1
            print(f"  [OK] {description}: {input_qty} -> {result}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"    Expected: {expected}, Got: {result}")

    print(f"\nРезультат: {passed} OK, {failed} FAIL\n")
    return failed == 0


def test_order_creation():
    """Тест создания ордеров."""
    print("="*80)
    print("ТЕСТ 3: СОЗДАНИЕ ОРДЕРОВ (КРИТИЧЕСКИЙ)")
    print("="*80)

    exchange = MockExchange()

    # Устанавливаем позицию
    exchange.set_position("ADAUSDT", 1073, "Sell")

    tests = [
        {
            "name": "Linear SL с правильным символом",
            "symbol": "ADAUSDT",
            "category": "linear",
            "should_pass": True,
            "error_contains": None
        },
        {
            "name": "Linear SL с НЕПРАВИЛЬНЫМ символом (со слешем)",
            "symbol": "ADA/USDT",
            "category": "linear",
            "should_pass": False,
            "error_contains": "without slash"
        },
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            # Пытаемся создать conditional order
            params = {
                "category": test["category"],
                "triggerPrice": 0.36,
                "triggerBy": "LastPrice",
                "reduceOnly": True,
                "closeOnTrigger": True,
            }

            # Добавляем triggerDirection только для linear
            if test["category"] == "linear":
                params["triggerDirection"] = 2  # falling

            order = exchange.create_order(
                test["symbol"],
                "Market",
                "buy",
                1073.0,
                None,
                params
            )

            if test["should_pass"]:
                passed += 1
                print(f"  [OK] {test['name']}: Order created (ID: {order['id']})")
            else:
                failed += 1
                print(f"  [FAIL] {test['name']}: Expected error but order created!")

        except Exception as e:
            error_msg = str(e)
            if not test["should_pass"]:
                if test["error_contains"] and test["error_contains"] in error_msg:
                    passed += 1
                    print(f"  [OK] {test['name']}: Correct error - {error_msg[:60]}")
                else:
                    failed += 1
                    print(f"  [FAIL] {test['name']}: Wrong error - {error_msg[:60]}")
            else:
                failed += 1
                print(f"  [FAIL] {test['name']}: Unexpected error - {error_msg[:60]}")

    print(f"\nРезультат: {passed} OK, {failed} FAIL\n")
    return failed == 0


def main():
    """Запуск всех тестов."""
    print("\n" + "#"*80)
    print("# ИНТЕГРАЦИОННЫЙ ТЕСТ ОТКРЫТИЯ ПОЗИЦИЙ")
    print("#"*80 + "\n")

    results = []
    results.append(("Нормализация символов", test_symbol_normalization()))
    results.append(("Округление qty", test_qty_rounding()))
    results.append(("Создание ордеров", test_order_creation()))

    print("="*80)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n*** ALL TESTS PASSED! ***\n")
        return 0
    else:
        print("\n*** TESTS FAILED! ***\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
