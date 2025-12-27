#!/usr/bin/env python3
"""Тест параметров conditional order для ONE-WAY mode."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from unittest.mock import Mock, patch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*80)
print("ТЕСТ: Параметры conditional order для ONE-WAY mode")
print("="*80)

# Mock exchange
mock_exchange = Mock()
mock_exchange.options = {'defaultType': 'swap'}

# Mock position list response for ONE-WAY mode
mock_position_response = {
    "result": {
        "list": [{
            "symbol": "BNBUSDT",
            "side": "Buy",
            "size": "1.0",
            "positionIdx": 0,  # ONE-WAY mode!
        }]
    }
}

mock_exchange.private_get_v5_position_list = Mock(return_value=mock_position_response)

# Mock create_order to capture params
created_orders = []

def mock_create_order(symbol, order_type, side, amount, price, params):
    created_orders.append({
        'symbol': symbol,
        'type': order_type,
        'side': side,
        'amount': amount,
        'price': price,
        'params': params.copy()
    })
    return {'id': 'test_order_123', 'orderId': 'test_order_123'}

mock_exchange.create_order = Mock(side_effect=mock_create_order)

# Mock fetch_open_orders
mock_exchange.fetch_open_orders = Mock(return_value=[])

# Import after mocking
from logging_utils import place_conditional_exit

# Test 1: SL order for ONE-WAY mode
print("\nТест 1: SL ордер для ONE-WAY mode")
print("-"*80)

created_orders.clear()

result = place_conditional_exit(
    exchange=mock_exchange,
    symbol="BNB/USDT",
    category="linear",
    exit_side="sell",
    amount=1.0,
    trigger_price=1091.0,
    is_tp=False,
    is_hedge=False,  # ONE-WAY mode
    position_idx=0,
)

if created_orders:
    order = created_orders[0]
    params = order['params']

    print(f"  Symbol: {order['symbol']}")
    print(f"  Type: {order['type']}")
    print(f"  Side: {order['side']}")
    print(f"  Amount: {order['amount']}")
    print(f"  Params: {params}")

    # Check critical params
    checks = [
        ("category", params.get('category') == 'linear', f"category={params.get('category')}"),
        ("triggerPrice", params.get('triggerPrice') == 1091.0, f"triggerPrice={params.get('triggerPrice')}"),
        ("triggerDirection", params.get('triggerDirection') == 1, f"triggerDirection={params.get('triggerDirection')}"),
        ("positionIdx", params.get('positionIdx') == 0, f"positionIdx={params.get('positionIdx')}"),  # КРИТИЧНО!
        ("reduceOnly", params.get('reduceOnly') is True, f"reduceOnly={params.get('reduceOnly')}"),
    ]

    all_ok = True
    for name, check, value in checks:
        status = "[OK]" if check else "[FAIL]"
        print(f"  {status} {name}: {value}")
        if not check:
            all_ok = False

    if all_ok:
        print("\n  [SUCCESS] Все параметры правильные!")
    else:
        print("\n  [FAIL] Некоторые параметры неправильные!")
else:
    print("  [ERROR] Order не был создан!")

# Test 2: TP order for ONE-WAY mode
print("\nТест 2: TP ордер для ONE-WAY mode")
print("-"*80)

created_orders.clear()

result = place_conditional_exit(
    exchange=mock_exchange,
    symbol="BNB/USDT",
    category="linear",
    exit_side="sell",
    amount=1.0,
    trigger_price=1150.0,
    is_tp=True,
    is_hedge=False,  # ONE-WAY mode
    position_idx=0,
)

if created_orders:
    order = created_orders[0]
    params = order['params']

    print(f"  TP order params: {params}")

    # For TP, triggerDirection should be 2
    if params.get('positionIdx') == 0 and params.get('triggerDirection') == 2:
        print("  [OK] TP параметры правильные (positionIdx=0, triggerDirection=2)")
    else:
        print(f"  [FAIL] positionIdx={params.get('positionIdx')}, triggerDirection={params.get('triggerDirection')}")
else:
    print("  [ERROR] TP order не был создан!")

print("\n" + "="*80)
print("ВЫВОД:")
print("="*80)
print("Для ONE-WAY mode conditional orders ДОЛЖНЫ содержать positionIdx=0")
print("Без positionIdx Bybit API может отклонить ордер молча!")
print("="*80)
