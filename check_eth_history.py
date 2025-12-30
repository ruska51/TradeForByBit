#!/usr/bin/env python3
"""Check ETH/USDT trade and order history"""

import ccxt
import json
from datetime import datetime, timezone

# Initialize exchange
exchange = ccxt.bybit({
    'apiKey': 'xxxxxxxxxx',  # Will use from env or config
    'secret': 'xxxxxxxxxx',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'sandboxMode': True,  # TESTNET
    }
})

exchange.set_sandbox_mode(True)

try:
    # Fetch recent trades
    print("=== RECENT ETH/USDT:USDT TRADES ===\n")
    trades = exchange.fetch_my_trades('ETH/USDT:USDT', limit=10)

    for trade in trades[-5:]:
        dt = datetime.fromtimestamp(trade['timestamp']/1000, tz=timezone.utc)
        print(f"Time: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Side: {trade['side']}")
        print(f"  Price: ${trade['price']:.2f}")
        print(f"  Amount: {trade['amount']}")
        print(f"  Cost: ${trade['cost']:.2f}")
        print(f"  Fee: {trade.get('fee', {})}")
        print(f"  Order ID: {trade.get('order')}")
        print("-" * 80)

    # Fetch recent orders
    print("\n=== RECENT ETH/USDT:USDT ORDERS ===\n")
    orders = exchange.fetch_orders('ETH/USDT:USDT', limit=10)

    for order in orders[-5:]:
        dt = datetime.fromtimestamp(order['timestamp']/1000, tz=timezone.utc)
        print(f"Time: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Type: {order['type']}")
        print(f"  Side: {order['side']}")
        print(f"  Price: ${order.get('price', 0):.2f}")
        print(f"  Amount: {order['amount']}")
        print(f"  Filled: {order.get('filled', 0)}")
        print(f"  Status: {order['status']}")
        print(f"  Stop Loss: {order.get('stopLoss')}")
        print(f"  Take Profit: {order.get('takeProfit')}")
        print("-" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
