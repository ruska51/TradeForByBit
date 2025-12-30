"""Отмена всех открытых ордеров."""
import ccxt
import os

exchange = ccxt.bybit({
    'apiKey': os.getenv("BYBIT_API_KEY"),
    'secret': os.getenv("BYBIT_API_SECRET"),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'sandboxMode': True}
})
exchange.set_sandbox_mode(True)

print("=== CANCELLING ALL OPEN ORDERS ===")
try:
    orders = exchange.fetch_open_orders()
    print(f"Found {len(orders)} open orders\n")
    
    if not orders:
        print("No orders to cancel")
    
    for i, order in enumerate(orders, 1):
        symbol = order.get('symbol', 'UNKNOWN')
        order_id = order.get('id', 'UNKNOWN')
        order_type = order.get('type', 'UNKNOWN')
        side = order.get('side', 'UNKNOWN')
        
        print(f"[{i}/{len(orders)}] {symbol} {side} {order_type}")
        print(f"  ID: {order_id}")
        
        try:
            exchange.cancel_order(order_id, symbol)
            print(f"  ✅ Cancelled\n")
        except Exception as e:
            print(f"  ❌ {e}\n")
    
    print("=== DONE ===")
except Exception as e:
    print(f"FATAL: {e}")
    import traceback
    traceback.print_exc()
