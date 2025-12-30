"""
Закрытие всех открытых позиций на Bybit testnet.
"""

import ccxt
import time
import os

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if not API_KEY or not API_SECRET:
    print("ERROR: BYBIT_API_KEY or BYBIT_API_SECRET not set!")
    print("Please set environment variables first.")
    exit(1)

exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # futures
        'sandboxMode': True,
    }
})

exchange.set_sandbox_mode(True)

def close_all_positions():
    print("=== CLOSING ALL POSITIONS ===\n")

    try:
        positions = exchange.fetch_positions()
        print(f"Total positions: {len(positions)}")

        open_positions = [p for p in positions if abs(float(p.get('contracts', 0))) > 0]
        print(f"Open positions: {len(open_positions)}\n")

        if not open_positions:
            print("✅ No open positions")
            return

        for i, pos in enumerate(open_positions, 1):
            symbol = pos['symbol']
            qty = float(pos['contracts'])
            side = pos['side']
            entry = pos.get('entryPrice', 0)
            pnl = pos.get('unrealizedPnl', 0)

            print(f"[{i}/{len(open_positions)}] {symbol}: {side} {qty} @ {entry}, P&L: {pnl}")

            try:
                close_side = 'sell' if qty > 0 else 'buy'
                close_qty = abs(qty)

                order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=close_qty,
                    params={'category': 'linear', 'reduceOnly': True}
                )

                print(f"  ✅ Closed!\n")
                time.sleep(1)

            except Exception as e:
                print(f"  ❌ Error: {e}\n")

        print("=== DONE ===")

    except Exception as e:
        print(f"FATAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    close_all_positions()
