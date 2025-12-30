"""
Проверка всех открытых позиций и их SL/TP на бирже.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from exchange_adapter import ExchangeAdapter
from credentials import API_KEY, API_SECRET
import json

def main():
    print("=== CHECKING ALL OPEN POSITIONS ===\n")

    # Инициализируем адаптер
    config = {
        "sandbox": True,
        "futures": True,
        "exchange_id": "bybit",
        "apiKey": API_KEY,
        "secret": API_SECRET,
    }
    adapter = ExchangeAdapter(config=config)
    exchange = adapter.x

    category = "linear"

    try:
        # Получаем все позиции
        positions = exchange.fetch_positions(params={"category": category})

        print(f"Total positions fetched: {len(positions)}\n")

        # Фильтруем открытые позиции
        open_positions = []
        for pos in positions:
            size = float(pos.get("contracts", 0) or pos.get("size", 0))
            if size > 0:
                open_positions.append(pos)

        if not open_positions:
            print("No open positions found")
            return

        print(f"Found {len(open_positions)} open position(s):\n")
        print("="*80)

        for i, pos in enumerate(open_positions, 1):
            symbol = pos.get('symbol')
            side = pos.get('side')
            contracts = pos.get('contracts')
            entry_price = pos.get('entryPrice')
            mark_price = pos.get('markPrice')
            pnl = pos.get('unrealizedPnl')
            leverage = pos.get('leverage')

            # Извлекаем info field
            info = pos.get("info", {})

            # Bybit V5 поля для SL/TP
            tp_price = info.get("takeProfit") or info.get("tpPrice")
            sl_price = info.get("stopLoss") or info.get("slPrice")

            has_tp = tp_price is not None and str(tp_price) != "" and str(tp_price) != "0"
            has_sl = sl_price is not None and str(sl_price) != "" and str(sl_price) != "0"

            print(f"Position #{i}: {symbol}")
            print(f"  Side: {side}")
            print(f"  Contracts: {contracts}")
            print(f"  Entry Price: {entry_price}")
            print(f"  Mark Price: {mark_price}")
            print(f"  Unrealized PnL: {pnl}")
            print(f"  Leverage: {leverage}")
            print(f"  Take Profit: {tp_price if has_tp else 'NOT SET'}")
            print(f"  Stop Loss: {sl_price if has_sl else 'NOT SET'}")

            if has_tp and has_sl:
                print(f"  Status: [OK] Both TP and SL are set")
            elif has_tp:
                print(f"  Status: [!] Only TP is set, SL missing")
            elif has_sl:
                print(f"  Status: [!] Only SL is set, TP missing")
            else:
                print(f"  Status: [X] Both TP and SL are MISSING")

            print("="*80)

    except Exception as e:
        print(f"[X] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
