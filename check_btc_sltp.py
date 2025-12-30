"""
Проверка наличия SL/TP для BTC позиции на бирже.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from exchange_adapter import ExchangeAdapter
from credentials import API_KEY, API_SECRET
import json

def main():
    print("=== CHECKING BTC POSITION SL/TP ===\n")

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

    symbol = "BTC/USDT:USDT"
    category = "linear"

    try:
        # Получаем позиции
        positions = exchange.fetch_positions([symbol], params={"category": category})

        print(f"Total positions fetched: {len(positions)}\n")

        # Ищем BTC позицию
        btc_pos = None
        for pos in positions:
            size = float(pos.get("contracts", 0) or pos.get("size", 0))
            if size > 0:
                btc_pos = pos
                break

        if not btc_pos:
            print("[X] No open BTC position found")
            return

        print("[OK] BTC Position Found:")
        print(f"  Symbol: {btc_pos.get('symbol')}")
        print(f"  Side: {btc_pos.get('side')}")
        print(f"  Contracts: {btc_pos.get('contracts')}")
        print(f"  Entry Price: {btc_pos.get('entryPrice')}")
        print(f"  Mark Price: {btc_pos.get('markPrice')}")
        print(f"  Unrealized PnL: {btc_pos.get('unrealizedPnl')}")
        print(f"  Leverage: {btc_pos.get('leverage')}")
        print()

        # Извлекаем info field
        info = btc_pos.get("info", {})

        # Bybit V5 поля для SL/TP
        tp_price = info.get("takeProfit") or info.get("tpPrice")
        sl_price = info.get("stopLoss") or info.get("slPrice")

        print("SL/TP Status:")
        print(f"  Take Profit: {tp_price}")
        print(f"  Stop Loss: {sl_price}")
        print()

        has_tp = tp_price is not None and str(tp_price) != "" and str(tp_price) != "0"
        has_sl = sl_price is not None and str(sl_price) != "" and str(sl_price) != "0"

        if has_tp and has_sl:
            print("[OK] RESULT: Both TP and SL are SET on exchange")
        elif has_tp:
            print("[!] RESULT: Only TP is set, SL is missing")
        elif has_sl:
            print("[!] RESULT: Only SL is set, TP is missing")
        else:
            print("[X] RESULT: Both TP and SL are MISSING on exchange")

        print("\n" + "="*50)
        print("Full 'info' field from exchange:")
        print("="*50)
        print(json.dumps(info, indent=2))

    except Exception as e:
        print(f"[X] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
