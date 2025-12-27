#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест подключения к Bybit с новыми настройками recvWindow
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import logging
import time
from exchange_adapter import ExchangeAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def test_exchange_connection():
    """Тест подключения к Bybit exchange"""

    print("\n" + "="*60)
    print("ТЕСТ ПОДКЛЮЧЕНИЯ К BYBIT")
    print("="*60)

    # Загружаем конфигурацию
    config = {
        "exchange_id": "bybit",
        "sandbox": True,
        "futures": True,
    }

    # Пытаемся загрузить API ключи из .env
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()

        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            print("[OK] API keys loaded from .env")
        else:
            print("[WARN] API keys not found in .env")
    except Exception as e:
        print(f"[WARN] Could not load .env: {e}")

    print(f"\nКонфигурация:")
    print(f"  Exchange: {config['exchange_id']}")
    print(f"  Sandbox: {config['sandbox']}")
    print(f"  Futures: {config['futures']}")

    # Create adapter
    print(f"\nCreating ExchangeAdapter...")
    try:
        adapter = ExchangeAdapter(config=config)
        print("[OK] Adapter created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create adapter: {e}")
        return False

    # Check that exchange is initialized
    if not adapter.x:
        print("[ERROR] Exchange object not created")
        return False

    print(f"[OK] Exchange ID: {adapter.x.id}")

    # Check recvWindow
    options = getattr(adapter.x, "options", {})
    recv_window = options.get("recvWindow", "NOT SET")
    print(f"\n  recvWindow: {recv_window}")

    if recv_window == 20000:
        print("  [OK] recvWindow set correctly (20000)")
    else:
        print(f"  [ERROR] recvWindow should be 20000, not {recv_window}")

    # Check loaded markets count
    markets = getattr(adapter.x, "markets", {})
    print(f"\n  Loaded markets: {len(markets)}")

    if len(markets) > 0:
        print("  [OK] Markets loaded successfully")
        # Show some examples
        sample_symbols = list(markets.keys())[:5]
        print(f"  Sample symbols: {', '.join(sample_symbols)}")
    else:
        print("  [WARN] No markets loaded")

    # Check server time
    print(f"\n  Time check:")
    try:
        local_time = int(time.time() * 1000)
        server_time = adapter.x.fetch_time()
        diff_ms = abs(server_time - local_time)
        diff_sec = diff_ms / 1000

        print(f"    Local:  {local_time}")
        print(f"    Server: {server_time}")
        print(f"    Difference: {diff_ms} ms ({diff_sec:.2f} sec)")

        if diff_sec < 10:
            print(f"    [OK] Time synchronized (diff < 10 sec)")
        else:
            print(f"    [WARN] Large time difference! May need sync")
    except Exception as e:
        print(f"    [ERROR] Failed to get server time: {e}")

    print("\n" + "="*60)
    print("ТЕСТ ЗАВЕРШЁН")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_exchange_connection()
    sys.exit(0 if success else 1)
