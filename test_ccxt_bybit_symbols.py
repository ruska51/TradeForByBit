#!/usr/bin/env python3
"""Test CCXT Bybit symbol formats."""

import ccxt

# Create Bybit testnet exchange
exchange = ccxt.bybit({
    'apiKey': 'dummy',
    'secret': 'dummy',
    'options': {
        'defaultType': 'swap',  # linear perpetual futures
    }
})

exchange.set_sandbox_mode(True)

# Load markets (skip currencies to avoid API auth errors)
print("Loading markets...")
try:
    markets = exchange.fetch_markets()
    exchange.markets = exchange.index_by(markets, 'symbol')
except Exception as e:
    print(f"Error loading markets: {e}")
    print("Using hardcoded symbol knowledge instead...")
    # We'll just print the expected formats
    print("\nExpected CCXT Bybit formats:")
    print("  Spot:             SOL/USDT")
    print("  Linear Perpetual: SOL/USDT:USDT")
    import sys
    sys.exit(0)

print(f"\nTotal markets: {len(exchange.markets)}")

# Check SOL formats
sol_formats = {}
for symbol in exchange.markets.keys():
    if 'SOL' in symbol and 'USDT' in symbol:
        market = exchange.markets[symbol]
        sol_formats[symbol] = {
            'type': market.get('type'),
            'linear': market.get('linear'),
            'spot': market.get('spot'),
            'swap': market.get('swap'),
            'future': market.get('future'),
        }

print("\nSOL/USDT related symbols:")
for symbol, info in sorted(sol_formats.items()):
    print(f"  {symbol:25} | {info}")

# The key insight:
print("\n" + "="*80)
print("CCXT Bybit Symbol Formats:")
print("="*80)
print("Spot:            SOL/USDT")
print("Linear Perpetual: SOL/USDT:USDT   (note the :USDT suffix!)")
print("="*80)
