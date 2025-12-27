#!/usr/bin/env python3
"""Test what symbols the exchange actually has loaded."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from exchange_adapter import get_exchange

# Get exchange
ex = get_exchange()

# Check if SOLUSDT exists
sol_slash = 'SOL/USDT' in ex.markets
sol_no_slash = 'SOLUSDT' in ex.markets

print(f"Exchange markets loaded: {len(ex.markets)} symbols")
print(f"'SOL/USDT' in markets: {sol_slash}")
print(f"'SOLUSDT' in markets: {sol_no_slash}")

# Check what SOL-related symbols exist
sol_symbols = [s for s in ex.markets.keys() if 'SOL' in s and 'USDT' in s]
print(f"\nAll SOL/USDT symbols in markets:")
for s in sol_symbols[:10]:  # First 10
    market = ex.markets[s]
    market_type = market.get('type', 'unknown')
    linear = market.get('linear', False)
    spot = market.get('spot', False)
    print(f"  {s:20} | type={market_type:10} | linear={linear} | spot={spot}")
