#!/usr/bin/env python3
"""Test what CCXT actually sends to Bybit API."""

import sys
sys.path.insert(0, 'C:/Users/fishf/Documents/GitHub/TradeForByBit')

from credentials import API_KEY, API_SECRET
import ccxt

exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'swap'},
    'verbose': True,  # Enable verbose logging to see API calls
})

exchange.set_sandbox_mode(True)

print("Testing CCXT symbol format for API calls...")
print("="*80)

# Try to fetch ticker with CCXT format
symbol_ccxt = "XRP/USDT:USDT"

print(f"\nCalling fetch_ticker('{symbol_ccxt}')...")
print("Watch the verbose output to see what symbol CCXT sends to API\n")

try:
    ticker = exchange.fetch_ticker(symbol_ccxt)
    print(f"\nSuccess! Got ticker for {symbol_ccxt}")
    print(f"Last price: {ticker.get('last')}")
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "="*80)
print("Check the verbose output above to see if CCXT converts:")
print("  XRP/USDT:USDT -> XRPUSDT (for API)")
print("="*80)
