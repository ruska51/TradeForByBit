"""Async helpers for ExchangeAdapter."""
import asyncio

from exchange_adapter import ExchangeAdapter
from main import ADAPTER  # type: ExchangeAdapter


async def fetch_ohlcv_async(symbol: str, timeframe: str, limit: int = 200):
    """Fetch OHLCV data asynchronously using ``ADAPTER``."""
    return await asyncio.to_thread(
        ADAPTER.fetch_ohlcv, symbol, timeframe, limit
    )  # via ExchangeAdapter


async def fetch_multi_ohlcv_async(symbol: str, timeframes: list[str], limit: int = 200):
    """Fetch several timeframes using ``ADAPTER.fetch_multi_ohlcv``."""
    return await asyncio.to_thread(
        ADAPTER.fetch_multi_ohlcv, symbol, timeframes, limit
    )  # via ExchangeAdapter
