"""Market scanning utilities for selecting tradeable symbols."""
from __future__ import annotations

import logging
from logging_utils import log
from typing import List

import ccxt
try:  # ccxt.pro is optional
    import ccxt.pro as ccxtpro
    HAS_PRO = True
except Exception:  # pragma: no cover - optional dependency
    ccxtpro = None
    HAS_PRO = False
import pandas as pd
import asyncio

from metrics_utils import backtest_metrics

from exchange_adapter import ExchangeAdapter

ADAPTER: ExchangeAdapter | None = None


def _get_adapter() -> ExchangeAdapter:
    global ADAPTER
    if ADAPTER is None:
        from main import ADAPTER as MAIN_ADAPTER  # type: ignore
        ADAPTER = MAIN_ADAPTER
    return ADAPTER


def _exchange_params(enable_rate_limit: bool = True) -> tuple[dict, bool]:
    adapter = _get_adapter()
    params: dict = {"enableRateLimit": enable_rate_limit}
    params["options"] = {"defaultType": "swap", "defaultSubType": "linear"}
    for key in ("apiKey", "secret"):
        value = adapter.config.get(key)
        if value:
            params[key] = value
    sandbox = getattr(adapter, "sandbox", False)
    return params, sandbox


def scan_markets(volume_threshold: float = 100_000,
                 timeframe: str = "1d",
                 limit: int = 90) -> List[str]:
    """Return a list of symbols that pass liquidity and performance filters."""
    params, sandbox = _exchange_params(enable_rate_limit=False)
    exchange = ccxt.bybit(params)
    if hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(sandbox)
    try:
        markets = exchange.fetch_markets()
    finally:
        if hasattr(exchange, "close"):
            exchange.close()

    pairs: List[str] = []
    usdt_markets = [m for m in markets if m.get("quote") == "USDT"]
    for m in usdt_markets:
        status = m.get("info", {}).get("status") or m.get("active")
        if status not in {"TRADING", True}:
            continue
        base = m.get("base", "")
        if base.endswith("UP") or base.endswith("DOWN"):
            continue
        quote_volume = 0.0
        if isinstance(m.get("info"), dict):
            qv = m["info"].get("quoteVolume") or m["info"].get("quoteVolume24h")
            if qv is not None:
                try:
                    quote_volume = float(qv)
                except Exception:
                    quote_volume = 0.0
        if quote_volume < volume_threshold:
            continue
        symbol = m.get("symbol")
        if not symbol or ":" in symbol:
            continue
        try:
            ohlcv = _get_adapter().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:  # pragma: no cover - network errors
            log(logging.ERROR, "scan", symbol, f"fetch failed: {e}")
            continue
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        roi = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        metrics = backtest_metrics(df["close"])
        sharpe = metrics["sharpe"]
        drawdown = metrics["max_drawdown"]
        log(logging.INFO, "scan", symbol,
            f"ROI={roi:.2%}, Sharpe={sharpe:.2f}, DD={drawdown:.2%}")
        if roi > 0.005 and sharpe > 0.3 and drawdown > -0.05:
            pairs.append(symbol)
    return pairs


async def scan_usdt_symbols(volume_threshold: float = 100_000,
                            timeframe: str = "1d",
                            limit: int = 90,
                            top_n: int | None = None) -> List[str]:
    """Asynchronously scan all USDT pairs and apply performance filters.

    If ``top_n`` is provided, only the best ``top_n`` pairs ranked by ROI are
    returned.
    """
    params, sandbox = _exchange_params()
    if HAS_PRO:
        exchange = ccxtpro.bybit(params)
        if hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(sandbox)
        markets = await exchange.fetch_markets()
    else:
        exchange = ccxt.bybit(params)
        if hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(sandbox)
        markets = await asyncio.to_thread(exchange.fetch_markets)

    usdt_markets = [m for m in markets if m.get("quote") == "USDT"]
    pairs_info: list[tuple[str, float]] = []
    semaphore = asyncio.Semaphore(20)
    tasks = []

    async def check_market(m):
        status = m.get("info", {}).get("status") or m.get("active")
        if status not in {"TRADING", True}:
            return
        base = m.get("base", "")
        if base.endswith("UP") or base.endswith("DOWN"):
            return
        quote_volume = 0.0
        if isinstance(m.get("info"), dict):
            qv = m["info"].get("quoteVolume") or m["info"].get("quoteVolume24h")
            if qv is not None:
                try:
                    quote_volume = float(qv)
                except Exception:
                    quote_volume = 0.0
        if quote_volume < volume_threshold:
            return
        symbol = m.get("symbol")
        if not symbol or ":" in symbol:
            return
        try:
            ohlcv = await asyncio.to_thread(
                _get_adapter().fetch_ohlcv, symbol, timeframe, limit
            )
        except Exception as e:
            log(logging.ERROR, "scan", symbol, f"fetch failed: {e}")
            return
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        roi = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        metrics = backtest_metrics(df["close"])
        sharpe = metrics["sharpe"]
        drawdown = metrics["max_drawdown"]
        log(logging.INFO, "scan", symbol,
            f"ROI={roi:.2%}, Sharpe={sharpe:.2f}, DD={drawdown:.2%}")
        if roi > 0.005 and sharpe > 0.3 and drawdown > -0.05:
            pairs_info.append((symbol, roi))

    try:
        for m in usdt_markets:
            async def wrapper(market=m):
                async with semaphore:
                    await check_market(market)
            tasks.append(asyncio.create_task(wrapper()))

        await asyncio.gather(*tasks)
    finally:
        if HAS_PRO:
            await exchange.close()
        else:
            if hasattr(exchange, "close"):
                await asyncio.to_thread(exchange.close)

    pairs_info.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        pairs_info = pairs_info[:top_n]
    pairs = [sym for sym, _ in pairs_info]
    return pairs


async def scan_symbols(min_volume: float = 300_000,
                       timeframe: str = "1h",
                       limit: int = 24,
                       top_n: int = 40,
                       min_volatility: float = 0.02) -> list[str]:
    """Return liquid USDT pairs passing a quick performance check.

    The ``min_volume`` threshold was relaxed and an explicit volatility filter
    was added to broaden the list of tradable instruments without sacrificing
    quality.  The widened default ``top_n`` of 40 allows scanning a broader
    universe of symbols while still returning only the best ``top_n`` pairs
    ranked by ROI.
    """
    params, sandbox = _exchange_params()
    if HAS_PRO:
        exchange = ccxtpro.bybit(params)
        if hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(sandbox)
        markets = await exchange.fetch_markets()
    else:
        exchange = ccxt.bybit(params)
        if hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(sandbox)
        markets = await asyncio.to_thread(exchange.fetch_markets)

    candidates: list[dict] = []
    for m in markets:
        if m.get("quote") != "USDT":
            continue
        status = m.get("info", {}).get("status") or m.get("active")
        if status not in {"TRADING", True}:
            continue
        if m.get("futureDelivery") or m.get("delivery") or m.get("expired"):
            continue
        candidates.append(m)

    pairs_info: list[tuple[str, float]] = []
    semaphore = asyncio.Semaphore(20)
    tasks = []

    async def check_symbol(m: dict) -> None:
        symbol = m.get("symbol")
        if not symbol or ":" in symbol:
            return
        if HAS_PRO:
            try:
                ticker = await exchange.fetch_ticker(symbol)
            except Exception as e:  # pragma: no cover - network errors
                log(logging.ERROR, "scan", symbol, f"ticker failed: {e}")
                return
        else:
            try:
                ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
            except Exception as e:  # pragma: no cover - network errors
                log(logging.ERROR, "scan", symbol, f"ticker failed: {e}")
                return

        qv = ticker.get("quoteVolume") or ticker.get("info", {}).get("quoteVolume")
        price = ticker.get("last")
        try:
            quote_volume = float(qv or 0)
        except Exception:
            quote_volume = 0.0
        try:
            last_price = float(price or 0)
        except Exception:
            last_price = 0.0
        if quote_volume < min_volume or last_price <= 0.001:
            return

        try:
            ohlcv = await asyncio.to_thread(
                _get_adapter().fetch_ohlcv, symbol, timeframe, limit
            )
        except Exception as e:  # pragma: no cover - network errors
            log(logging.ERROR, "scan", symbol, f"fetch failed: {e}")
            return

        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        returns = df["close"].pct_change(fill_method=None)
        vol = returns.std()
        roi = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        metrics = backtest_metrics(df["close"])
        sharpe = metrics["sharpe"]
        drawdown = metrics["max_drawdown"]
        log(
            logging.INFO,
            "scan",
            symbol,
            f"ROI={roi:.2%}, Sharpe={sharpe:.2f}, Drawdown={drawdown:.2%}, Vol={vol:.2%}",
        )
        if (
            roi > 0.005
            and sharpe > 0.3
            and drawdown > -0.05
            and vol >= min_volatility
        ):
            pairs_info.append((symbol, roi))

    try:
        for m in candidates:
            async def wrapper(market=m):
                async with semaphore:
                    await check_symbol(market)
            tasks.append(asyncio.create_task(wrapper()))

        await asyncio.gather(*tasks)
    finally:
        if HAS_PRO:
            await exchange.close()
        else:
            if hasattr(exchange, "close"):
                await asyncio.to_thread(exchange.close)

    pairs_info.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        pairs_info = pairs_info[:top_n]
    pairs = [sym for sym, _ in pairs_info]

    log(
        logging.INFO,
        "scan",
        "",
        f"✅ Selected {len(pairs)} pairs from {len(candidates)} available",
    )
    return pairs
