"""Utility functions for detailed trade decision logging and runtime logs."""
import csv
import math
import os
import logging
import time
import sys
from datetime import datetime, timezone
from collections import defaultdict
from threading import Lock
from logging.handlers import RotatingFileHandler
from pathlib import Path

from memory_utils import memory_manager

try:  # pragma: no cover - optional dependency for runtime environments
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - ccxt may be stubbed in tests
    ccxt = None  # type: ignore

LOG_DIR = Path(__file__).resolve().parent / "logs"

_LAST_LOGGED_MESSAGE: dict[str, float | str | None] = {"text": None, "ts": 0.0}


def log_once(logger, level: str, text: str, window: float = 5.0) -> None:
    """Emit *text* via *logger* at *level* unless repeated within *window* seconds."""

    try:
        now = time.time()
    except Exception:
        now = 0.0
    last_text = _LAST_LOGGED_MESSAGE.get("text")
    last_ts = float(_LAST_LOGGED_MESSAGE.get("ts") or 0.0)
    if last_text == text and now - last_ts < window:
        return
    _LAST_LOGGED_MESSAGE["text"] = text
    _LAST_LOGGED_MESSAGE["ts"] = now
    getattr(logger, level)(text)

# ``colorama`` is an optional dependency used only for colored console output.
# In minimal environments (such as some CI systems) it may be absent.  Import it
# with a graceful fallback that simply disables coloring so that the logging
# utilities remain functional even without the package installed.
try:  # pragma: no cover - import error path exercised only when missing
    from colorama import Fore, Style, init as colorama_init
except Exception:  # pragma: no cover - colorama optional
    class _NoColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = _NoColor()  # type: ignore
    Style = _NoColor()  # type: ignore

    def colorama_init(*_args, **_kwargs) -> None:  # type: ignore
        """Fallback no-op initializer when colorama is missing."""
        return None

colorama_init(autoreset=True)

MIN_NOTIONAL = getattr(sys.modules.get("main"), "MIN_NOTIONAL", 10.0)


def _is_bybit_exchange(exchange) -> bool:
    """Return ``True`` when *exchange* refers to Bybit."""

    ex_id = getattr(exchange, "id", None)
    if ex_id:
        ex_id = str(ex_id).lower()
    else:
        name = getattr(getattr(exchange, "__class__", None), "__name__", "")
        ex_id = str(name).lower()
    return "bybit" in ex_id


def _tick_info(market):
    """Return Bybit tick and lot size information from *market* metadata."""

    if not isinstance(market, dict):
        return 0.0, 0.0
    precision = market.get("precision") or {}
    info = market.get("info") or {}
    try:
        tick = float(precision.get("price", 0) or 0)
    except Exception:
        tick = 0.0
    if tick <= 0:
        tick_raw = ((info.get("priceFilter") or {}).get("tickSize")) if isinstance(info, dict) else 0
        try:
            tick = float(tick_raw or 0)
        except Exception:
            tick = 0.0
    try:
        step = float(precision.get("amount", 0) or 0)
    except Exception:
        step = 0.0
    if step <= 0 and isinstance(info, dict):
        lot_info = info.get("lotSizeFilter") or {}
        try:
            step = float(lot_info.get("qtyStep", 0) or 0)
        except Exception:
            step = 0.0
    return tick, step


def _price_qty_to_precision(ex, symbol, price=None, amount=None):
    """Return *price* and *amount* adjusted to exchange precision."""

    market = None
    try:
        market = ex.market(symbol)
    except Exception:
        market = None
    if price is not None:
        try:
            price = float(ex.price_to_precision(symbol, price))
        except Exception:
            price = float(price)
    if amount is not None:
        try:
            amount = float(ex.amount_to_precision(symbol, amount))
        except Exception:
            amount = float(amount)
    return price, amount


def _min_step_qty(ex, symbol: str) -> tuple[float, float]:
    """Return market ``(min_qty, step_qty)``; ``0.0`` when unavailable."""

    try:
        market = ex.market(symbol)
    except Exception:
        market = None

    info = (market or {}).get("info") or {}
    lot = (info.get("lotSizeFilter") or {}) if isinstance(info, dict) else {}

    try:
        min_qty = float(lot.get("minOrderQty") or lot.get("minQty") or 0)
    except Exception:
        min_qty = 0.0
    try:
        step = float(lot.get("qtyStep") or 0)
    except Exception:
        step = 0.0

    if step <= 0:
        precision = (market or {}).get("precision") or {}
        try:
            step = float(precision.get("amount") or 0)
        except Exception:
            step = 0.0

    return (max(min_qty, 0.0), max(step, 0.0))


def _round_qty(ex, symbol: str, qty: float) -> float:
    """Return *qty* rounded to precision and raised to ``min_qty`` when needed."""

    _, adjusted = _price_qty_to_precision(ex, symbol, price=None, amount=qty)
    min_qty, _step = _min_step_qty(ex, symbol)
    if min_qty and (adjusted or 0) < min_qty:
        adjusted = min_qty
    _, adjusted = _price_qty_to_precision(ex, symbol, price=None, amount=adjusted)
    return float(adjusted or 0.0)


def _clean_params(d: dict | None) -> dict | None:
    """Return a copy of *d* without ``None`` or blank-string values."""

    if d is None:
        return None
    out: dict = {}
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out


def enter_ensure_filled(
    ex,
    symbol: str,
    side: str,
    qty: float,
    *,
    category: str = "linear",
    prefer_limit: bool = True,
    slip_pct: float = 0.001,
    wait_fill_sec: float = 2.0,
) -> tuple[str | None, float]:
    """Place an IOC limit (or market) entry and ensure it fills."""

    norm = _normalize_bybit_symbol(ex, symbol, category)
    try:
        ticker = ex.fetch_ticker(norm)
    except Exception:
        ticker = {}

    last = float(
        (ticker or {}).get("last")
        or (ticker or {}).get("ask")
        or (ticker or {}).get("bid")
        or (ticker or {}).get("close")
        or 0.0
    )
    if last <= 0 and isinstance(ticker, dict):
        info = ticker.get("info") or {}
        try:
            last = float(
                info.get("lastPrice")
                or info.get("markPrice")
                or info.get("indexPrice")
                or info.get("price")
                or 0.0
            )
        except Exception:
            last = 0.0
    if last <= 0:
        last = 1.0

    price = last * (1 + slip_pct if side.lower() == "buy" else 1 - slip_pct)
    price, qty = _price_qty_to_precision(ex, norm, price=price, amount=qty)
    qty = _round_qty(ex, norm, float(qty or 0.0))
    if qty <= 0:
        return None, 0.0

    params = {"category": category, "reduceOnly": False, "timeInForce": "IOC"}
    order_type = "limit" if prefer_limit else "market"
    price_arg = price if order_type == "limit" else None
    order = ex.create_order(norm, order_type, side, qty, price_arg, params)
    oid = order.get("id") if isinstance(order, dict) else None
    filled = float(order.get("filled") or 0) if isinstance(order, dict) else 0.0
    t0 = time.time()
    while time.time() - t0 < wait_fill_sec and filled < qty:
        time.sleep(0.15)
        try:
            state = ex.fetch_order(oid, norm, params={"category": category}) if oid else None
        except Exception:
            state = None
        if isinstance(state, dict):
            try:
                filled = float(state.get("filled") or filled)
            except Exception:
                pass
            status = str(state.get("status") or state.get("info", {}).get("orderStatus") or "").lower()
            if status in {"canceled", "cancelled", "closed", "filled"}:
                break

    remainder = max(qty - filled, 0.0)
    _, remainder = _price_qty_to_precision(ex, norm, price=None, amount=remainder)
    remainder = _round_qty(ex, norm, float(remainder or 0.0))
    if remainder > 0:
        try:
            if oid:
                ex.cancel_order(oid, norm, {"category": category})
        except Exception:
            pass
        ex.create_order(norm, "market", side, remainder, None, {"category": category, "reduceOnly": False})
        filled = qty
    return oid, float(filled)


def _bybit_trigger_for_exit(
    side_open: str,
    last: float,
    base_price: float,
    pct: float,
    *,
    is_tp: bool,
) -> tuple[float, str, str]:
    min_off = 0.001  # 0.1 %
    if side_open == 'buy':  # long
        if is_tp:
            trig = max(base_price * (1 + pct), last * (1 + min_off))
            direction = 'ascending'
            side_to_send = 'sell'
        else:
            trig = min(base_price * (1 - pct), last * (1 - min_off))
            direction = 'descending'
            side_to_send = 'sell'
    else:  # short
        if is_tp:
            trig = min(base_price * (1 - pct), last * (1 - min_off))
            direction = 'descending'
            side_to_send = 'buy'
        else:
            trig = max(base_price * (1 + pct), last * (1 + min_off))
            direction = 'ascending'
            side_to_send = 'buy'
    return trig, direction, side_to_send


def normalize_bybit_category(value: str | None) -> str | None:
    """Return canonical Bybit category name for *value*.

    Bybit's unified v5 API frequently reports categories such as
    ``LinearPerpetual`` or ``optionU`` via the CCXT metadata.  Passing those
    raw strings back to the order endpoints leads to ``Illegal category``
    errors.  The helper collapses the various aliases to the small set of
    values accepted by the API.
    """

    if not isinstance(value, str):
        return None

    lowered = value.strip().lower()
    if not lowered:
        return None

    explicit = {
        "spot": "spot",
        "linear": "linear",
        "inverse": "inverse",
        "option": "option",
        "options": "option",
        "swap": "swap",
        "future": "swap",
        "futures": "swap",
    }
    if lowered in explicit:
        return explicit[lowered]

    if "linear" in lowered or "usdt" in lowered or "perpetual" in lowered:
        return "linear"
    if "inverse" in lowered or "coin" in lowered:
        return "inverse"
    if "option" in lowered:
        return "option"
    if "spot" in lowered:
        return "spot"
    if "swap" in lowered or "future" in lowered:
        return "swap"

    return None


def _market_category_from_meta(market: dict | None) -> str | None:
    """Return Bybit-style market category from a CCXT market description."""

    if not isinstance(market, dict):
        return None

    info = market.get("info") or {}
    if isinstance(info, dict):
        for key in ("category", "contractType", "productType", "market"):
            value = normalize_bybit_category(info.get(key))
            if value:
                return value

    market_type = normalize_bybit_category(market.get("type"))
    if market_type:
        if market_type == "spot":
            return "spot"
        if market_type == "option":
            return "option"
        if market_type == "linear":
            return "linear"
        if market_type == "inverse":
            return "inverse"
        if market_type == "swap":
            settle = str(market.get("settle") or market.get("settleId") or "").upper()
            quote = str(market.get("quote") or "").upper()
            if settle and quote and settle != quote:
                return "inverse"
            return "linear"

    market_type = str(market.get("type") or "").lower()
    if market.get("spot") or market_type == "spot":
        return "spot"
    if market.get("option") or market_type == "option":
        return "option"
    if market.get("inverse"):
        return "inverse"
    if market.get("linear"):
        return "linear"
    if market.get("swap"):
        settle = str(market.get("settle") or market.get("settleId") or "").upper()
        quote = str(market.get("quote") or "").upper()
        if settle and quote and settle != quote:
            return "inverse"
        return "linear"
    if market_type in {"swap", "future"}:
        settle = str(market.get("settle") or market.get("settleId") or "").upper()
        quote = str(market.get("quote") or "").upper()
        if settle and quote and settle != quote:
            return "inverse"
        if settle or quote:
            return "linear"
    return None


def detect_market_category(exchange, symbol: str) -> str | None:
    """Best-effort detection of the market category for ``symbol``.

    The helper mirrors :meth:`ExchangeAdapter._detect_bybit_category` so the
    trading utilities can consistently decide whether an instrument is spot or
    derivative even when the adapter itself is bypassed (``safe_create_order``
    works directly with the underlying CCXT exchange instance).
    """

    if not symbol:
        return None

    markets_loaded = getattr(exchange, "markets", None)
    if not markets_loaded:
        loader = getattr(exchange, "load_markets", None)
        if callable(loader):
            try:  # pragma: no branch - network interaction only exercised at runtime
                loader()
            except Exception:
                pass

    def _explicit_spot_meta(meta: dict | None) -> bool:
        if not isinstance(meta, dict):
            return False

        if meta.get("spot") is True:
            return True

        market_type = str(meta.get("type") or "").lower()
        if market_type == "spot":
            return True

        info = meta.get("info")
        if isinstance(info, dict):
            for key in ("category", "contractType", "productType", "market"):
                value = info.get(key)
                if isinstance(value, str) and value.lower() == "spot":
                    return True

        return False

    normalized_symbol = _normalize_bybit_symbol(exchange, symbol, None)
    search_symbols: list[str] = []
    for candidate in (normalized_symbol, symbol):
        if isinstance(candidate, str) and candidate and candidate not in search_symbols:
            search_symbols.append(candidate)

    market = None
    for candidate in search_symbols:
        try:  # pragma: no cover - depends on CCXT metadata availability
            market = exchange.market(candidate)
            if market:
                break
        except Exception:
            market = None

    lookup_symbol = search_symbols[0] if search_symbols else symbol

    category = _market_category_from_meta(market)
    spot_confident = False
    if category:
        if category == "spot":
            spot_confident = _explicit_spot_meta(market)
        else:
            return category

    markets = getattr(exchange, "markets", {}) or {}
    if lookup_symbol in markets:
        category = _market_category_from_meta(markets.get(lookup_symbol))
        if category:
            if category == "spot":
                spot_confident = spot_confident or _explicit_spot_meta(
                    markets.get(lookup_symbol)
                )
            else:
                return category
    if lookup_symbol != symbol and symbol in markets:
        category = _market_category_from_meta(markets.get(symbol))
        if category:
            if category == "spot":
                spot_confident = spot_confident or _explicit_spot_meta(markets.get(symbol))
            else:
                return category

    base = quote = None
    if isinstance(market, dict):
        base = market.get("base")
        quote = market.get("quote")
    if (not base or not quote) and "/" in lookup_symbol:
        base, quote = lookup_symbol.split("/", 1)
    if (not base or not quote) and "/" in symbol:
        base, quote = symbol.split("/", 1)

    base = str(base or "").upper()
    quote = str(quote or "").upper()

    if markets and base and quote:
        for meta in markets.values():
            if not isinstance(meta, dict):
                continue
            meta_base = str(meta.get("base") or "").upper()
            meta_quote = str(meta.get("quote") or "").upper()
            if not meta_base or not meta_quote:
                mapped = meta.get("symbol") or meta.get("id")
                if isinstance(mapped, str) and "/" in mapped:
                    meta_base, meta_quote = mapped.split("/", 1)
                    meta_base = meta_base.upper()
                    meta_quote = meta_quote.split(":", 1)[0].upper()
            if meta_base != base or meta_quote != quote:
                continue
            candidate_category = _market_category_from_meta(meta)
            if candidate_category:
                if candidate_category == "spot":
                    spot_confident = spot_confident or _explicit_spot_meta(meta)
                else:
                    return candidate_category

    category = None

    if lookup_symbol in markets:
        category = _market_category_from_meta(markets.get(lookup_symbol))
        if category == "spot":
            spot_confident = spot_confident or _explicit_spot_meta(markets.get(lookup_symbol))
    if (not category or category == "spot") and lookup_symbol != symbol and symbol in markets:
        other_category = _market_category_from_meta(markets.get(symbol))
        if other_category:
            if other_category == "spot":
                spot_confident = spot_confident or _explicit_spot_meta(markets.get(symbol))
            else:
                return other_category
        if not category:
            category = other_category

    if category == "spot":
        def _has_linear_params(obj) -> bool:
            params = getattr(obj, "params", None)
            if isinstance(params, dict):
                cat = params.get("category") or params.get("categoryType")
                if isinstance(cat, str) and cat.lower() == "linear":
                    return True
            return False

        futures_hint = bool(getattr(exchange, "futures", False)) or _has_linear_params(
            exchange
        )
        adapter = getattr(exchange, "adapter", None)
        if adapter is not None:
            futures_hint = futures_hint or bool(getattr(adapter, "futures", False))
            futures_hint = futures_hint or _has_linear_params(adapter)
        if not futures_hint:
            main_mod = sys.modules.get("main")
            main_adapter = getattr(main_mod, "ADAPTER", None) if main_mod else None
            if main_adapter and getattr(main_adapter, "x", None) is exchange:
                futures_hint = bool(getattr(main_adapter, "futures", False))
                futures_hint = futures_hint or _has_linear_params(main_adapter)
        derivative_category: str | None = None
        if futures_hint and markets and base and quote:
            for meta in markets.values():
                if not isinstance(meta, dict):
                    continue
                meta_base = str(meta.get("base") or "").upper()
                meta_quote = str(meta.get("quote") or "").upper()
                mapped_symbol = meta.get("symbol") or meta.get("id")
                if (not meta_base or not meta_quote) and isinstance(mapped_symbol, str) and "/" in mapped_symbol:
                    meta_base, meta_quote = mapped_symbol.split("/", 1)
                    meta_base = meta_base.upper()
                    meta_quote = meta_quote.split(":", 1)[0].upper()
                if meta_base != base or meta_quote != quote:
                    continue
                meta_category = _market_category_from_meta(meta)
                if meta_category in {"linear", "inverse"}:
                    derivative_category = meta_category
                    break
                if meta_category == "swap":
                    derivative_category = "linear"
                    break

        if derivative_category:
            return derivative_category

        if spot_confident or not futures_hint:
            return "spot"

        return "spot"

    return category


def _normalize_bybit_symbol(exchange, symbol: str, category: str | None) -> str:
    """Return CCXT symbol matching the requested Bybit ``category``."""

    if not _is_bybit_exchange(exchange):
        return symbol

    category_norm = str(category or "").lower()
    markets = getattr(exchange, "markets", {}) or {}
    markets_by_id = getattr(exchange, "markets_by_id", {}) or {}

    def _derive_contract_symbol() -> str | None:
        if ":" in symbol:
            return None
        if "/" not in symbol:
            return None
        if category_norm not in {"linear", "inverse"}:
            return None
        base_raw, quote_raw = symbol.split("/", 1)
        base_raw = base_raw.upper()
        quote_raw = quote_raw.split(":", 1)[0].upper()
        settle = quote_raw if category_norm == "linear" else base_raw
        candidate = f"{base_raw}/{quote_raw}:{settle}"
        if not markets and not markets_by_id:
            return candidate
        if candidate in markets or candidate in markets_by_id:
            return candidate
        return None

    derived_symbol = _derive_contract_symbol()
    if derived_symbol is not None:
        return derived_symbol

    def _dict_prefers_linear(data: dict | None, *, _depth: int = 0) -> bool:
        if not isinstance(data, dict) or _depth > 2:
            return False

        keywords = {"linear", "swap", "future", "futures", "contract"}
        for key in (
            "category",
            "categoryType",
            "defaultType",
            "defaultSubType",
            "type",
            "subType",
            "market",
            "contractType",
            "productType",
            "settle",
            "defaultSettle",
        ):
            value = data.get(key)
            if isinstance(value, str) and value:
                lowered = value.lower()
                if lowered in keywords or any(word in lowered for word in keywords):
                    return True

        for key in ("linear", "swap"):
            if data.get(key) is True:
                return True

        for nested in data.values():
            if isinstance(nested, dict) and _dict_prefers_linear(nested, _depth=_depth + 1):
                return True

        return False

    def _prefers_linear_market() -> bool:
        def _prefers_from_obj(obj) -> bool:
            if obj is None:
                return False
            if getattr(obj, "futures", False):
                return True
            for attr in ("params", "options", "config"):
                if _dict_prefers_linear(getattr(obj, attr, None)):
                    return True
            return False

        if _prefers_from_obj(exchange):
            return True

        adapter = getattr(exchange, "adapter", None)
        if _prefers_from_obj(adapter):
            return True

        main_mod = sys.modules.get("main")
        if main_mod is not None:
            main_adapter = getattr(main_mod, "ADAPTER", None)
            if getattr(main_adapter, "x", None) is exchange and _prefers_from_obj(main_adapter):
                return True

        return False

    prefers_linear = category_norm not in {"spot", "inverse"} and _prefers_linear_market()

    def _extract_base_quote(meta: dict | None) -> tuple[str, str]:
        if not isinstance(meta, dict):
            return "", ""
        base_raw = str(meta.get("base") or "").upper()
        quote_raw = str(meta.get("quote") or "").upper()
        mapped = meta.get("symbol") or meta.get("id")
        if (not base_raw or not quote_raw) and isinstance(mapped, str) and "/" in mapped:
            base_raw, quote_part = mapped.split("/", 1)
            base_raw = base_raw.upper()
            quote_raw = quote_part.split(":", 1)[0].upper()
        return base_raw, quote_raw

    def _mapped_symbol(meta: dict | None) -> str | None:
        if not isinstance(meta, dict):
            return None
        mapped = meta.get("symbol") or meta.get("id")
        if isinstance(mapped, str) and mapped:
            return mapped
        return None

    def _match_with_category(meta: dict | None) -> str | None:
        if not isinstance(meta, dict):
            return None
        meta_category = _market_category_from_meta(meta)
        if category_norm == "linear":
            if (
                meta.get("linear")
                or meta.get("swap")
                or meta_category in {"linear", "swap"}
            ):
                return _mapped_symbol(meta)
        elif category_norm == "inverse":
            if meta.get("inverse") or meta_category == "inverse":
                return _mapped_symbol(meta)
        elif category_norm == meta_category:
            return _mapped_symbol(meta)
        return None

    try:  # pragma: no cover - relies on CCXT metadata
        current = exchange.market(symbol)
    except Exception:
        current = markets.get(symbol) or markets_by_id.get(symbol)

    if not category_norm or category_norm == "spot":
        mapped = _mapped_symbol(current)
        if mapped and not prefers_linear:
            return mapped
        if symbol in markets:
            mapped = _mapped_symbol(markets.get(symbol))
            if mapped and not prefers_linear:
                return mapped

        base, quote = _extract_base_quote(current)
        if (not base or not quote) and "/" in symbol:
            base_raw, quote_raw = symbol.split("/", 1)
            base = base or base_raw.upper()
            quote = quote or quote_raw.split(":", 1)[0].upper()

        preferred_linear = None
        preferred_spot = None
        fallback_symbol = None

        for meta in markets.values():
            if not isinstance(meta, dict):
                continue
            if base and quote:
                meta_base, meta_quote = _extract_base_quote(meta)
                if meta_base and meta_quote and (
                    meta_base != base or meta_quote != quote
                ):
                    continue
            mapped_meta = _mapped_symbol(meta)
            if not mapped_meta:
                continue
            meta_category = _market_category_from_meta(meta)
            if prefers_linear:
                if (
                    meta.get("linear")
                    or meta.get("swap")
                    or meta_category in {"linear", "swap"}
                ) and preferred_linear is None:
                    preferred_linear = mapped_meta
                    if meta_category == "linear":
                        break
                if preferred_linear and meta_category in {"linear", "swap"}:
                    continue
            if category_norm == "spot":
                if meta_category == "spot" or (
                    meta_category is None and meta.get("spot") is not False
                ):
                    return mapped_meta
                continue
            if meta_category in {None, "", "spot"} and preferred_spot is None:
                preferred_spot = mapped_meta
                if meta_category == "spot" and not prefers_linear:
                    break
            elif fallback_symbol is None:
                fallback_symbol = mapped_meta

        if category_norm == "spot":
            return preferred_spot or symbol
        if prefers_linear and preferred_linear:
            return preferred_linear
        return preferred_spot or fallback_symbol or symbol

    mapped = _match_with_category(current)
    if mapped:
        return mapped

    base = quote = ""
    if isinstance(current, dict):
        base, quote = _extract_base_quote(current)
    if (not base or not quote) and "/" in symbol:
        base_raw, quote_raw = symbol.split("/", 1)
        base = base or base_raw.upper()
        quote = quote or quote_raw.split(":", 1)[0].upper()

    if markets and base and quote:
        for meta in markets.values():
            if not isinstance(meta, dict):
                continue
            meta_base, meta_quote = _extract_base_quote(meta)
            if not meta_base or not meta_quote:
                continue
            if meta_base != base or meta_quote != quote:
                continue
            mapped = _match_with_category(meta)
            if mapped:
                return mapped

    if "/" in symbol:
        base_raw, quote_raw = symbol.split("/", 1)
        candidates = [
            f"{base_raw}{quote_raw}",
            f"{base_raw}{quote_raw}:{quote_raw}",
            f"{base_raw}/{quote_raw}:{quote_raw}",
        ]
        for cand in candidates:
            meta = markets.get(cand) or markets_by_id.get(cand)
            mapped = _match_with_category(meta)
            if mapped:
                return mapped

    return symbol


def has_open_position(ex, symbol: str, category: str = "linear") -> tuple[float, float]:
    """Return the signed and absolute quantity of the open position."""

    norm = _normalize_bybit_symbol(ex, symbol, category)
    base_norm = norm.replace(":USDT", "").replace(":USDC", "")
    try:
        pos_list = ex.fetch_positions([norm], params={"category": category})
    except Exception:
        try:
            pos_list = ex.fetch_positions(params={"category": category})
        except Exception:
            pos_list = []
    qty_signed = 0.0
    for p in pos_list or []:
        sym = str(p.get("symbol") or p.get("info", {}).get("symbol") or "")
        sym_base = sym.replace(":USDT", "").replace(":USDC", "")
        if sym != norm and sym_base != base_norm:
            continue
        q = (
            p.get("contracts")
            or p.get("info", {}).get("size")
            or p.get("info", {}).get("positionAmt")
            or 0
        )
        try:
            q = float(q)
        except Exception:
            q = 0.0
        side = (p.get("side") or p.get("info", {}).get("side") or "").lower()
        if side in ("short", "sell"):
            q = -abs(q)
        elif side in ("long", "buy"):
            q = abs(q)
        qty_signed += q
    _, qty_abs = _price_qty_to_precision(ex, norm, price=None, amount=abs(qty_signed))
    qty_signed = qty_signed if qty_signed >= 0 else -qty_abs
    return float(qty_signed), float(qty_abs)


def _get_position_size(exchange, symbol: str, category: str = "linear") -> float:
    """Return absolute size of the open position for ``symbol`` in base units."""

    _, qty_abs = has_open_position(exchange, symbol, category)
    return float(qty_abs)


def _with_bybit_order_params(
    exchange, symbol: str, params: dict | None
) -> tuple[dict | None, str | None]:
    """Inject Bybit specific parameters when required.

    Recent versions of the Bybit API require the ``category`` argument for
    every order management request.  Without it orders are rejected with
    ``Param error!``, preventing the bot from opening positions.  The helper
    keeps the behaviour unchanged for other exchanges while transparently
    adding the missing parameter for Bybit.
    """

    if not _is_bybit_exchange(exchange):
        return params, None

    merged = dict(params or {})
    resolved_category = str(merged.get("category") or "").lower()

    if not resolved_category:
        detected = detect_market_category(exchange, symbol)
        if detected:
            resolved_category = str(detected).lower()
        else:
            markets = getattr(exchange, "markets", {}) or {}
            normalized_symbol = _normalize_bybit_symbol(exchange, symbol, None)
            market_category = None

            for candidate in (symbol, normalized_symbol):
                if not isinstance(candidate, str) or not candidate:
                    continue
                meta = markets.get(candidate)
                if not isinstance(meta, dict):
                    continue
                candidate_category = _market_category_from_meta(meta)
                if not candidate_category:
                    continue
                if candidate_category == "swap":
                    market_category = "linear"
                else:
                    market_category = candidate_category
                break

            if not market_category and markets:
                derivative_symbols: set[str] = set()
                for meta in markets.values():
                    if not isinstance(meta, dict):
                        continue
                    meta_category = _market_category_from_meta(meta)
                    if meta_category in {"linear", "inverse", "swap"}:
                        mapped = meta.get("symbol") or meta.get("id")
                        if isinstance(mapped, str) and mapped:
                            derivative_symbols.add(mapped)
                if any(
                    isinstance(candidate, str)
                    and candidate
                    and candidate in derivative_symbols
                    for candidate in (symbol, normalized_symbol)
                ):
                    market_category = "linear"

            resolved_category = market_category or "spot"

    if resolved_category == "swap":
        resolved_category = "linear"

    derivative_hint = False
    for flag in ("reduceOnly", "closeOnTrigger", "closePosition"):
        if merged.get(flag) is True:
            derivative_hint = True
            break
    if not derivative_hint:
        derivative_hint = any(key in merged for key in ("slOrderType", "tpOrderType"))

    if resolved_category == "spot" and derivative_hint:
        resolved_category = "linear"

    if resolved_category:
        merged["category"] = resolved_category
    else:
        resolved_category = "linear"
        merged["category"] = resolved_category

    if resolved_category in {"linear", "inverse"}:
        # ``positionIdx`` defaults to 0 (one-way mode) which matches the bot
        # assumptions but some unified accounts require the field to be set
        # explicitly for conditional orders.
        merged.setdefault("positionIdx", merged.get("positionIdx", 0))
    else:
        # Avoid including derivatives-only parameters for spot / options.
        merged.pop("positionIdx", None)

    if derivative_hint and merged.get("tpSlMode") in (None, ""):
        merged.setdefault("tpSlMode", "Full")

    if merged.get("slOrderType") and not (
        merged.get("stopLoss") or merged.get("takeProfit")
    ):
        merged.pop("slOrderType", None)

    return merged, resolved_category


def _bybit_tpsl_params(
    *,
    category: str = "linear",
    stop_loss: float | None = None,
    take_profit: float | None = None,
    sl_order_type: str | None = None,
    tpsl_mode: str = "Full",
    tpsl_trigger_by: str = "MarkPrice",
    trigger_direction: str | int | None = None,
    extra: dict | None = None,
) -> dict:
    """Return Bybit specific parameters for take-profit / stop-loss orders."""

    params: dict = dict(extra or {})
    params["category"] = category
    if (stop_loss is not None) or (take_profit is not None) or (sl_order_type is not None):
        params["tpSlMode"] = tpsl_mode
        params["tpslTriggerBy"] = tpsl_trigger_by
        if take_profit is not None:
            params["takeProfit"] = float(take_profit)
        if stop_loss is not None:
            params["stopLoss"] = float(stop_loss)
            if sl_order_type:
                params["slOrderType"] = sl_order_type
        if trigger_direction:
            td = trigger_direction
            if isinstance(td, str):
                low = td.strip().lower()
                if low in ("ascending", "rising", "up"):
                    td = 1
                elif low in ("descending", "falling", "down"):
                    td = 2
                else:
                    try:
                        td = int(low)
                    except Exception:
                        td = None
            try:
                if td is not None:
                    params["triggerDirection"] = int(td)
            except Exception:
                pass
    return params


def _normalize_bybit_order_type_value(order_type: str) -> str:
    """Translate high level order types to Bybit supported values."""

    normalized = str(order_type or "").lower()
    mapping = {
        "stop_market": "market",
        "take_profit_market": "market",
        "stop_limit": "limit",
        "take_profit_limit": "limit",
        "stop": "market",
        "take_profit": "market",
    }
    return mapping.get(normalized, normalized)


def _normalize_balance_params(exchange, params: dict | None) -> dict | None:
    """Translate generic balance parameters to exchange specific ones."""

    if not _is_bybit_exchange(exchange):
        return params

    adjusted = dict(params or {})
    typ = str(adjusted.pop("type", "")).lower()
    futures_aliases = {"future", "futures", "contract", "swap", "linear"}

    # Unified accounts are now the default mode on Bybit testnet and
    # production.  Explicitly request the unified balance to avoid the API
    # error ``accountType only support UNIFIED`` that otherwise prevents the
    # bot from trading when it attempts to read the account balance.
    if typ in futures_aliases:
        adjusted.setdefault("accountType", "UNIFIED")
    elif typ:
        adjusted.setdefault("accountType", typ.upper())
    else:
        adjusted.setdefault("accountType", "UNIFIED")

    return adjusted


def setup_logging(level: int = logging.INFO, to_console: bool = True) -> None:
    """Configure rotating file and optional console logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(message)s"
    )

    app = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    app.setLevel(level)
    app.setFormatter(fmt)

    err = RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    err.setLevel(logging.WARNING)
    err.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        # Preserve pytest's LogCaptureHandler to avoid breaking caplog-based tests.
        handler_name = h.__class__.__name__
        if handler_name == "LogCaptureHandler" or hasattr(h, "records"):
            continue
        root.removeHandler(h)

    root.addHandler(app)
    root.addHandler(err)

    if to_console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    logging.getLogger("ccxt.base.exchange").setLevel(logging.WARNING)

# [ANCHOR:TRADES_CSV_HEADER]
TRADES_CSV_HEADER = [
    "trade_id",
    "timestamp_entry",
    "symbol",
    "side",
    "entry_price",
    "sl",
    "tp",
    "qty",
    "reason",
    "source",
    "reduced_risk",
    "timestamp_exit",
    "exit_price",
    "profit",
    "pnl_net",
    "exit_type",
    "trail_triggered",
    "time_stop_triggered",
    "duration_sec",
]

# [ANCHOR:LOG_ENTRY_FIELDS]
LOG_ENTRY_FIELDS = [
    "trade_id",
    "timestamp_entry",
    "symbol",
    "side",
    "entry_price",
    "sl",
    "tp",
    "qty",
    "reason",
    "source",
    "reduced_risk",
]

# [ANCHOR:LOG_EXIT_FIELDS]
LOG_EXIT_FIELDS = [
    "trade_id",
    "timestamp_close",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "qty",
    "profit",
    "profit_pct",
    "exit_type",
    "entry_time",
    "duration_min",
    "stop_loss_triggered",
    "take_profit_triggered",
    "trailing_profit_used",
    "source",
    "reduced_risk",
    "order_id",
]


def _rotate_legacy_file(path_obj: Path, label: str = "legacy") -> Path:
    """Rename existing file to a ``*_{label}`` variant avoiding overwrites."""

    backup = path_obj.with_name(f"{path_obj.stem}_{label}{path_obj.suffix}")
    counter = 1
    while backup.exists():
        backup = path_obj.with_name(
            f"{path_obj.stem}_{label}{counter}{path_obj.suffix}"
        )
        counter += 1
    path_obj.rename(backup)
    return backup


def ensure_trades_csv_header(path: str) -> None:
    """Ensure ``path`` exists with ``TRADES_CSV_HEADER`` columns.

    If the file already exists, any missing columns from ``TRADES_CSV_HEADER``
    are appended to the header while keeping the existing column order.
    Existing data rows are left untouched which preserves backward
    compatibility with older logs.
    """

    header = TRADES_CSV_HEADER
    path_obj = Path(path)

    if path_obj.exists():
        with open(path_obj, "r", newline="", encoding="utf-8") as f:
            lines = f.readlines()
        header_line = lines[0].strip() if lines else ""
        existing = header_line.split(",") if header_line else []
        first_col = existing[0] if existing else ""
        if first_col != "trade_id":
            _rotate_legacy_file(path_obj, "misaligned")
            with open(path_obj, "w", newline="", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
            return
        missing = [c for c in header if c not in existing]
        if not missing:
            return
        lines[0] = ",".join(existing + missing) + "\n"
        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            f.writelines(lines)
        return

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")


def ensure_report_schema(path: str, expected_header: list[str]) -> None:
    """Reset *path* when the CSV header does not match ``expected_header``."""

    path_obj = Path(path)
    if not path_obj.exists():
        return
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return

    if not first_line:
        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(expected_header) + "\n")
        return

    existing = [col.strip() for col in first_line.split(",") if col.strip()]
    if not existing:
        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(expected_header) + "\n")
        return

    first_col = existing[0]
    if first_col != expected_header[0]:
        _rotate_legacy_file(path_obj, "misaligned")
        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(expected_header) + "\n")
        return

    if existing == expected_header:
        return

    _rotate_legacy_file(path_obj)
    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(expected_header) + "\n")

# [ANCHOR:ENTRY_LOG_UTILS]
def _now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _normalize_side(v: str | None) -> str:
    if not v:
        return "LONG"
    s = str(v).strip().upper()
    if s in ("LONG", "SHORT"):
        return s
    if s == "BUY":
        return "LONG"
    if s == "SELL":
        return "SHORT"
    return "LONG"

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

# [ANCHOR:LOG_ENTRY_NORMALIZER]
def normalize_entry_ctx(ctx: dict | None) -> dict:
    """Return ``ctx`` with default ``source`` and ``reduced_risk`` fields."""
    out = dict(ctx or {})
    out["source"] = out.get("source", "live")
    out["reduced_risk"] = bool(out.get("reduced_risk", False))
    return out


def _normalize_entry_row(entry_ctx: dict, symbol: str) -> dict:
    ctx = dict(entry_ctx or {})

    # --- timestamp normalization -------------------------------------------------
    ts_raw = ctx.get("timestamp_entry") or ctx.get("entry_time") or ctx.get("timestamp")
    if ts_raw is None:
        ts_val = _now_utc_iso()
    else:
        try:
            if isinstance(ts_raw, (int, float)):
                dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            else:
                s = str(ts_raw)
                if s.endswith("Z"):
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
            ts_val = dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except Exception:
            ts_val = _now_utc_iso()

    # --- required numeric fields ------------------------------------------------
    entry_price = ctx.get("entry_price") or ctx.get("price")
    if entry_price is None:
        raise ValueError("entry_price is required")
    entry_price = _to_float(entry_price)

    qty = ctx.get("qty") or ctx.get("quantity") or ctx.get("size")
    if qty is None:
        raise ValueError("qty is required")
    qty = _to_float(qty)

    reason = ctx.get("reason") or ctx.get("entry_reason") or "model_confirmed"
    reason = str(reason)

    sl_val = ctx.get("sl")
    if sl_val is None:
        sl_val = ctx.get("sl_price")
    tp_val = ctx.get("tp")
    if tp_val is None:
        tp_val = ctx.get("tp_price")

    return {
        "trade_id": str(ctx.get("trade_id") or ctx.get("id") or f"t_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"),
        "timestamp_entry": ts_val,
        "symbol": symbol,
        "side": _normalize_side(ctx.get("side")),
        "entry_price": entry_price,
        "sl": _to_float(sl_val) if sl_val is not None else "",
        "tp": _to_float(tp_val) if tp_val is not None else "",
        "qty": qty,
        "reason": reason,
        "source": str(ctx.get("source", "live")),
        "reduced_risk": bool(ctx.get("reduced_risk", False)),
    }

ALLOW_MARKET_FALLBACK = True
MAX_PERCENT_DIFF = 0.0015

COLOR_MAP = {
    'training': Fore.CYAN,
    'open': Fore.GREEN,
}


def colorize(msg: str, color_key: str) -> str:
    """Return ``msg`` wrapped with ANSI codes for ``color_key``."""
    color = COLOR_MAP.get(color_key, '')
    return f"{color}{msg}{Style.RESET_ALL}" if color else msg


def setup_logger(
    log_file: str = "bot.log",
    err_file: str | None = None,
    redirect_streams: bool = False,
) -> None:
    """Configure root logger with color support and file logging.

    Parameters
    ----------
    log_file:
        Path to the file where all log messages should be written. Defaults to
        ``bot.log`` in the current working directory.
    err_file:
        Deprecated. Retained for backward compatibility but ignored.
    redirect_streams:
        Deprecated. Stream redirection is disabled; the root logger always
        outputs to ``sys.stdout`` and the log file simultaneously.
    """
    try:
        from colorlog import ColoredFormatter
    except Exception:  # pragma: no cover - colorlog optional
        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
            )
        )
    else:
        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)s | %(message)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                },
            )
        )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handlers: list[logging.Handler] = [handler]

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logger.handlers = handlers


def log(level: int, context: str, *parts) -> None:
    """Log using unified ``context | part1 | part2`` formatting."""

    import logging

    logging.log(level, " | ".join([str(context)] + [str(p) for p in parts]))


def log_prediction_error(context: str, symbol: str, expected: int, got: int | None) -> None:
    """Helper to log mismatched or empty predictions.

    ``expected`` is the number of rows fed to the model while ``got`` is the
    length of the returned prediction array (or ``None``).
    """

    if got in (None, 0):
        msg = f"empty prediction: expected {expected}, got {got}"
    else:
        msg = f"prediction length {got} != {expected}"
    log(logging.ERROR, context, symbol, msg)


# status collected during one bot cycle
_candle_status: defaultdict[str, dict[str, int | None]] = defaultdict(dict)
_order_status: defaultdict[str, list[str]] = defaultdict(list)
_info_status: defaultdict[str, dict[str, str]] = defaultdict(dict)
_error_status: defaultdict[str, list[str]] = defaultdict(list)
_no_data_status: defaultdict[str, dict[str, str]] = defaultdict(dict)


def record_summary(symbol: str, mode: str, atr: float, adx: float, vol_ratio: float, signal: str) -> None:
    """Store summary metrics for ``symbol`` to be emitted later."""
    _info_status[symbol]["summary_base"] = {
        "mode": mode,
        "atr": atr,
        "adx": adx,
        "vol_ratio": vol_ratio,
        "signal": signal,
    }


def emit_summary(symbol: str, reason: str) -> None:
    """Log the summary for ``symbol`` with the final ``reason``."""
    base = _info_status.get(symbol, {}).get("summary_base")
    if not base:
        return
    msg = (
        f"mode={base['mode']} atr={base['atr']:.2f} adx={base['adx']:.1f} "
        f"vol={base['vol_ratio']:.2f} signal={base['signal']} reason={reason}"
    )
    log(logging.INFO, "summary", symbol, msg)


def log_decision(symbol: str, reason: str, *, decision: str = "skip", path: str = "decision_log.csv") -> None:
    """Append a trade decision (entry/skip) to ``decision_log.csv``."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "symbol", "signal", "reason"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), symbol, decision, reason])
    msg = f"Skipped: {reason}" if decision == "skip" else reason
    logging.getLogger().info("decision | %s | %s | %s", decision, symbol, msg)
    _info_status[symbol]["last_reason"] = reason
    emit_summary(symbol, reason)


_ENTRY_LOCK = Lock()
# [ANCHOR:LOG_ENTRY_IMPL]
ENTRY_CACHE_TTL_MIN = 10
_ENTRY_CACHE: dict[tuple[str, str, float], tuple[int, str]] = {}

_EXIT_LOCK = Lock()
_LOGGED_EXIT_IDS = {}  # TTL 6h


def log_entry(symbol: str, entry_ctx: dict, log_path: str) -> str | None:
    ensure_trades_csv_header(log_path)
    try:
        ctx = normalize_entry_ctx(entry_ctx)
        row = _normalize_entry_row(ctx, symbol)

        ts = datetime.fromisoformat(row["timestamp_entry"].replace("Z", "+00:00"))
        timestamp_ms = int(ts.timestamp() * 1000)
        trade_id = f"{timestamp_ms}-{symbol.replace('/', '_')}"
        row["trade_id"] = trade_id

        tick_size = ctx.get("tick_size")
        if tick_size:
            s = str(tick_size)
            tick_digits = len(s.split(".")[1].rstrip("0")) if "." in s else 0
        else:
            tick_digits = 8
        key = (symbol, row["side"], round(row["entry_price"], tick_digits))

        with _ENTRY_LOCK:
            now_ms = int(time.time() * 1000)
            ttl_ms = ENTRY_CACHE_TTL_MIN * 60 * 1000
            for k, (ts_ms, _) in list(_ENTRY_CACHE.items()):
                if now_ms - ts_ms > ttl_ms:
                    _ENTRY_CACHE.pop(k, None)
            cached = _ENTRY_CACHE.get(key)
            if cached and now_ms - cached[0] < ttl_ms:
                return cached[1]

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRADES_CSV_HEADER)
                out_row = {k: row.get(k, "") for k in TRADES_CSV_HEADER}
                w.writerow(out_row)

            _ENTRY_CACHE[key] = (now_ms, trade_id)
        return trade_id
    except Exception as e:
        import logging
        logging.exception("log_entry failed: %s", e)
        return None


# [ANCHOR:LOG_EXIT_IMPL]
def log_exit_from_order(symbol: str, order: dict, commission: float, trade_log_path: str) -> bool:
    """Idempotently append trade exit info to CSV log.

    Determines ``exit_type`` and related flags based on order/context.
    """
    ensure_trades_csv_header(trade_log_path)
    try:
        import sys

        main_mod = sys.modules.get("main")
        open_trade_ctx = getattr(main_mod, "open_trade_ctx", {})
        register_trade_result = getattr(main_mod, "register_trade_result", None)

        _LOGGED_EXIT_IDS.clear()

        ctx = open_trade_ctx.get(symbol)
        if not ctx:
            return False

        trade_id = str(ctx.get("trade_id") or f"{symbol}_{ctx.get('entry_time')}")

        with _EXIT_LOCK:
            now_ts = int(time.time())
            for k, t in list(_LOGGED_EXIT_IDS.items()):
                if now_ts - t > 6 * 3600:
                    _LOGGED_EXIT_IDS.pop(k, None)
            if trade_id in _LOGGED_EXIT_IDS and symbol not in open_trade_ctx:
                return False

            otype = (
                str(
                    order.get("type")
                    or order.get("info", {}).get("type")
                    or order.get("info", {}).get("origType", "")
                ).upper()
            )

            price = next(
                (
                    _to_float(order.get(k))
                    for k in ("avgPrice", "average", "price", "stopPrice")
                    if order.get(k) is not None
                ),
                0.0,
            )

            entry_price = _to_float(ctx.get("entry_price"))
            qty = _to_float(ctx.get("qty"))
            side = _normalize_side(ctx.get("side"))

            entry_time_str = ctx.get("entry_time")
            if entry_time_str:
                entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            else:
                entry_dt = datetime.now(timezone.utc)
                entry_time_str = entry_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            now_dt = datetime.now(timezone.utc)
            duration_sec = (now_dt - entry_dt).total_seconds()

            gross = (price - entry_price) * qty if side == "LONG" else (entry_price - price) * qty
            fee = commission * (entry_price + price) * qty
            profit = gross - fee

            exit_hint = ctx.pop("exit_type_hint", None)
            trail_active = bool(ctx.get("trailing_profit_used"))

            if otype == "STOP_MARKET":
                exit_type = "TRAIL_STOP" if trail_active else "SL"
            elif otype in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
                exit_type = "TP"
            elif exit_hint == "TIME":
                exit_type = "TIME"
            elif exit_hint == "TP":
                exit_type = "TP"
            else:
                exit_type = "MANUAL"

            row = {
                "trade_id": trade_id,
                "timestamp_entry": entry_time_str,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "sl": _to_float(ctx.get("sl_price")) if ctx.get("sl_price") is not None else "",
                "tp": _to_float(ctx.get("tp_price")) if ctx.get("tp_price") is not None else "",
                "qty": qty,
                "reason": ctx.get("reason", ""),
                "source": ctx.get("source", "live"),
                "reduced_risk": bool(ctx.get("reduced_risk", False)),
                "timestamp_exit": now_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "exit_price": price,
                "profit": profit,
                "pnl_net": profit,
                "exit_type": exit_type,
                "trail_triggered": exit_type == "TRAIL_STOP",
                "time_stop_triggered": exit_type == "TIME",
                "duration_sec": duration_sec,
            }

            with open(trade_log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRADES_CSV_HEADER)
                w.writerow(row)

            _LOGGED_EXIT_IDS[trade_id] = now_ts

        try:
            memory_manager.add_trade_close({**ctx, **row})
        except Exception:  # pragma: no cover - defensive
            logging.exception("add_trade_close failed")
        if register_trade_result:
            try:
                register_trade_result(symbol, profit, trade_log_path)
            except Exception:  # pragma: no cover - defensive
                logging.exception("register_trade_result failed")

        open_trade_ctx.pop(symbol, None)
        return True
    except Exception as e:  # pragma: no cover - robust logging
        logging.exception("log_exit_from_order failed: %s", e)
        return False


def record_candle_status(symbol: str, tf: str, count: int | None) -> None:
    """Store candle fetch status for later summary."""
    _candle_status[symbol][tf] = count


def record_no_data(symbol: str, scope: str, message: str | None = None) -> None:
    """Track missing data without polluting the error log."""

    details = _no_data_status[symbol]
    if scope not in details:
        details[scope] = message or ""
        memory_manager.add_event(
            "no_data",
            {"symbol": symbol, "scope": scope, "message": message or ""},
        )


def clear_no_data(symbol: str, scope: str) -> None:
    """Remove stored no-data marker when candles become available again."""

    details = _no_data_status.get(symbol)
    if details and scope in details:
        details.pop(scope, None)


def record_backtest(symbol: str, pct: float) -> None:
    """Store backtest return percentage for a symbol."""
    _info_status[symbol]["backtest"] = f"{pct*100:+.2f}%"


def record_pattern(symbol: str, name: str) -> None:
    """Store detected chart pattern for a symbol."""
    _info_status[symbol]["pattern"] = name


def record_error(symbol: str, msg: str) -> None:
    """Collect one-off runtime errors."""
    if msg not in _error_status[symbol]:
        _error_status[symbol].append(msg)
    memory_manager.add_event("error", {"symbol": symbol, "message": msg})


SOFT_ORDER_ERRORS = {
    "below_min_amount",
    "below_min_cost",
    "non_positive_qty",
    "normalization_failed",
}


def _normalize_order_qty(
    exchange,
    symbol: str,
    qty: float,
    price: float | None,
    order_type: str | None = None,
    *,
    side: str | None = None,
) -> tuple[float, str | None]:
    """Return quantity adjusted for exchange precision and limits."""

    def _safe_float(value, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(result):
            return float(default)
        return result

    def _safe_int(value) -> int | None:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue

    order_label = (order_type or "unknown").upper()

    market: dict = {}
    try:
        market = exchange.market(symbol) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("order_norm | %s | market lookup failed: %s", symbol, exc)
        market = {}

    precision = (market.get("precision") or {}) if isinstance(market, dict) else {}
    limits = (market.get("limits") or {}) if isinstance(market, dict) else {}
    amount_limits = (limits.get("amount") or {}) if isinstance(limits, dict) else {}
    cost_limits = (limits.get("cost") or {}) if isinstance(limits, dict) else {}

    amount_precision = _safe_int((precision or {}).get("amount"))

    amount_min = _safe_float(amount_limits.get("min"), 0.0)
    amount_step = _safe_float(amount_limits.get("step"), 0.0)
    if amount_step <= 0:
        amount_step = _safe_float(amount_limits.get("stepSize"), 0.0)

    info = market.get("info") if isinstance(market, dict) else None
    filters = []
    if isinstance(info, dict):
        if amount_min <= 0:
            amount_min = _safe_float(info.get("minQty") or info.get("min_qty"), amount_min)
        if amount_step <= 0:
            amount_step = _safe_float(info.get("stepSize") or info.get("step_size"), amount_step)
        filters = info.get("filters") or []
    elif isinstance(info, list):
        filters = info

    min_cost_val = _safe_float(cost_limits.get("min"), 0.0)
    notional_min = min_cost_val
    if isinstance(info, dict) and notional_min <= 0:
        notional_min = _safe_float(
            info.get("minNotional")
            or info.get("min_notional")
            or info.get("notional"),
            notional_min,
        )

    if isinstance(filters, list):
        for flt in filters:
            if not isinstance(flt, dict):
                continue
            ftype = (flt.get("filterType") or flt.get("filter_type") or "").upper()
            if ftype == "LOT_SIZE":
                if amount_min <= 0:
                    amount_min = _safe_float(flt.get("minQty"), amount_min)
                if amount_step <= 0:
                    amount_step = _safe_float(flt.get("stepSize"), amount_step)
            elif ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                candidate = _safe_float(
                    flt.get("minNotional")
                    or flt.get("notional")
                    or flt.get("min_notional"),
                    0.0,
                )
                if candidate > 0:
                    notional_min = candidate

    if amount_precision is not None and amount_step <= 0:
        try:
            amount_step = 1 / (10 ** max(amount_precision, 0))
        except Exception:
            amount_step = 0.0

    if amount_step <= 0:
        amount_step = _safe_float(market.get("lot"), amount_step)
    if amount_step <= 0:
        amount_step = _safe_float(market.get("step"), amount_step)
    if amount_step <= 0:
        amount_step = 10 ** -8

    if amount_min <= 0:
        amount_min = amount_step

    symbol_upper = symbol.upper()
    min_notional_fallback = 10.0 if symbol_upper.endswith("USDT") else float(MIN_NOTIONAL or 10.0)
    required_notional = max(min_cost_val, float(MIN_NOTIONAL))
    notional_min = max(notional_min, required_notional, min_notional_fallback)

    skip_notional = False
    try:
        best_price = float(price)
        if not math.isfinite(best_price) or best_price <= 0:
            raise ValueError
    except Exception:
        ticker = None
        try:
            ticker = exchange.fetch_ticker(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("order_norm | %s | fetch_ticker failed: %s", symbol, exc)
            ticker = None
        best_price = None
        if isinstance(ticker, dict):
            side_label = (side or "").lower()
            if side_label == "buy":
                preferred = ("ask", "last", "close", "bid")
            elif side_label == "sell":
                preferred = ("bid", "last", "close", "ask")
            else:
                preferred = ("last", "close", "ask", "bid")
            for key in preferred:
                val = ticker.get(key)
                candidate = _safe_float(val, 0.0)
                if candidate > 0:
                    best_price = candidate
                    break
            if (not best_price or best_price <= 0) and isinstance(ticker.get("info"), dict):
                info_block = ticker["info"]
                for key in ("lastPrice", "close", "markPrice", "price"):
                    candidate = _safe_float(info_block.get(key), 0.0)
                    if candidate > 0:
                        best_price = candidate
                        break
        if not best_price or best_price <= 0:
            logging.debug(
                "order_norm | %s | price unavailable; skipping notional checks", symbol
            )
            best_price = 1.0
            skip_notional = True

    def _ceil_step(value: float) -> float:
        if amount_step <= 0:
            return value
        if value <= 0:
            return 0.0
        return math.ceil(value / amount_step - 1e-12) * amount_step

    def _apply_precision(value: float) -> float:
        rounded = float(value)
        if rounded <= 0:
            return 0.0
        rounded = _ceil_step(rounded)
        if amount_precision is not None:
            try:
                rounded = round(rounded, max(amount_precision, 0))
            except Exception:
                pass
        if hasattr(exchange, "amount_to_precision"):
            try:
                rounded = float(exchange.amount_to_precision(symbol, rounded))
            except Exception:  # pragma: no cover - defensive
                pass
        rounded = _ceil_step(rounded)
        if amount_precision is not None:
            try:
                rounded = round(rounded, max(amount_precision, 0))
            except Exception:
                pass
        return float(rounded)

    try:
        qty_value = float(qty)
        if not math.isfinite(qty_value):
            qty_value = 0.0
    except Exception:
        qty_value = 0.0

    adjustments: list[str] = []

    if qty_value < amount_min:
        adjustments.append(
            f"raised_to_min_amount(prev={qty_value}, new={amount_min}, min={amount_min})"
        )
        qty_value = amount_min

    qty_value = _apply_precision(max(qty_value, amount_min))

    if skip_notional:
        need_qty = amount_min
    else:
        need_qty = _ceil_step(notional_min / best_price)
        if need_qty < amount_min:
            need_qty = amount_min

    if not skip_notional and qty_value * best_price < notional_min:
        prev_qty = qty_value
        qty_value = _apply_precision(max(qty_value, need_qty))
        adjustments.append(
            f"raised_to_min_notional(prev={prev_qty}, new={qty_value}, price={best_price}, minNotional={notional_min})"
        )

    if qty_value <= 0:
        prev_qty = qty_value
        qty_value = _apply_precision(max(amount_min, amount_step))
        adjustments.append(
            f"raised_to_min_amount(prev={prev_qty}, new={qty_value}, min={amount_min})"
        )

    if not skip_notional and qty_value * best_price < notional_min:
        prev_qty = qty_value
        qty_value = _apply_precision(max(need_qty, amount_min))
        adjustments.append(
            f"raised_to_min_notional(prev={prev_qty}, new={qty_value}, price={best_price}, minNotional={notional_min})"
        )

    notional = qty_value * best_price
    if qty_value <= 0 or (not skip_notional and notional < notional_min):
        raise ValueError(
            f"order_norm | {symbol} | unable to satisfy min limits qty={qty_value} min={amount_min} "
            f"notional={notional} minNotional={notional_min}"
        )

    if adjustments:
        logging.info(
            "order_norm | %s | %s | qty=%.8f notional=%.4f | %s",
            symbol,
            order_label,
            qty_value,
            notional,
            "; ".join(adjustments),
        )

    return float(qty_value), None


def safe_fetch_balance(exchange, params: dict | None = None, *, retries: int = 1, delay: float = 1.0):
    """Fetch account balance with basic rate limit handling."""

    attempt = 0
    params = _normalize_balance_params(exchange, params)
    while True:
        try:
            if params is None:
                return exchange.fetch_balance()
            return exchange.fetch_balance(params)
        except Exception as exc:  # pragma: no cover - network errors
            message = str(exc).lower()
            is_rate_limited = (
                "too many requests" in message
                or "429" in message
                or "rate limit" in message
                or "-1003" in message
            )
            if attempt < retries and is_rate_limited:
                logging.warning("exchange | fetch_balance rate limited: %s", exc)
                time.sleep(max(delay, 0.1))
                attempt += 1
                continue
            logging.warning("exchange | fetch_balance failed: %s", exc)
            raise


def safe_create_order(exchange, symbol: str, order_type: str, side: str,
                      qty: float, price=None, params=None):
    """Create an order with retry and exchange specific safeguards.

    For Bybit v5 the helper automatically injects ``category='linear'`` and
    ensures that attached take-profit / stop-loss instructions include the
    mandatory ``tpSlMode`` metadata when ``slOrderType`` is present.  This
    prevents API rejections such as ``slOrderType can not have a value when
    tpSlMode is empty`` while keeping the public function signature unchanged.
    """
    params, category = _with_bybit_order_params(exchange, symbol, params)
    resolved_category = category or ((params or {}).get("category") if params else None)
    base_params: dict | None = dict(params or {}) if params is not None else None
    is_bybit = _is_bybit_exchange(exchange)
    if not getattr(exchange, "markets", None):
        try:
            exchange.load_markets()
        except Exception:
            pass
    normalized_symbol = _normalize_bybit_symbol(exchange, symbol, resolved_category)
    display_symbol = symbol
    status_key = display_symbol
    symbol = normalized_symbol

    otype = str(order_type or "").lower()
    requested_market = otype.endswith("market")
    side = str(side or "").lower()
    adj_price = None if requested_market else price

    ticker = None
    try:
        fetched = exchange.fetch_ticker(symbol)
        if isinstance(fetched, dict):
            ticker = fetched
    except Exception:
        ticker = None

    def _extract_last_price() -> float | None:
        if not ticker:
            return None
        candidates = []
        for key in ("last", "close", "markPrice", "price"):
            if key in ticker:
                candidates.append(ticker.get(key))
        if side == "buy":
            candidates.extend([ticker.get("ask"), ticker.get("bid")])
        else:
            candidates.extend([ticker.get("bid"), ticker.get("ask")])
        for raw in candidates:
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
        return None

    last_price = _extract_last_price()
    entry_price = adj_price
    is_exit_order = any(tok in str(order_type).upper() for tok in ("STOP", "TAKE_PROFIT"))
    is_entry_order = not is_exit_order and otype in {"market", "limit"}

    if is_entry_order and last_price is not None:
        offset = 0.001 if side == "buy" else -0.001
        try:
            entry_price = float(last_price * (1 + offset))
        except Exception:
            entry_price = last_price
    elif adj_price is not None:
        entry_price = adj_price

    if entry_price is not None:
        entry_price, _ = _price_qty_to_precision(exchange, symbol, price=entry_price, amount=None)

    price_reference = entry_price if entry_price is not None else last_price
    try:
        norm_qty, skip_reason = _normalize_order_qty(
            exchange,
            symbol,
            qty,
            price_reference,
            order_type=order_type,
            side=side,
        )
    except Exception as exc:
        logging.warning(
            "order | %s | normalization failed: %s", display_symbol, exc
        )
        log(
            logging.WARNING,
            "order",
            display_symbol,
            f"normalization failed: {exc}",
        )
        tag = "order_normalization_failed"
        if tag not in _order_status[status_key]:
            _order_status[status_key].append(tag)
        return None, "normalization_failed"
    if skip_reason is not None:
        msg = f"order skipped: {skip_reason} (requested={qty})"
        log(logging.INFO, "order", display_symbol, msg)
        tag = f"order_{skip_reason}"
        if tag not in _order_status[status_key]:
            _order_status[status_key].append(tag)
        return None, skip_reason
    qty = norm_qty

    if entry_price is not None:
        entry_price, qty = _price_qty_to_precision(exchange, symbol, price=entry_price, amount=qty)
    else:
        _, qty = _price_qty_to_precision(exchange, symbol, price=None, amount=qty)

    adj_price = entry_price

    final_params: dict | None = base_params
    if is_bybit:
        def _normalize_trigger_direction_value(value):
            if value is None:
                return None
            td = value
            if isinstance(td, str):
                low = td.strip().lower()
                if low in ("ascending", "rising", "up"):
                    td = 1
                elif low in ("descending", "falling", "down"):
                    td = 2
                else:
                    try:
                        td = int(low)
                    except Exception:
                        td = None
            try:
                if td is not None:
                    return int(td)
            except Exception:
                return None
            return td

        def _float_or_none(value):
            if value is None:
                return None
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(result):
                return None
            return result

        params_dict = dict(base_params or {})
        upper_type = str(order_type or "").upper()
        is_exit_order = any(tok in upper_type for tok in ("STOP", "TAKE_PROFIT"))

        stop_loss = _float_or_none(params_dict.get("stopLoss"))
        take_profit = _float_or_none(params_dict.get("takeProfit"))
        if not is_exit_order:
            if stop_loss is None and "STOP" in upper_type:
                stop_loss = _float_or_none(
                    params_dict.get("stopPrice") or params_dict.get("triggerPrice")
                )
            if take_profit is None and "TAKE_PROFIT" in upper_type:
                take_profit = _float_or_none(
                    params_dict.get("stopPrice") or params_dict.get("triggerPrice")
                )

        sl_order_type = params_dict.get("slOrderType")
        tpsl_trigger_by = (
            params_dict.get("tpslTriggerBy")
            or params_dict.get("tpsl_trigger_by")
            or "MarkPrice"
        )

        entry_reference = adj_price if adj_price is not None else last_price
        if entry_reference is None:
            entry_reference = price

        main_module = sys.modules.get("main")
        if (stop_loss is None or take_profit is None) and not is_exit_order and entry_reference:
            try:
                entry_value = float(entry_reference)
                sl_pct = float(getattr(main_module, "SL_PCT", 0.02) or 0.02)
                tp_pct = float(getattr(main_module, "TP_PCT", 0.04) or 0.04)
                if side.lower() == "buy":
                    stop_candidate = entry_value * (1 - sl_pct)
                    take_candidate = entry_value * (1 + tp_pct)
                else:
                    stop_candidate = entry_value * (1 + sl_pct)
                    take_candidate = entry_value * (1 - tp_pct)
                if stop_loss is None:
                    stop_loss = stop_candidate
                if take_profit is None:
                    take_profit = take_candidate if take_candidate > 0 else None
                if stop_loss is not None and not sl_order_type:
                    sl_order_type = "Market"
            except (TypeError, ValueError):
                stop_loss = stop_loss if stop_loss is not None else None
                take_profit = take_profit if take_profit is not None else None

        trigger_direction = _normalize_trigger_direction_value(
            params_dict.get("triggerDirection")
        )
        if trigger_direction is None and stop_loss is not None and entry_reference:
            try:
                entry_value = float(entry_reference)
                stop_value = float(stop_loss)
            except (TypeError, ValueError):
                entry_value = stop_value = None
            if entry_value is not None and stop_value is not None:
                mapping = getattr(main_module, "BYBIT_TRIGGER_DIRECTIONS", {}) or {}
                if stop_value < entry_value:
                    trigger_direction = mapping.get("falling") or "descending"
                elif stop_value > entry_value:
                    trigger_direction = mapping.get("rising") or "ascending"

        trigger_direction = _normalize_trigger_direction_value(trigger_direction)
        if trigger_direction is None:
            params_dict.pop("triggerDirection", None)
        else:
            params_dict["triggerDirection"] = trigger_direction

        if stop_loss is not None:
            params_dict["stopLoss"] = float(stop_loss)
        else:
            params_dict.pop("stopLoss", None)
        if take_profit is not None:
            params_dict["takeProfit"] = float(take_profit)
        else:
            params_dict.pop("takeProfit", None)

        cat = str(params_dict.get("category") or resolved_category or "").lower()
        if cat in ("", "swap", "usdt_perp"):
            cat = "linear"
        if cat not in ("linear", "inverse", "option", "spot"):
            cat = "linear"
        params_dict["category"] = cat
        if cat in ("linear", "inverse"):
            params_dict.setdefault("positionIdx", 0)
        else:
            params_dict.pop("positionIdx", None)
        resolved_category = cat

        has_sl = params_dict.get("stopLoss") is not None
        has_tp = params_dict.get("takeProfit") is not None
        if not is_exit_order and (has_sl or has_tp):
            params_dict.setdefault("tpSlMode", "Full")
            params_dict.setdefault("tpslTriggerBy", tpsl_trigger_by)
        if params_dict.get("slOrderType") and not (has_sl or has_tp):
            params_dict.pop("slOrderType", None)

        if is_exit_order:
            for key in (
                "slOrderType",
                "tpSlMode",
                "tpslMode",
                "tpslTriggerBy",
                "stopLoss",
                "takeProfit",
            ):
                params_dict.pop(key, None)
            params_dict.setdefault("reduceOnly", True)
            params_dict.setdefault("closeOnTrigger", True)

        final_params = params_dict

    if not is_exit_order:
        if final_params is None:
            final_params = {}
        final_params.setdefault("reduceOnly", False)

    final_params = _clean_params(final_params)

    if final_params is not None and not final_params:
        final_params = None

    market_retry_allowed = False
    market_retry_used = False
    last_error: str | None = None

    for attempt in range(2):
        try:
            if is_entry_order:
                if attempt == 0:
                    send_type = "limit"
                    price_arg = adj_price
                elif attempt == 1 and market_retry_allowed:
                    send_type = "market"
                    price_arg = None
                else:
                    break
            else:
                send_type = order_type
                price_arg = None if requested_market else adj_price

            order_type_api_current = send_type
            if is_bybit:
                order_type_api_current = _normalize_bybit_order_type_value(send_type)

            params_for_call, category_for_call = _with_bybit_order_params(
                exchange,
                display_symbol,
                final_params,
            )
            params_for_call = _clean_params(params_for_call)
            conditional_exit = bool(
                params_for_call
                and str(params_for_call.get("reduceOnly")).lower() == "true"
                and (
                    params_for_call.get("triggerPrice") is not None
                    or params_for_call.get("triggerDirection") is not None
                )
            )
            if conditional_exit:
                price_arg = None
                if isinstance(params_for_call, dict):
                    params_for_call.pop("slOrderType", None)
            call_symbol = _normalize_bybit_symbol(exchange, display_symbol, category_for_call)

            order = exchange.create_order(
                call_symbol,
                order_type_api_current,
                side,
                qty,
                price_arg,
                params_for_call,
            )
            order_id = order.get("id") or order.get("orderId")
            status = str(order.get("status", "")).upper()
            if not order_id or status.lower() in {"rejected", "canceled"}:
                raise ValueError(f"order {status or 'invalid'}")
            avg_price = order.get("avgPrice") or order.get("average") or order.get("price") or adj_price
            filled = float(order.get("executedQty") or order.get("filled") or qty)
            original = float(order.get("origQty") or qty)
            # discard any previously stored failure messages after a successful order
            _order_status[status_key] = [
                m
                for m in _order_status[status_key]
                if not m.lower().startswith("order failed")
            ]
            if market_retry_used:
                fallback_tag = "order_market_fallback"
                if fallback_tag not in _order_status[status_key]:
                    _order_status[status_key].append(fallback_tag)
            if status == "FILLED" or filled == original:
                msg = f"✅ Order {side.upper()} {filled:g} @ {avg_price} ({status or 'FILLED'})"
                if msg not in _order_status[status_key]:
                    _order_status[status_key].append(msg)
                return order_id, None
            else:
                msg = f"⚠️ Order {status}: {filled:g}/{original:g} filled"
                if msg not in _order_status[status_key]:
                    _order_status[status_key].append(msg)
                return order_id, None
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            last_error = err_str
            if is_entry_order and attempt == 0:
                band_error = any(code in err_str for code in ("30208", "10001", "-4131"))
                band_error = band_error or ("price" in err_lower and "band" in err_lower)
                param_error = "request parameter error" in err_lower
                if ALLOW_MARKET_FALLBACK and (band_error or param_error):
                    market_retry_allowed = True
                    log_once(
                        logging,
                        "warning",
                        f"order | {display_symbol} | limit rejected ({err_str}); retrying as MARKET",
                    )
                    tag = "order_price_band_retry"
                    if tag not in _order_status[status_key]:
                        _order_status[status_key].append(tag)
                    market_retry_used = True
                    continue
            if "170131" in err_str or "insufficient balance" in err_lower:
                msg = "order_insufficient_balance"
                if msg not in _order_status[status_key]:
                    _order_status[status_key].append(msg)
                log(
                    logging.WARNING,
                    "order",
                    display_symbol,
                    "insufficient balance while creating order",
                )
                return None, "insufficient_balance"
            specific = None
            if "-2019" in err_str and "Margin is insufficient" in err_str:
                specific = "❌ Недостаточно маржи, позиция не может быть открыта."
            elif "-2027" in err_str:
                specific = "❌ Превышена максимальная позиция при текущем плече."
            elif "-4005" in err_str:
                specific = "❌ Объем ордера превышает допустимый максимум."
            elif "-4164" in err_str:
                specific = "❌ Минимальный объем ордера 10 USDT."
            msg = f"Order failed: {specific or err_str}"
            if msg not in _order_status[status_key]:
                _order_status[status_key].append(msg)
            return None, err_str

    return None, last_error or "order_failed"


def place_conditional_exit(ex, symbol: str, side_open: str, base_price: float, pct: float, *, is_tp: bool):
    cat = "linear"
    norm = _normalize_bybit_symbol(ex, symbol, cat)
    try:
        ticker = ex.fetch_ticker(norm)
    except Exception:
        ticker = {}
    last = float((ticker or {}).get('last') or (ticker or {}).get('bid') or (ticker or {}).get('ask') or 0.0)
    trig, direction, side_to_send = _bybit_trigger_for_exit(side_open, last, base_price, pct, is_tp=is_tp)
    trig, _ = _price_qty_to_precision(ex, norm, price=trig, amount=None)
    _, pos_qty = has_open_position(ex, norm, cat)
    if pos_qty <= 0:
        raise RuntimeError(f"exit skipped: no position yet for {symbol}")
    amount = _round_qty(ex, norm, pos_qty)
    try:
        stops = ex.fetch_open_orders(norm, params={'category': cat, 'orderFilter': 'StopOrder'})
        for o in stops[2:]:
            ex.cancel_order(o['id'], norm, {'category': cat})
    except Exception:
        pass
    params = {
        'category': cat,
        'orderType': 'Market',
        'triggerPrice': float(trig),
        'triggerDirection': direction,
        'triggerBy': 'LastPrice',
        'reduceOnly': True,
        'closeOnTrigger': True,
        'tpSlMode': 'Full',
        'positionIdx': 0,
    }
    params = _clean_params(params)
    try:
        resp = ex.create_order(norm, 'market', side_to_send, amount, None, params)
    except Exception as exc:
        return None, str(exc)
    order_id = None
    if isinstance(resp, dict):
        order_id = resp.get('id') or resp.get('orderId')
    return order_id, None


def wait_position_after_entry(ex, symbol: str, category: str = "linear", timeout_sec: float = 3.0) -> float:
    norm = _normalize_bybit_symbol(ex, symbol, category)
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        _, qabs = has_open_position(ex, norm, category)
        if qabs > 0:
            return qabs
        time.sleep(0.15)
    return 0.0


def has_pending_entry(ex, symbol: str, side: str, category: str = "linear") -> bool:
    norm = _normalize_bybit_symbol(ex, symbol, category)
    try:
        opens = ex.fetch_open_orders(norm, params={"category": category})
    except Exception:
        opens = []
    side = side.lower()
    for o in opens or []:
        if str(o.get("reduceOnly")).lower() == "true":
            continue
        if (o.get("side") or "").lower() != side:
            continue
        return True
    return False


def safe_set_leverage(exchange, symbol: str, leverage: int, attempts: int = 2) -> bool:
    """Set leverage with retry and structured logging."""

    from exchange_adapter import LEVERAGE_SKIPPED, set_valid_leverage

    for attempt in range(attempts):
        L = set_valid_leverage(exchange, symbol, leverage)
        if L is LEVERAGE_SKIPPED:
            log(logging.INFO, "leverage", symbol, "Leverage setup skipped")
            return True
        if L is not None:
            log(logging.INFO, "leverage", symbol, f"Leverage set to {L}x")
            return True
        if attempt < attempts - 1:
            time.sleep(2)
            continue
        log(logging.WARNING, "leverage", symbol, f"Failed to set leverage {leverage} (soft)")
        tag = "leverage_failed_soft"
        if tag not in _order_status[symbol]:
            _order_status[symbol].append(tag)
        return False


def flush_symbol_logs(symbol: str) -> None:
    """Emit aggregated log messages and clear stored data."""
    _info_status.pop(symbol, None)
    _candle_status.pop(symbol, None)
    orders = _order_status.pop(symbol, [])
    errors = _error_status.pop(symbol, [])

    for msg in orders:
        level = logging.INFO
        lower = msg.lower()
        if msg.startswith("❌"):
            level = logging.ERROR
        elif lower.startswith("order failed") or msg.startswith("⚠️"):
            level = logging.WARNING
        log(level, "order", symbol, msg)
    for err in errors:
        log(logging.ERROR, "error", symbol, err)


def flush_cycle_logs() -> None:
    """Flush logs for all symbols."""
    for sym in set(_candle_status) | set(_order_status) | set(_info_status) | set(_error_status):
        flush_symbol_logs(sym)
