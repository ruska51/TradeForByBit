# [ANCHOR:SYMBOL_UTILS_HEADER]
import time
import logging
from typing import Tuple

MARKET_TTL = 300  # seconds


def normalize_symbol_for_exchange(exchange, symbol: str, markets_cache: dict) -> str:
    """Return ``symbol`` adjusted to what ``exchange`` understands.

    The implementation is intentionally tiny and only covers the behaviours
    required by tests:

    - if the exact symbol exists in ``exchange.markets`` it is returned as-is;
    - if a ``":USDT"`` suffix exists in markets it is appended;
    - if the un-slashed version exists in ``markets_by_id`` the mapped symbol is
      returned.
    """

    if not markets_cache.get("loaded"):
        markets_cache["by_name"] = set(getattr(exchange, "markets", {}) or {})
        markets_cache["by_id"] = {
            k: v.get("symbol") for k, v in getattr(exchange, "markets_by_id", {}).items()
        }
        markets_cache["loaded"] = True

    by_name = markets_cache.get("by_name", set())
    if symbol in by_name:
        return symbol
    suffix = f"{symbol}:USDT"
    if suffix in by_name:
        return suffix
    by_id = markets_cache.get("by_id", {})
    alt = by_id.get(symbol.replace("/", ""))
    if alt:
        return alt
    return symbol


def filter_supported_symbols(adapter, symbols: list[str], markets_cache) -> tuple[list[str], list[str], bool]:
    """Filter ``symbols`` using ``adapter.load_markets``.

    On any market loading error the original list is returned and no exception
    is raised (soft degradation).
    """

    try:
        now = time.time()
        if ("set" not in markets_cache) or (now - markets_cache.get("ts", 0) > MARKET_TTL):
            markets_cache["set"] = adapter.load_markets()
            markets_cache["ts"] = now
        markets = markets_cache.get("set", set())
    except Exception as e:  # pragma: no cover - warning only
        logging.warning(
            "filter | markets unavailable, skipping symbol-filter: %s", e
        )
        if hasattr(adapter, "supports_symbol"):
            supported = [s for s in symbols if adapter.supports_symbol(s)]
            removed = [s for s in symbols if not adapter.supports_symbol(s)]
            if not supported:
                logging.info(
                    "filter | no supported symbols after supports_symbol check; falling back to original list (degraded mode)"
                )
                return symbols[:], [], True
            return supported, removed, False
        return symbols[:], [], False

    if not markets:
        if hasattr(adapter, "_warn_once"):
            adapter._warn_once(("markets", "empty"), "filter | empty markets set, skipping symbol-filter")
        else:
            logging.info("filter | empty markets set, skipping symbol-filter")
        return symbols[:], [], False

    supported = [s for s in symbols if s in markets]
    removed = [s for s in symbols if s not in markets]

    if not supported:
        logging.info(
            "filter | no supported symbols after filtering; falling back to original list (degraded mode)"
        )
        return symbols[:], [], True

    return supported, removed, False


def filter_linear_markets(
    adapter,
    symbols: list[str],
    markets_cache,
    *,
    force_reload: bool = False,
) -> Tuple[list[str], list[str]]:
    """Return ``symbols`` split into supported linear swaps and pending ones.

    The function inspects the loaded CCXT markets and keeps only those where the
    ``linear`` flag is truthy which corresponds to USDT-margined perpetual
    contracts on Bybit.  Symbols that do not expose a linear contract are
    returned separately so the caller can re-check them later without letting
    them block trading cycles.
    """

    try:
        now = time.time()
        needs_reload = force_reload or ("set" not in markets_cache)
        needs_reload |= now - markets_cache.get("ts", 0.0) > MARKET_TTL
        if needs_reload:
            markets_cache["set"] = adapter.load_markets()
            markets_cache["ts"] = now
        markets: dict = markets_cache.get("set", {}) or {}
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.warning("filter | linear markets unavailable: %s", exc)
        return symbols[:], []

    supported: list[str] = []
    pending: list[str] = []

    for symbol in symbols:
        market = markets.get(symbol)
        if not market:
            pending.append(symbol)
            continue

        linear_flag = market.get("linear")
        info = market.get("info") or {}
        contract_type = str(info.get("contractType") or info.get("contract_type") or "")
        is_linear = False
        if isinstance(linear_flag, bool):
            is_linear = linear_flag
        elif isinstance(linear_flag, (int, float)):
            is_linear = bool(int(linear_flag))
        elif isinstance(linear_flag, str):
            is_linear = linear_flag.lower() in {"1", "true", "linear", "usdm"}
        if not is_linear and contract_type:
            is_linear = "linear" in contract_type.lower()

        if is_linear:
            supported.append(symbol)
        else:
            pending.append(symbol)

    return supported, pending

