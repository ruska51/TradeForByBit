from __future__ import annotations

"""Light-weight exchange adapter using only the CCXT backend."""

import logging
import os
import time
import importlib
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from logging_utils import (
    log,
    normalize_bybit_category,
    log_once,
)


LEVERAGE_SKIPPED = object()


def _ccxt_symbol(symbol: str) -> str:
    """Return the short CCXT identifier without settlement suffixes."""

    if not isinstance(symbol, str):
        return symbol

    value = symbol.strip()
    if not value:
        return value

    if ":" in value:
        value = value.split(":", 1)[0]

    if "/" in value:
        base, quote = value.split("/", 1)
        value = f"{base}{quote}"

    return value


def detect_market_category(exchange, symbol: str) -> str | None:
    """Best-effort detection of the market category for ``symbol``.

    Returns ``"linear"`` or ``"inverse"`` when a derivative contract can be
    detected.  Spot markets result in ``None``.  ``"swap"`` hints are treated as
    ``"linear"`` because Bybit represents perpetual contracts that way.
    """

    if not exchange or not isinstance(symbol, str) or not symbol.strip():
        return None

    markets_loaded = getattr(exchange, "markets", None)
    if not markets_loaded:
        loader = getattr(exchange, "load_markets", None)
        if callable(loader):
            try:
                loader()
            except Exception:
                pass

    def _normalize(value) -> str | None:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if not lowered:
                return None
            if "inverse" in lowered:
                return "inverse"
            if "linear" in lowered:
                return "linear"
            if lowered in {"swap", "perp", "perpetual"}:
                return "linear"
        return None

    def _market_category(meta: dict | None) -> str | None:
        if not isinstance(meta, dict):
            return None

        if meta.get("inverse"):
            return "inverse"
        if meta.get("linear"):
            return "linear"
        if meta.get("swap"):
            return "linear"

        info = meta.get("info")
        if isinstance(info, dict):
            for key in (
                "category",
                "contractType",
                "productType",
                "market",
                "subType",
            ):
                normalized = _normalize(info.get(key))
                if normalized:
                    return normalized

        for key in ("category", "type"):
            normalized = _normalize(meta.get(key))
            if normalized:
                return normalized

        settle = str(meta.get("settle") or meta.get("settleId") or "").upper()
        quote = str(meta.get("quote") or "").upper()
        if settle and quote:
            if settle != quote:
                return "inverse"
            return "linear"

        market_type = str(meta.get("type") or "").lower()
        if market_type in {"swap", "future", "futures", "contract"}:
            return "linear"

        return None

    def _lookup_market(sym: str) -> dict | None:
        try:
            market = exchange.market(sym)
            if isinstance(market, dict) and market:
                return market
        except Exception:
            pass

        markets = getattr(exchange, "markets", {}) or {}
        if isinstance(markets, dict):
            market = markets.get(sym)
            if isinstance(market, dict) and market:
                return market

        markets_by_id = getattr(exchange, "markets_by_id", {}) or {}
        if isinstance(markets_by_id, dict):
            market = markets_by_id.get(sym)
            if isinstance(market, dict) and market:
                return market

        return None

    def _iter_markets():
        markets = getattr(exchange, "markets", {}) or {}
        if isinstance(markets, dict):
            yield from markets.values()
        markets_by_id = getattr(exchange, "markets_by_id", {}) or {}
        if isinstance(markets_by_id, dict):
            yield from markets_by_id.values()

    def _parse_base_quote(sym: str) -> tuple[str | None, str | None]:
        if not isinstance(sym, str):
            return (None, None)
        if "/" in sym:
            base, rest = sym.split("/", 1)
            quote = rest.split(":", 1)[0]
            base = base.strip().upper()
            quote = quote.strip().upper()
            return (base or None, quote or None)
        return (None, None)

    symbol = symbol.strip()
    short_symbol = _ccxt_symbol(symbol)
    market_symbol = symbol.split(":", 1)[0]
    candidates: list[str] = []
    for candidate in (symbol, market_symbol, short_symbol):
        if isinstance(candidate, str) and candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        market = _lookup_market(candidate)
        category = _market_category(market)
        if category:
            return category

    base = quote = None
    for candidate in candidates:
        base, quote = _parse_base_quote(candidate)
        if base and quote:
            break

    if not (base and quote):
        for market in _iter_markets():
            mapped_symbol = None
            if isinstance(market, dict):
                mapped_symbol = market.get("symbol") or market.get("id")
            if isinstance(mapped_symbol, str):
                base, quote = _parse_base_quote(mapped_symbol)
                if base and quote and _market_category(market):
                    if short_symbol and short_symbol.upper() == _ccxt_symbol(mapped_symbol).upper():
                        return _market_category(market)
        return None

    base = base.upper()
    quote = quote.upper()

    for market in _iter_markets():
        if not isinstance(market, dict):
            continue
        meta_base = str(market.get("base") or "").upper()
        meta_quote = str(market.get("quote") or "").upper()
        if not meta_base or not meta_quote:
            mapped = market.get("symbol") or market.get("id")
            if isinstance(mapped, str):
                meta_base, meta_quote = _parse_base_quote(mapped)
                meta_base = (meta_base or "").upper()
                meta_quote = (meta_quote or "").upper()
        if not meta_base or not meta_quote:
            continue
        if meta_base != base or meta_quote != quote:
            continue
        category = _market_category(market)
        if category:
            return category

    return None


# ``ccxt`` is imported lazily so tests can monkeypatch the module before the
# adapter attempts to use it.  The variable is populated on first use.
_ccxt = None
_binance_import_attempted = False


# ---------------------------------------------------------------------------
# custom exceptions used by the tests


class AdapterInitError(RuntimeError):
    """Raised when the adapter fails to initialise a backend."""


class AdapterOHLCVUnavailable(RuntimeError):
    """Raised when OHLCV data cannot be fetched."""


class AdapterOrdersUnavailable(RuntimeError):
    """Raised when order management API is unusable."""


# ---- timeframe helpers ----------------------------------------------------
TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1d",
}


def to_ccxt(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    return symbol[:-4] + "/" + symbol[-4:]


def to_sdk(symbol: str) -> str:
    # ``python-binance`` is no longer used but a couple of modules still call
    # :func:`to_sdk` in tests, hence it stays as a thin helper.
    return symbol.replace("/", "")


# ---------------------------------------------------------------------------


def _bybit_leverage_candidates(exchange, symbol: str) -> list[str]:
    """Return possible Bybit symbols for leverage configuration.

    Assumptions:
    - ``symbol`` is a CCXT unified symbol.
    - ``exchange`` exposes ``market`` metadata when available.
    - USDT settled contracts require the ``:USDT`` suffix.
    """

    candidates: list[str] = []
    market_obj: dict[str, Any] | None = None

    try:
        market_data = exchange.market(symbol)
        if isinstance(market_data, dict) and market_data:
            market_obj = market_data
    except Exception:
        market_obj = None

    for key in ("symbol", "id"):
        if market_obj and isinstance(market_obj.get(key), str):
            value = market_obj[key]
            if value and value not in candidates:
                candidates.append(value)

    if isinstance(symbol, str) and symbol and symbol not in candidates:
        candidates.append(symbol)

    enriched: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in enriched:
            enriched.append(candidate)
        if (
            candidate
            and "/" in candidate
            and ":" not in candidate
            and candidate.split("/", 1)[1].upper() == "USDT"
        ):
            with_suffix = f"{candidate}:USDT"
            if with_suffix not in enriched:
                enriched.append(with_suffix)
    return enriched or candidates


def set_valid_leverage(exchange, symbol: str, leverage: int | float):
    """Set leverage for Bybit linear/inverse derivatives using minimal, explicit params."""

    try:
        L = int(leverage)
    except Exception:
        L = int(float(leverage))

    cat = detect_market_category(exchange, symbol)
    cat_norm = str(cat or "").lower()
    if not cat_norm or cat_norm == "swap":
        cat_norm = "linear"
    if cat_norm not in {"linear", "inverse"}:
        cat_norm = "linear"

    exchange_id = str(getattr(exchange, "id", "") or "").lower()
    if exchange_id == "bybit":
        symbol_candidates = _bybit_leverage_candidates(exchange, symbol)
    else:
        symbol_candidates = [symbol]

    params = {"category": cat_norm, "buyLeverage": L, "sellLeverage": L}

    last_exc: Exception | None = None
    for idx, candidate in enumerate(symbol_candidates):
        target_symbol = candidate if isinstance(candidate, str) else str(candidate)
        if not target_symbol:
            continue
        try:
            return exchange.set_leverage(L, target_symbol, params)
        except TypeError:
            raise
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
            lowered = str(exc).lower()
            if any(token in lowered for token in ("soft", "cross", "not modify")):
                log_once("info", f"leverage | {symbol} | skipped ({exc})", window_sec=30.0)
                return LEVERAGE_SKIPPED
            retry_tokens = (
                "only support linear and inverse",
                "linear contract",
                "invalid symbol",
                "symbol invalid",
                "not linear",
            )
            if exchange_id == "bybit" and idx < len(symbol_candidates) - 1:
                if any(token in lowered for token in retry_tokens):
                    continue
            break

    if last_exc is not None:
        log_once(
            "warning",
            f"leverage | {symbol} | failed: {last_exc}",
            window_sec=30.0,
        )
    return None


def validate_api(exchange) -> None:
    """Validate API credentials and exit when Bybit rejects them."""

    if exchange is None:
        return

    api_key = getattr(exchange, "apiKey", None)
    secret = getattr(exchange, "secret", None)
    if not api_key or not secret:
        return

    try:
        exchange.check_required_credentials()
    except Exception as exc:
        logging.error("adapter | API credentials missing or invalid: %s", exc)
        sys.exit("API credentials missing or invalid; please update settings.")

    try:
        exchange.fetch_balance({"type": "future"})
    except Exception as exc:
        raw_message = str(exc)
        message = raw_message.lower()
        code = getattr(exc, "code", None)
        is_bybit = str(getattr(exchange, "id", "")).lower() == "bybit"
        if is_bybit and (code == 10003 or "10003" in raw_message):
            logging.error("adapter | Bybit rejected credentials with code 10003: %s", exc)
            sys.exit("Bybit API key rejected (code 10003); please update credentials.")
        if "api key" in message and "invalid" in message:
            logging.error("adapter | API key is invalid: %s", exc)
            sys.exit("API key is invalid; please check credentials.")
        if "authentication" in message or "auth" in message:
            logging.error("adapter | authentication failed during API validation: %s", exc)
            sys.exit("API authentication failed; please verify credentials.")
        logging.warning("adapter | API validation warning: %s", exc)
def safe_fetch_closed_orders(exchange, symbol: str | None = None, limit: int = 50, params: dict | None = None):
    """Return closed orders without raising on failures."""

    try:
        return exchange.fetch_closed_orders(symbol, None, limit, params or {})
    except TypeError:
        try:
            return exchange.fetch_closed_orders(symbol, None, limit)
        except TypeError:
            try:
                return exchange.fetch_closed_orders(symbol, None)
            except TypeError:
                return exchange.fetch_closed_orders(symbol)
    except Exception as e:  # pragma: no cover - network errors
        logging.warning(
            "exchange | fetch_closed_orders | %s | %s", symbol or "ALL", e
        )
        return []


class ExchangeAdapter:
    """Minimal CCXT based exchange adapter."""

    def __init__(self, exchange: Any | None = None, config: Optional[Dict[str, Any]] = None):
        self.x: Any | None = None
        self.config = config or {}
        self.sandbox = bool(self.config.get("sandbox", False))
        self.futures = bool(self.config.get("futures", False))
        self.exchange_id = (
            self.config.get("exchange_id")
            or os.getenv("EXCHANGE_ID", "bybit")
        ).lower()
        self.config.setdefault("exchange_id", self.exchange_id)
        self.backend = "ccxt"
        self.ccxt_id: str | None = None
        self.last_warn_at: Dict[tuple[str, str], float] = {}
        self.markets_loaded_at: float = 0.0
        self._no_data_symbols: dict[str, float] = {}
        self._ohlcv_cache: dict[tuple[str, str, int], tuple[float, list[list]]]
        self._ohlcv_cache = {}
        self._log_once_cache: dict[tuple[str, str], float] = {}

        self._detect_backend()

        # ensure markets are populated during initialisation
        if getattr(self, "x", None):
            if not getattr(self.x, "markets", None):
                self.markets_loaded_at = 0.0
                if self.load_markets_safe() and not getattr(self.x, "markets", None):
                    logging.warning(
                        "adapter | load_markets_safe succeeded but markets empty"
                    )
            self._verify_exchange_options()

    def _ensure_no_data_store(self) -> dict[str, float]:
        store = getattr(self, "_no_data_symbols", None)
        if store is None:
            store = {}
            self._no_data_symbols = store
        return store

    # ------------------------------------------------------------------
    def _default_params(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        include_position_idx: bool = True,
        symbol: str | None = None,
    ) -> Dict[str, Any]:
        """Return params extended with exchange specific defaults.

        Some Bybit endpoints (private order management) require ``positionIdx``
        while public market-data endpoints reject it.  Previously the adapter
        always injected ``positionIdx`` which resulted in empty responses when
        requesting OHLCV candles on the testnet API.  The caller can now opt out
        via ``include_position_idx`` so market data requests remain compatible
        without regressing the trading endpoints that still expect the value.
        """

        base = dict(params or {})
        ex_id = str(getattr(self, "exchange_id", "") or "").lower()
        if not ex_id and getattr(self, "x", None):
            ex_id = str(getattr(self.x, "id", "") or "").lower()
        normalized_symbol = symbol
        if symbol:
            normalized_symbol = self._ccxt_symbol(symbol)
        if "bybit" in ex_id:
            if normalized_symbol:
                category = self._detect_bybit_category(normalized_symbol)
                if category:
                    category = category.lower()
                    if category in {"", "swap"}:
                        category = "linear"
                    if category == "spot" and self.futures:
                        category = "linear"
                    base.setdefault("category", category)
            elif self.futures:
                base.setdefault("category", "linear")
            if include_position_idx and self.futures:
                base.setdefault("positionIdx", base.get("positionIdx", 0))
            elif not include_position_idx:
                base.pop("positionIdx", None)
        return base

    # ------------------------------------------------------------------
    def _ccxt_symbol(self, symbol: str) -> str:
        """Return symbol string recognised by the underlying CCXT exchange."""

        ex = getattr(self, "x", None)
        if not symbol or not ex:
            return self._finalize_ccxt_symbol(symbol)

        markets = getattr(ex, "markets", {}) or {}
        if symbol in markets:
            return self._finalize_ccxt_symbol(symbol)

        try:
            market = ex.market(symbol)
            if isinstance(market, dict):
                mapped = market.get("symbol")
                if isinstance(mapped, str) and mapped in markets:
                    return self._finalize_ccxt_symbol(mapped)
        except Exception:
            pass

        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            candidate = f"{base}/{quote}:{quote}"
            if candidate in markets:
                return self._finalize_ccxt_symbol(candidate)

        markets_by_id = getattr(ex, "markets_by_id", {}) or {}
        lookup = symbol.replace("/", "")
        market = markets_by_id.get(lookup)
        if isinstance(market, dict):
            mapped = market.get("symbol") or market.get("info", {}).get("symbol")
            if isinstance(mapped, str):
                return self._finalize_ccxt_symbol(mapped)
        elif isinstance(market, str):
            return self._finalize_ccxt_symbol(market)

        if symbol.endswith(":USDT"):
            trimmed = symbol.split(":", 1)[0]
            if trimmed in markets:
                return self._finalize_ccxt_symbol(trimmed)

        return self._finalize_ccxt_symbol(symbol)

    def _finalize_ccxt_symbol(self, candidate: str) -> str:
        """Return *candidate* without Bybit settle suffixes."""

        resolved = candidate or ""
        if ":" in resolved and "/" not in resolved:
            resolved = resolved.split(":", 1)[0]
        if "/" not in resolved and len(resolved) > 4:
            try:
                converted = to_ccxt(resolved)
            except Exception:
                converted = None
            if isinstance(converted, str) and converted:
                resolved = converted
        return resolved or candidate

    # ------------------------------------------------------------------
    def _detect_bybit_category(self, symbol: str) -> str | None:
        """Best-effort detection of Bybit market category for ``symbol``."""

        ex = getattr(self, "x", None)
        if not ex:
            return None

        ccxt_symbol = self._ccxt_symbol(symbol)

        no_data = self._ensure_no_data_store()
        if symbol in no_data:
            return None
        market: dict[str, Any] | None = None
        try:
            market_obj = ex.market(ccxt_symbol)
            if isinstance(market_obj, dict) and market_obj:
                market = market_obj
        except Exception:
            pass

        if market is None:
            markets = getattr(ex, "markets", {}) or {}
            market = markets.get(ccxt_symbol) or markets.get(symbol)

        if not isinstance(market, dict):
            return None

        info = market.get("info") or {}
        category_raw = (
            info.get("category")
            or info.get("contractType")
            or info.get("productType")
        )
        category = normalize_bybit_category(category_raw)
        if category:
            return category

        market_type = str(market.get("type") or "").lower()
        if market.get("spot") or market_type == "spot":
            return "spot"
        if market.get("option") or market_type == "option":
            return "option"
        if market.get("inverse"):
            return "inverse"
        if market.get("linear"):
            return "linear"

        if market_type in {"swap", "future"}:
            settle = str(market.get("settle") or market.get("settleId") or "").upper()
            quote = str(market.get("quote") or "").upper()
            if settle and quote and settle != quote:
                return "inverse"
            if settle or quote:
                return "linear"

        return None

    # ------------------------------------------------------------------
    @property
    def is_futures(self) -> bool:
        """Return whether the adapter is configured for futures trading."""
        return bool(self.futures)

    # ------------------------------------------------------------------
    def _get_ohlcv_cache(self) -> dict[tuple[str, str, int], tuple[float, list[list]]]:
        cache = getattr(self, "_ohlcv_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_ohlcv_cache", cache)
        return cache

    # ------------------------------------------------------------------
    def _detect_backend(self) -> None:
        """Select backend based on configuration and availability."""

        global _binance_import_attempted
        choice = (self.config.get("EXCHANGE_BACKEND") or os.getenv("EXCHANGE_BACKEND", "ccxt")).lower()
        if choice == "auto" and not _binance_import_attempted:
            _binance_import_attempted = True
            try:
                __import__("binance")  # type: ignore
                self.backend = "binance_sdk"
                return
            except Exception:
                pass
        elif choice not in {"ccxt", "auto"}:
            logging.warning("adapter | unsupported backend %s; using ccxt", choice)
        self._activate_ccxt()

    # ------------------------------------------------------------------
    def _activate_ccxt(self) -> None:
        """Instantiate a ccxt Binance class with OHLCV support."""

        global _ccxt
        if _ccxt is None:  # pragma: no cover - exercised in tests
            try:
                _ccxt = importlib.import_module("ccxt")  # type: ignore
            except Exception as exc:  # pragma: no cover - import failure
                logging.warning("adapter | ccxt import failed: %s", exc)
                raise AdapterInitError("ccxt not available") from exc

        try:  # pragma: no cover - logging only
            from importlib.metadata import version as _pkg_version

            ccxt_version = _pkg_version("ccxt")
        except Exception:  # pragma: no cover - fallback when metadata missing
            ccxt_version = getattr(_ccxt, "__version__", "?")
        logging.info("adapter | ccxt version: %s", ccxt_version)

        candidates: list[tuple[str, Callable[[dict], Any]]] = []

        if self.futures:
            if self.exchange_id.startswith("bybit"):
                options: dict[str, str] = {"defaultType": "linear"}
            else:
                options = {"defaultType": "future"}
        else:
            options = {"defaultType": "spot"}

        def add_candidate(name: str) -> None:
            ctor = getattr(_ccxt, name, None)
            if callable(ctor) and name not in {n for n, _ in candidates}:
                candidates.append((name, ctor))

        # Используем строго указанный exchange_id. Без fallback на Binance.
        normalized_id = self.exchange_id or ""
        if normalized_id == "bybit":
            candidate_ids: list[str] = ["bybit"]
        elif normalized_id:
            candidate_ids = [normalized_id]
        else:
            # по умолчанию выбираем bybit для futures
            candidate_ids = ["bybit"]

        for candidate_id in candidate_ids:
            add_candidate(candidate_id)

        cfg: dict[str, Any] = {"enableRateLimit": True}
        for key in ("apiKey", "secret"):
            if key in self.config and self.config[key]:
                cfg[key] = self.config[key]

        last_err: Exception | None = None
        tried: list[str] = []
        for name, ctor in candidates:
            tried.append(name)
            try:
                ex = ctor({**cfg, "options": options})
                if str(getattr(ex, "id", "") or "").lower() == "bybit":
                    current_options = dict(getattr(ex, "options", {}) or {})
                    current_options["defaultType"] = "linear"
                    ex.options = current_options
                if hasattr(ex, "set_sandbox_mode"):
                    ex.set_sandbox_mode(self.sandbox)

                mkts: dict[str, Any] = {}
                for _ in range(3):
                    try:
                        mkts = ex.load_markets(True) or {}
                        break
                    except Exception as exc:  # pragma: no cover - logging only
                        logging.warning("adapter | %s load_markets failed: %s", name, exc)
                        if self.sandbox and hasattr(ex, "set_sandbox_mode"):
                            try:
                                ex.set_sandbox_mode(False)
                                mkts = ex.load_markets(True) or {}
                            except Exception as exc2:  # pragma: no cover - logging only
                                logging.warning(
                                    "adapter | %s load_markets(nosandbox) failed: %s", name, exc2
                                )
                                mkts = {}
                            finally:
                                ex.set_sandbox_mode(self.sandbox)
                        time.sleep(0.5)
                if not mkts:
                    logging.warning(
                        "adapter | markets empty after load; continuing with empty markets"
                    )

                features = getattr(ex, "has", None)
                if features and features.get("fetchOHLCV") is True:
                    fetch_supported = True
                else:
                    fetch_supported = any(
                        hasattr(ex, method_name)
                        for method_name in ("fetch_ohlcv", "fetchOHLCV")
                    )
                if not fetch_supported:
                    raise AdapterInitError(f"{name}: fetchOHLCV not available")

                self.x = ex
                self.backend = "ccxt"
                self.ccxt_id = name
                self.markets_loaded_at = time.time()
                validate_api(self.x)
                logging.info("adapter | switched backend to ccxt:%s", name)
                logging.info(
                    "adapter | backend=%s sandbox=%s futures=%s ccxt=%s markets=%s",
                    self.backend,
                    self.sandbox,
                    self.futures,
                    ccxt_version,
                    len(mkts),
                )
                return
            except Exception as exc:  # pragma: no cover - logging only
                last_err = exc
                logging.warning("adapter | %s unusable: %r", name, exc)

        msg = (
            f"No usable CCXT exchange class with fetch_ohlcv (ccxt={ccxt_version}). "
            f"Tried: {tried}"
        )
        raise AdapterInitError(msg) from last_err

    # ------------------------------------------------------------------
    def load_markets_safe(self) -> bool:
        """Ensure that CCXT markets are available, retrying if needed."""

        if not getattr(self, "x", None):
            return False

        now = time.time()
        loaded_at = getattr(self, "markets_loaded_at", 0.0)
        if loaded_at and now - loaded_at < 15 * 60:
            return True

        markets: dict[str, Any] = {}
        for _ in range(3):
            try:
                markets = self.x.load_markets(True) or {}
                try:
                    setattr(self.x, "markets", markets)
                except Exception:
                    pass
                self.markets_loaded_at = time.time()
            except Exception as exc:  # pragma: no cover - logging only
                logging.warning("adapter | load_markets failed: %s", exc)
                markets = {}
            if markets:
                return True
            if self.sandbox and hasattr(self.x, "set_sandbox_mode"):
                try:
                    self.x.set_sandbox_mode(False)
                    markets = self.x.load_markets(True) or {}
                    try:
                        setattr(self.x, "markets", markets)
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - logging only
                    logging.warning("adapter | load_markets(nosandbox) failed: %s", exc)
                    markets = {}
                finally:
                    self.x.set_sandbox_mode(self.sandbox)
                if markets:
                    self.markets_loaded_at = time.time()
                    return True
            time.sleep(0.5)

        if not markets:
            self._warn_once(("markets", "empty"), "adapter | markets empty after load")
            return False
        return True

    # ------------------------------------------------------------------
    def _verify_exchange_options(self) -> None:
        """Warn if configured exchange options do not match trading mode."""

        if not getattr(self, "x", None):
            return

        opts = getattr(self.x, "options", {}) or {}
        if self.futures:
            dt = opts.get("defaultType")
            if dt and dt not in {"future", "swap"}:
                logging.warning("adapter | expected futures defaultType, got %s", dt)
            pt = opts.get("defaultSubType") or opts.get("productType")
            if pt and pt not in {"linear", "inverse", "usdm"}:
                logging.warning("adapter | unexpected futures productType %s", pt)
        else:
            dt = opts.get("defaultType")
            if dt and dt != "spot":
                logging.warning("adapter | expected spot defaultType, got %s", dt)

    # ------------------------------------------------------------------
    def _fetch_ohlcv_call(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        request_params: dict | None = None,
    ) -> list[list] | None:
        ex = getattr(self, "x", None)
        if not ex:
            raise AdapterOHLCVUnavailable("exchange unavailable")
        if request_params:
            try:
                return ex.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    params=request_params,
                )
            except TypeError:
                try:
                    return ex.fetch_ohlcv(
                        symbol,
                        timeframe,
                        None,
                        limit,
                        request_params,
                    )
                except TypeError:
                    return ex.fetch_ohlcv(symbol, timeframe, limit)
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except TypeError:
            return ex.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, *, limit: int = 500
    ) -> list[list] | None:
        if not symbol:
            raise AdapterOHLCVUnavailable("symbol required")
        if not timeframe:
            raise AdapterOHLCVUnavailable("timeframe required")

        ccxt_symbol = self._ccxt_symbol(symbol)

        cache_key = (symbol, timeframe, int(limit or 0))
        now = time.time()
        cache = self._get_ohlcv_cache()
        cached = cache.get(cache_key)
        if cached and now - cached[0] < 60:
            return [row[:] for row in cached[1]]

        markets = getattr(getattr(self, "x", None), "markets", None)
        if not markets:
            if not self.load_markets_safe():
                data = self._fetch_ohlcv_from_csv(symbol, timeframe, limit)
                if data:
                    logging.debug(
                        "adapter | fetch_ohlcv csv fallback success file=%s params=%s",
                        self._csv_path(symbol, timeframe),
                        {"symbol": symbol, "timeframe": timeframe, "limit": limit},
                    )
                    return data
                logging.warning("adapter | ohlcv unavailable: markets empty for %s", symbol)
                raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")
            markets = getattr(self.x, "markets", None)
            if not markets:
                data = self._fetch_ohlcv_from_csv(symbol, timeframe, limit)
                if data:
                    logging.debug(
                        "adapter | fetch_ohlcv csv fallback success file=%s params=%s",
                        self._csv_path(symbol, timeframe),
                        {"symbol": symbol, "timeframe": timeframe, "limit": limit},
                    )
                    return data
                logging.warning("adapter | ohlcv unavailable: markets empty for %s", symbol)
                raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")

        log_params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        exchange = getattr(self, "x", None) or getattr(self, "exchange", None)
        detected_cat = detect_market_category(exchange, symbol) if exchange else None
        cat_norm = str(detected_cat or "").lower()
        if not cat_norm or cat_norm == "swap":
            cat_norm = "linear"
        request_cat = cat_norm if cat_norm in {"linear", "inverse"} else None
        cat_label = request_cat or (cat_norm if cat_norm else "none")
        request_params: dict[str, Any] | None = {"category": request_cat} if request_cat else None
        no_data = self._ensure_no_data_store()

        try:
            data = self._fetch_ohlcv_call(ccxt_symbol, timeframe, limit, request_params)
            if not data:
                log_once(
                    "warning",
                    f"adapter | fetch_ohlcv empty: {symbol} {timeframe} cat={cat_label}",
                )
                no_data[symbol] = time.time()
                return None
            url = getattr(self.x, "last_request_url", "unknown")
            status = getattr(self.x, "last_http_status_code", "unknown")
            logging.debug(
                "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                url,
                log_params
                | ({"request_params": request_params} if request_params else {}),
                status,
            )
            self._store_ohlcv_cache(cache_key, data)
            no_data.pop(symbol, None)
            return data
        except AttributeError:
            self._activate_ccxt()
            markets = getattr(self.x, "markets", None)
            if not markets and not self.load_markets_safe():
                logging.warning("adapter | ohlcv unavailable: markets empty for %s", symbol)
                raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")
            features = getattr(self.x, "has", None)
            supported = False
            if features and features.get("fetchOHLCV") is True:
                supported = True
            else:
                supported = any(
                    hasattr(self.x, method_name)
                    for method_name in ("fetch_ohlcv", "fetchOHLCV")
                )
            if not supported:
                logging.warning("adapter | ohlcv unavailable: markets empty for %s", symbol)
                raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")
            try:
                data = self._fetch_ohlcv_call(ccxt_symbol, timeframe, limit, request_params)
                if not data:
                    log_once(
                        "warning",
                        f"adapter | fetch_ohlcv empty: {symbol} {timeframe} cat={cat_label}",
                    )
                    no_data[symbol] = time.time()
                    return None
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                    url,
                    log_params
                    | ({"request_params": request_params} if request_params else {}),
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
                no_data.pop(symbol, None)
                return data
            except Exception as exc:  # pragma: no cover - logging only
                if self._is_empty_result_exception(exc):
                    log_once(
                        "warning",
                        f"adapter | fetch_ohlcv failed: {symbol} {timeframe} cat={cat_label} ({exc})",
                    )
                    no_data[symbol] = time.time()
                    return None
                return self._handle_fetch_failure(
                    symbol,
                    timeframe,
                    limit,
                    log_params,
                    exc,
                    request_params=request_params,
                    request_symbol=ccxt_symbol,
                )
        except Exception as exc:  # pragma: no cover - logging only
            if self._is_empty_result_exception(exc):
                log_once(
                    "warning",
                    f"adapter | fetch_ohlcv failed: {symbol} {timeframe} cat={cat_label} ({exc})",
                )
                no_data[symbol] = time.time()
                return None
            if self._is_rate_limited(exc):
                time.sleep(2)
                try:
                    data = self._fetch_ohlcv_call(
                        ccxt_symbol, timeframe, limit, request_params
                    )
                except Exception as retry_exc:  # pragma: no cover - logging only
                    if self._is_empty_result_exception(retry_exc):
                        log_once(
                            "warning",
                            f"adapter | fetch_ohlcv failed: {symbol} {timeframe} cat={cat_label} ({retry_exc})",
                        )
                        no_data[symbol] = time.time()
                        return None
                    return self._handle_fetch_failure(
                        symbol,
                        timeframe,
                        limit,
                        log_params,
                        retry_exc,
                        request_params=request_params,
                        request_symbol=ccxt_symbol,
                    )
                if not data:
                    log_once(
                        "warning",
                        f"adapter | fetch_ohlcv empty: {symbol} {timeframe} cat={cat_label}",
                    )
                    no_data[symbol] = time.time()
                    return None
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv retry success url=%s params=%s status=%s",
                    url,
                    log_params
                    | ({"request_params": request_params} if request_params else {}),
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
                no_data.pop(symbol, None)
                return data
            return self._handle_fetch_failure(
                symbol,
                timeframe,
                limit,
                log_params,
                exc,
                request_params=request_params,
                request_symbol=ccxt_symbol,
            )

        raise AdapterOHLCVUnavailable("backend unsupported")

    def _csv_path(self, symbol: str, timeframe: str) -> Path:
        csv_dir = Path(self.config.get("csv_dir", "data"))
        return csv_dir / f"{symbol.replace('/', '')}_{timeframe}.csv"

    def _store_ohlcv_cache(self, key: tuple[str, str, int], data: list[list]) -> None:
        try:
            snapshot = [list(row) for row in data]
        except Exception:
            snapshot = data
        cache = self._get_ohlcv_cache()
        cache[key] = (time.time(), snapshot)

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        return (
            "too many requests" in msg
            or "429" in msg
            or "rate limit" in msg
            or "-1003" in msg
            or status_code == 429
        )

    @staticmethod
    def _is_empty_result_exception(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "empty result" in msg:
            return True
        if "unknown" in msg and "status" in msg:
            return True
        if "unknown status" in msg:
            return True
        return False

    def _fetch_ohlcv_from_csv(self, symbol: str, timeframe: str, limit: int) -> list[list]:
        path = self._csv_path(symbol, timeframe)
        if not path.exists():
            return []
        rows: list[list] = []
        try:
            with path.open() as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:
                        continue
                    rows.append(
                        [
                            int(float(row[0])),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                        ]
                    )
        except Exception as exc:  # pragma: no cover - logging only
            logging.warning("adapter | csv load failed %s: %s", path, exc)
            return []
        trimmed = rows[-limit:]
        if trimmed:
            cache_key = (symbol, timeframe, int(limit or 0))
            self._store_ohlcv_cache(cache_key, trimmed)
        return trimmed

    def _handle_fetch_failure(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        params: dict,
        exc: Exception,
        *,
        request_params: dict | None = None,
        request_symbol: str | None = None,
    ) -> list[list] | None:
        url = getattr(self.x, "last_request_url", "unknown")
        status = getattr(self.x, "last_http_status_code", "unknown")
        logging.warning(
            "adapter | fetch_ohlcv failed url=%s params=%s status=%s error=%s",
            url,
            params | ({"request_params": request_params} if request_params else {}),
            status,
            exc,
        )

        call_symbol = request_symbol or symbol
        no_data = self._ensure_no_data_store()

        msg = str(exc).lower()
        if self.sandbox and ("timeframe" in msg or "unsupported" in msg):
            if hasattr(self.x, "set_sandbox_mode"):
                try:
                    self.x.set_sandbox_mode(False)
                    data = self._fetch_ohlcv_call(call_symbol, timeframe, limit, request_params)
                    url = getattr(self.x, "last_request_url", "unknown")
                    status = getattr(self.x, "last_http_status_code", "unknown")
                    logging.debug(
                        "adapter | fetch_ohlcv live retry success url=%s params=%s status=%s",
                        url,
                        params | ({"request_params": request_params} if request_params else {}),
                        status,
                    )
                    self.x.set_sandbox_mode(True)
                    return data
                except Exception as live_exc:  # pragma: no cover - logging only
                    url = getattr(self.x, "last_request_url", "unknown")
                    status = getattr(self.x, "last_http_status_code", "unknown")
                    logging.warning(
                        "adapter | fetch_ohlcv live retry failed url=%s params=%s status=%s error=%s",
                        url,
                        params,
                        status,
                        live_exc,
                    )
                    try:
                        self.x.set_sandbox_mode(True)
                    except Exception:
                        pass

        data = self._fetch_ohlcv_from_csv(symbol, timeframe, limit)
        if data:
            logging.debug(
                "adapter | fetch_ohlcv csv fallback success file=%s params=%s",
                self._csv_path(symbol, timeframe),
                params | ({"request_params": request_params} if request_params else {}),
            )
            self._store_ohlcv_cache((symbol, timeframe, int(limit or 0)), data)
            no_data.pop(symbol, None)
            return data

        self._warn(
            "ohlcv_fail",
            f"{symbol}:{timeframe}",
            f"adapter | fetch_ohlcv giving up symbol={symbol} tf={timeframe} error={exc}",
        )
        no_data[symbol] = time.time()
        return None

    # ------------------------------------------------------------------
    def fetch_multi_ohlcv(self, symbol: str, timeframes: list[str], limit: int = 300) -> dict[str, list[list]]:
        if getattr(self, "x", None) is not None and not self.load_markets_safe():
            raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")

        results: dict[str, list[list]] = {}
        failed: list[str] = []
        statuses: list[str] = []
        for tf in timeframes:
            try:
                data = self.fetch_ohlcv(symbol, tf, limit=limit)
                if data:
                    results[tf] = data
                    statuses.append(f"{tf}:ok")
                else:
                    failed.append(tf)
                    statuses.append(f"{tf}:empty")
            except Exception:
                failed.append(tf)
                statuses.append(f"{tf}:fail")
        logging.info("adapter | %s | fetched %s", symbol, ", ".join(statuses))
        if failed:
            logging.warning("adapter | %s | failed %s", symbol, ",".join(failed))
        return results

    # ------------------------------------------------------------------
    def _normalize_symbol_key(self, symbol: str | None) -> str | None:
        """Return a canonical symbol representation for comparisons."""

        if not isinstance(symbol, str):
            return None

        candidate = self._ccxt_symbol(symbol)
        if not isinstance(candidate, str) or not candidate:
            candidate = symbol

        candidate = candidate.split(":", 1)[0]
        candidate = candidate.replace("/", "")
        candidate = candidate.strip().upper()
        return candidate or None

    def _normalize_position_symbol(self, position: dict[str, Any] | None) -> str | None:
        """Extract and normalise the symbol identifier from a position."""

        if not isinstance(position, dict):
            return None

        candidates: list[str] = []
        symbol = position.get("symbol")
        if isinstance(symbol, str) and symbol:
            candidates.append(symbol)

        info = position.get("info")
        if isinstance(info, dict) and info:
            for key in (
                "symbol",
                "symbolName",
                "instrumentId",
                "instrument_id",
                "instId",
                "productId",
                "pair",
                "market",
            ):
                value = info.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)

        for candidate in candidates:
            normalized = self._normalize_symbol_key(candidate)
            if normalized:
                return normalized

        return None

    def fetch_positions(
        self,
        symbols: list[str] | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        """Fetch open positions while applying Bybit specific defaults."""

        exchange = getattr(self, "x", None)
        if exchange is None or not hasattr(exchange, "fetch_positions"):
            return []

        symbol_arg: Any = None
        first_symbol: str | None = None
        if isinstance(symbols, (list, tuple, set)):
            normalized_symbols: list[str] = []
            for sym in symbols:
                if not sym:
                    continue
                if first_symbol is None:
                    first_symbol = sym
                normalized = self._ccxt_symbol(sym)
                if isinstance(normalized, str) and normalized:
                    normalized_symbols.append(normalized)
                else:
                    normalized_symbols.append(sym)
            if normalized_symbols:
                symbol_arg = normalized_symbols
        elif isinstance(symbols, str) and symbols:
            first_symbol = symbols
            normalized = self._ccxt_symbol(symbols)
            symbol_arg = [normalized] if isinstance(normalized, str) and normalized else [symbols]

        detected_cat = detect_market_category(exchange, first_symbol or "")
        cat_norm = str(detected_cat or "").lower()
        if not cat_norm or cat_norm == "swap":
            cat_norm = "linear"
        request_cat = cat_norm if cat_norm in {"linear", "inverse"} else None
        cat_label = request_cat or (cat_norm if cat_norm else "none")

        call_params: dict[str, Any] = dict(params or {})
        if request_cat:
            call_params["category"] = request_cat
        else:
            call_params.pop("category", None)

        try:
            try:
                positions = exchange.fetch_positions(symbol_arg, params=call_params)
            except TypeError:
                if symbol_arg is None:
                    positions = exchange.fetch_positions(params=call_params)
                else:
                    positions = exchange.fetch_positions(symbol_arg, call_params)
        except Exception as exc:
            message = str(exc).lower()
            symbol_label = first_symbol or "ALL"
            if "empty" in message or "unknown" in message or "status" in message:
                log_once(
                    "warning",
                    f"adapter | fetch_positions failed: {symbol_label} cat={cat_label} ({exc})",
                )
                return []
            raise

        if not positions:
            symbol_label = first_symbol or "ALL"
            log_once(
                "warning",
                f"adapter | fetch_positions empty: {symbol_label} cat={cat_label}",
            )
            return []

        if first_symbol:
            target = self._normalize_symbol_key(first_symbol)
            if target:
                filtered = [
                    position
                    for position in positions
                    if self._normalize_position_symbol(position) == target
                ]
                if not filtered:
                    log_once(
                        "warning",
                        f"adapter | fetch_positions empty: {first_symbol} cat={cat_label}",
                    )
                    return []
                positions = filtered

        return positions

    # ------------------------------------------------------------------
    def fetch_open_orders(self, symbol: str | None = None) -> tuple[int, list]:
        """Return ``(count, ids)`` of open orders without raising."""
        def _call_fetch(ex, sym, params):
            if not ex or not hasattr(ex, "fetch_open_orders"):
                return []
            if params:
                attempts = []
                if sym is not None:
                    attempts = [
                        lambda: ex.fetch_open_orders(sym, None, None, params),
                        lambda: ex.fetch_open_orders(sym, None, params),
                        lambda: ex.fetch_open_orders(sym, params),
                    ]
                else:
                    attempts = [
                        lambda: ex.fetch_open_orders(None, None, None, params),
                        lambda: ex.fetch_open_orders(None, None, params),
                        lambda: ex.fetch_open_orders(None, params),
                        lambda: ex.fetch_open_orders(params),
                    ]
                for attempt in attempts:
                    try:
                        result = attempt()
                        if result is not None:
                            return result
                    except TypeError:
                        continue
            if sym is not None:
                return ex.fetch_open_orders(sym)
            return ex.fetch_open_orders()

        try:
            ex = getattr(self, "x", None)
            normalized_symbol = self._ccxt_symbol(symbol) if symbol else symbol
            params = self._default_params(symbol=symbol)
            if ex and hasattr(ex, "fetch_open_orders"):
                sym_arg = normalized_symbol if symbol is not None else None
                orders = _call_fetch(ex, sym_arg, params) or []
                ids = [o.get("id") or o.get("orderId") for o in orders]
                return len(ids), ids
        except Exception as exc:  # pragma: no cover - logging only
            if self._is_rate_limited(exc):
                logging.warning("adapter | fetch_open_orders rate limited: %s", exc)
                time.sleep(1.0)
                try:
                    ex = getattr(self, "x", None)
                    normalized_symbol = self._ccxt_symbol(symbol) if symbol else symbol
                    params = self._default_params(symbol=symbol)
                    if ex and hasattr(ex, "fetch_open_orders"):
                        sym_arg = normalized_symbol if symbol is not None else None
                        orders = _call_fetch(ex, sym_arg, params) or []
                        ids = [o.get("id") or o.get("orderId") for o in orders]
                        return len(ids), ids
                except Exception as retry_exc:  # pragma: no cover - logging only
                    logging.warning("adapter | fetch_open_orders retry failed: %s", retry_exc)
            logging.warning("adapter | fetch_open_orders failed: %s", exc)
        return (0, [])

    # ------------------------------------------------------------------
    def cancel_open_orders(self, symbol: str | None = None) -> tuple[int, list]:
        """Cancel open orders and return ``(count, ids)``."""

        cancelled_ids: list = []
        try:
            if not getattr(self, "x", None):
                return (0, [])

            cnt, ids = self.fetch_open_orders(symbol)
            if not cnt:
                return (0, [])

            params = self._default_params(symbol=symbol)
            if hasattr(self.x, "cancel_all_orders"):
                try:
                    if params:
                        if symbol:
                            self.x.cancel_all_orders(symbol, params)
                        else:
                            self.x.cancel_all_orders(params)
                    else:
                        self.x.cancel_all_orders(symbol) if symbol else self.x.cancel_all_orders()
                except TypeError:
                    try:
                        self.x.cancel_all_orders(symbol) if symbol else self.x.cancel_all_orders()
                    except Exception:
                        pass
                except Exception:
                    pass
                cancelled_ids = ids
                return (len(cancelled_ids), cancelled_ids)

            for oid in ids:
                try:
                    if params:
                        try:
                            self.x.cancel_order(oid, symbol, params)
                        except TypeError:
                            self.x.cancel_order(oid, symbol)
                    else:
                        self.x.cancel_order(oid, symbol)
                    cancelled_ids.append(oid)
                except Exception as exc:  # pragma: no cover - logging only
                    logging.warning("adapter | cancel_order failed: %s", exc)
            return (len(cancelled_ids), cancelled_ids)

        except Exception as exc:  # pragma: no cover - logging only
            logging.warning("adapter | cancel_open_orders failed: %s", exc)
        return (0, [])

    # ------------------------------------------------------------------
    def load_markets(self, *a, **k):  # pragma: no cover - simple wrapper
        if self.backend == "binance_sdk" and getattr(self, "sdk", None):
            try:
                info = self.sdk.futures_exchange_info()
                symbols = {s.get("symbol", "").replace("USDT", "/USDT") for s in info.get("symbols", [])}
                return symbols
            except Exception:
                return set()
        if not self.load_markets_safe():
            return set()
        try:
            markets = self.x.load_markets(False) or {}
        except Exception:
            markets = {}

        names: set[str] = set()
        if isinstance(markets, dict):
            iterable = markets.items()
        else:
            iterable = ((name, {}) for name in markets)

        for name, info in iterable:
            if not name:
                continue
            if isinstance(name, str):
                names.add(name)
                if ":" in name:
                    names.add(name.split(":", 1)[0])
            if not isinstance(info, dict):
                continue
            symbol = info.get("symbol")
            if isinstance(symbol, str) and symbol:
                names.add(symbol)
                if ":" in symbol:
                    names.add(symbol.split(":", 1)[0])
            base = info.get("base")
            quote = info.get("quote")
            if isinstance(base, str) and isinstance(quote, str) and base and quote:
                names.add(f"{base}/{quote}")
        return names

    # ------------------------------------------------------------------
    def _warn_once(self, key: tuple[str, str], message: str) -> None:
        now = time.time()
        last = self.last_warn_at.get(key, 0.0)
        if now - last >= 60.0:
            logging.warning(message)
            self.last_warn_at[key] = now

    def _warn(self, topic: str, key: str, message: str) -> None:
        self._warn_once((topic, key), message)

    def log_once(self, level: str, message: str, *, interval: float = 60.0) -> None:
        cache = getattr(self, "_log_once_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_log_once_cache", cache)
        now = time.time()
        key = (level, message)
        last = cache.get(key, 0.0)
        if now - last < interval:
            return
        cache[key] = now
        logger = getattr(logging, level, None)
        if callable(logger):
            logger(message)
        else:
            logging.log(getattr(logging, level.upper(), logging.INFO), message)

    # convenience -------------------------------------------------------
    def supports_symbol(self, symbol: str, markets: Optional[set[str]] = None) -> bool:
        try:
            markets = markets or self.load_markets()
        except Exception:  # pragma: no cover - defensive
            self._warn_once(("markets", "unavailable"), "empty markets set, skipping symbol-filter")
            return True
        if not markets:
            self._warn_once(("markets", "empty"), "empty markets set, skipping symbol-filter")
            return True
        return symbol in markets


__all__ = [
    "ExchangeAdapter",
    "AdapterInitError",
    "AdapterOHLCVUnavailable",
    "AdapterOrdersUnavailable",
    "to_ccxt",
    "to_sdk",
]

