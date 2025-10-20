from __future__ import annotations

"""Light-weight exchange adapter using only the CCXT backend."""

import logging
import os
import time
import importlib
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from logging_utils import (
    log,
    normalize_bybit_category,
    detect_market_category,
)


LEVERAGE_SKIPPED = object()


def _ccxt_symbol(symbol: str) -> str:
    """Return a shortened CCXT symbol without settlement suffixes."""

    if not isinstance(symbol, str):
        return symbol
    if ":" in symbol:
        return symbol.split(":", 1)[0]
    return symbol


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


def set_valid_leverage(exchange, symbol: str, leverage: int | float):
    """Set leverage for derivatives while skipping spot markets when required."""

    try:
        L = max(1, int(leverage))
    except Exception:
        try:
            L = max(1, int(float(leverage)))
        except Exception:
            L = 1

    exid = str(getattr(exchange, "id", "") or "").lower()
    is_bybit = exid == "bybit"

    def _resolve_short_symbol(sym: str):
        if not isinstance(sym, str):
            return sym
        for source in (
            getattr(exchange, "adapter", None),
            exchange,
        ):
            converter = getattr(source, "_ccxt_symbol", None)
            if not callable(converter):
                continue
            try:
                mapped = converter(sym)
            except Exception:
                mapped = None
            if isinstance(mapped, str) and mapped:
                sym = mapped
                break
        return _ccxt_symbol(sym)

    cat = detect_market_category(exchange, symbol) or "linear"
    cat = str(cat or "").lower()
    if not cat or cat == "swap":
        cat = "linear"

    short_symbol = _resolve_short_symbol(symbol)
    params = {"category": cat, "buyLeverage": L, "sellLeverage": L}

    if is_bybit:
        if cat == "spot":
            return LEVERAGE_SKIPPED
        try:
            exchange.set_leverage(L, short_symbol, params)
            return L
        except Exception as e:  # pragma: no cover - network interaction
            logging.info(f"leverage | {symbol} | soft-skip: {e}")
            return None

    if cat == "spot":
        return LEVERAGE_SKIPPED

    market_meta: dict[str, Any] | None = None
    candidates: list[str] = []
    for candidate in (short_symbol, symbol):
        if isinstance(candidate, str) and candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            market_obj = exchange.market(candidate)
        except Exception:
            markets = getattr(exchange, "markets", {}) or {}
            market_obj = markets.get(candidate)
        if isinstance(market_obj, dict):
            market_meta = market_obj
            break

    if isinstance(market_meta, dict):
        derivative_hint = bool(
            market_meta.get("linear")
            or market_meta.get("inverse")
            or market_meta.get("swap")
        )
        market_type = str(market_meta.get("type") or "").lower()
        spot_hint = bool(market_meta.get("spot") or market_type == "spot")
        info = market_meta.get("info")
        if isinstance(info, dict):
            for key in ("category", "contractType", "productType", "market"):
                hint = normalize_bybit_category(info.get(key))
                if hint == "spot":
                    spot_hint = True
                elif hint in {"linear", "inverse"}:
                    derivative_hint = True
        type_hint = normalize_bybit_category(market_meta.get("type"))
        if type_hint == "spot":
            spot_hint = True
        elif type_hint in {"linear", "inverse"}:
            derivative_hint = True
        if spot_hint and not derivative_hint:
            return LEVERAGE_SKIPPED

    try:
        exchange.set_leverage(L, short_symbol)
        return L
    except Exception as exc:
        logging.info("exchange_adapter | leverage | %s | soft-skip: %s", symbol, exc)
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
        self._ohlcv_cache: dict[tuple[str, str, int], tuple[float, list[list]]]
        self._ohlcv_cache = {}

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
                options: dict[str, str] = {"defaultType": "swap"}
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
        request_params = self._default_params(
            include_position_idx=False, symbol=ccxt_symbol or symbol
        ) or None
        is_futures = bool(getattr(self, "futures", getattr(self, "is_futures", False)))
        if is_futures:
            cat = str((request_params or {}).get("category") or "").lower()
            if cat in {"", "swap"}:
                cat = "linear"
            if cat == "spot":
                cat = "linear"
            if cat:
                request_params = dict(request_params or {})
                request_params["category"] = cat

        try:
            data = self._fetch_ohlcv_call(ccxt_symbol, timeframe, limit, request_params)
            if not data:
                self._warn(
                    "ohlcv_empty",
                    f"{symbol}:{timeframe}",
                    f"adapter | fetch_ohlcv empty result symbol={symbol} tf={timeframe}",
                )
                return None
            url = getattr(self.x, "last_request_url", "unknown")
            status = getattr(self.x, "last_http_status_code", "unknown")
            logging.debug(
                "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                url,
                log_params | ({"request_params": request_params} if request_params else {}),
                status,
            )
            self._store_ohlcv_cache(cache_key, data)
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
                    self._warn(
                        "ohlcv_empty",
                        f"{symbol}:{timeframe}",
                        f"adapter | fetch_ohlcv empty result symbol={symbol} tf={timeframe}",
                    )
                    return None
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                    url,
                    log_params | ({"request_params": request_params} if request_params else {}),
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
                return data
            except Exception as exc:  # pragma: no cover - logging only
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
            if self._is_rate_limited(exc):
                time.sleep(2)
                try:
                    data = self._fetch_ohlcv_call(
                        ccxt_symbol, timeframe, limit, request_params
                    )
                except Exception as retry_exc:  # pragma: no cover - logging only
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
                    self._warn(
                        "ohlcv_empty",
                        f"{symbol}:{timeframe}",
                        f"adapter | fetch_ohlcv empty result symbol={symbol} tf={timeframe}",
                    )
                    return None
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv retry success url=%s params=%s status=%s",
                    url,
                    log_params | ({"request_params": request_params} if request_params else {}),
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
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
            return data

        self._warn(
            "ohlcv_fail",
            f"{symbol}:{timeframe}",
            f"adapter | fetch_ohlcv giving up symbol={symbol} tf={timeframe} error={exc}",
        )
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
    def fetch_positions(
        self,
        symbols: list[str] | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        """Fetch open positions while applying Bybit specific defaults."""

        ex = getattr(self, "x", None)
        if ex is None or not hasattr(ex, "fetch_positions"):
            return []

        merged: dict[str, Any] = dict(params or {})
        cat_raw = str(merged.get("category") or "").lower()

        normalized_symbols: list[str] | None = None
        first_symbol: str | None = None
        if symbols:
            normalized_symbols = []
            for sym in symbols:
                if not sym:
                    continue
                mapped = self._ccxt_symbol(sym)
                normalized_symbols.append(mapped)
                if first_symbol is None:
                    first_symbol = sym

        if not cat_raw and first_symbol:
            detected = detect_market_category(ex, first_symbol)
            if detected:
                cat_raw = str(detected).lower()
        if not cat_raw and self.futures:
            cat_raw = "linear"

        if cat_raw in {"", "swap", "spot"}:
            category = "linear"
        elif cat_raw in {"linear", "inverse"}:
            category = cat_raw
        else:
            category = "linear"

        merged["category"] = category
        merged.pop("positionIdx", None)

        call_params = merged or None
        symbol_arg: Any = normalized_symbols

        attempts: list[Callable[[], Any]] = []
        if symbol_arg is not None:
            if call_params is not None:
                attempts.append(lambda: ex.fetch_positions(symbol_arg, call_params))
            attempts.append(lambda: ex.fetch_positions(symbol_arg))
        else:
            if call_params is not None:
                attempts.append(lambda: ex.fetch_positions(None, call_params))
            attempts.append(lambda: ex.fetch_positions())

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                result = attempt()
                if result is not None:
                    return result
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if call_params and "category" in call_params:
                    self._warn(
                        "positions",
                        str(first_symbol or symbol_arg or "ALL"),
                        f"adapter | fetch_positions warning symbol={first_symbol or 'ALL'} error={exc}",
                    )

        if last_error is not None:
            self._warn(
                "positions",
                str(first_symbol or symbol_arg or "ALL"),
                f"adapter | fetch_positions failed symbol={first_symbol or 'ALL'} error={last_error}",
            )
        return []

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

