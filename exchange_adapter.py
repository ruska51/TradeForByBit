from __future__ import annotations

"""Light-weight exchange adapter using only the CCXT backend."""

import logging
import os
import time
import importlib
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from logging_utils import log


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


def set_valid_leverage(exchange, symbol: str, desired: int):
    """Apply leverage respecting exchange limits."""

    import logging

    market: dict[str, Any] = {}
    try:
        maybe_market = exchange.market(symbol)  # type: ignore[attr-defined]
        if isinstance(maybe_market, dict):
            market = maybe_market
    except Exception:  # pragma: no cover - defensive
        market = {}

    limits = (market.get("limits", {}) or {}) if isinstance(market, dict) else {}
    lev = limits.get("leverage", {}) or {}

    min_l = int(lev.get("min") or 1)
    max_l = int(lev.get("max") or 20)
    step = int(lev.get("step") or 1)

    L = max(min(desired, max_l), min_l)
    L = int((L // step) * step)

    market_type = market.get("type") if isinstance(market, dict) else None
    category: str | None = None
    if isinstance(market_type, str):
        lower_type = market_type.lower()
        if lower_type in {"linear", "inverse"}:
            category = lower_type
    if category is None and isinstance(market, dict):
        if bool(market.get("linear")):
            category = "linear"
        elif bool(market.get("inverse")):
            category = "inverse"
        else:
            info = market.get("info") if isinstance(market.get("info"), dict) else {}
            if isinstance(info, dict):
                raw_category = str(
                    info.get("category")
                    or info.get("contractType")
                    or info.get("contract_type")
                    or ""
                ).lower()
                if raw_category in {"linear", "inverse"}:
                    category = raw_category

    params: dict[str, Any] = {}
    if getattr(exchange, "id", "").lower() == "bybit" and category:
        params = {"category": category}
    try:
        if hasattr(exchange, "set_leverage"):
            try:
                exchange.set_leverage(L, symbol, params)
            except TypeError:
                exchange.set_leverage(L, symbol)
        else:
            try:
                exchange.setLeverage(L, symbol, params)  # type: ignore[attr-defined]
            except TypeError:
                exchange.setLeverage(L, symbol)  # type: ignore[attr-defined]
        logging.info("leverage | %s | set %s", symbol, L)
    except Exception as e:  # pragma: no cover - network errors
        logging.warning(
            "leverage | %s | failed to set leverage %s: %s", symbol, L, e
        )
        return None
    return L


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
                options: dict[str, str] = {"defaultType": "swap", "defaultSubType": "linear"}
            else:
                options = {"defaultType": "future", "defaultSubType": "linear"}
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
            if dt and dt not in {"future", "swap", "linear"}:
                logging.warning("adapter | expected futures defaultType, got %s", dt)
            pt = opts.get("defaultSubType") or opts.get("productType")
            if pt and pt not in {"linear", "usdm"}:
                logging.warning("adapter | unexpected futures productType %s", pt)
        else:
            dt = opts.get("defaultType")
            if dt and dt != "spot":
                logging.warning("adapter | expected spot defaultType, got %s", dt)

    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> list[list]:
        if not symbol:
            raise AdapterOHLCVUnavailable("symbol required")
        if not timeframe:
            raise AdapterOHLCVUnavailable("timeframe required")

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

        params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}

        try:
            data = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)  # type: ignore[operator]
            if not data:
                raise AdapterOHLCVUnavailable("empty result")
            url = getattr(self.x, "last_request_url", "unknown")
            status = getattr(self.x, "last_http_status_code", "unknown")
            logging.debug(
                "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                url,
                params,
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
                data = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not data:
                    raise AdapterOHLCVUnavailable("empty result")
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv success url=%s params=%s status=%s",
                    url,
                    params,
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
                return data
            except Exception as exc:  # pragma: no cover - logging only
                return self._handle_fetch_failure(symbol, timeframe, limit, params, exc)
        except Exception as exc:  # pragma: no cover - logging only
            if self._is_rate_limited(exc):
                time.sleep(2)
                try:
                    data = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)  # type: ignore[operator]
                except Exception as retry_exc:  # pragma: no cover - logging only
                    return self._handle_fetch_failure(symbol, timeframe, limit, params, retry_exc)
                if not data:
                    return self._handle_fetch_failure(symbol, timeframe, limit, params, exc)
                url = getattr(self.x, "last_request_url", "unknown")
                status = getattr(self.x, "last_http_status_code", "unknown")
                logging.debug(
                    "adapter | fetch_ohlcv retry success url=%s params=%s status=%s",
                    url,
                    params,
                    status,
                )
                self._store_ohlcv_cache(cache_key, data)
                return data
            return self._handle_fetch_failure(symbol, timeframe, limit, params, exc)

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
    ) -> list[list]:
        url = getattr(self.x, "last_request_url", "unknown")
        status = getattr(self.x, "last_http_status_code", "unknown")
        logging.warning(
            "adapter | fetch_ohlcv failed url=%s params=%s status=%s error=%s",
            url,
            params,
            status,
            exc,
        )

        msg = str(exc).lower()
        if self.sandbox and ("timeframe" in msg or "unsupported" in msg):
            if hasattr(self.x, "set_sandbox_mode"):
                try:
                    self.x.set_sandbox_mode(False)
                    data = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                    url = getattr(self.x, "last_request_url", "unknown")
                    status = getattr(self.x, "last_http_status_code", "unknown")
                    logging.debug(
                        "adapter | fetch_ohlcv live retry success url=%s params=%s status=%s",
                        url,
                        params,
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
                params,
            )
            self._store_ohlcv_cache((symbol, timeframe, int(limit or 0)), data)
            return data

        raise AdapterOHLCVUnavailable(str(exc)) from exc

    # ------------------------------------------------------------------
    def fetch_multi_ohlcv(self, symbol: str, timeframes: list[str], limit: int = 300) -> dict[str, list[list]]:
        if getattr(self, "x", None) is not None and not self.load_markets_safe():
            raise AdapterOHLCVUnavailable(f"markets empty for {symbol}")

        results: dict[str, list[list]] = {}
        failed: list[str] = []
        statuses: list[str] = []
        for tf in timeframes:
            try:
                results[tf] = self.fetch_ohlcv(symbol, tf, limit=limit)
                statuses.append(f"{tf}:ok")
            except Exception:
                failed.append(tf)
                statuses.append(f"{tf}:fail")
        logging.info("adapter | %s | fetched %s", symbol, ", ".join(statuses))
        if failed:
            logging.warning("adapter | %s | failed %s", symbol, ",".join(failed))
        return results

    # ------------------------------------------------------------------
    def fetch_open_orders(self, symbol: str | None = None) -> tuple[int, list]:
        """Return ``(count, ids)`` of open orders without raising."""
        try:
            ex = getattr(self, "x", None)
            if ex and hasattr(ex, "fetch_open_orders"):
                orders = ex.fetch_open_orders(symbol) if symbol else ex.fetch_open_orders()
                orders = orders or []
                ids = [o.get("id") or o.get("orderId") for o in orders]
                return len(ids), ids
        except Exception as exc:  # pragma: no cover - logging only
            if self._is_rate_limited(exc):
                logging.warning("adapter | fetch_open_orders rate limited: %s", exc)
                time.sleep(1.0)
                try:
                    ex = getattr(self, "x", None)
                    if ex and hasattr(ex, "fetch_open_orders"):
                        orders = (
                            ex.fetch_open_orders(symbol)
                            if symbol
                            else ex.fetch_open_orders()
                        )
                        orders = orders or []
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

            if hasattr(self.x, "cancel_all_orders"):
                try:
                    self.x.cancel_all_orders(symbol) if symbol else self.x.cancel_all_orders()
                except Exception:
                    pass
                cancelled_ids = ids
                return (len(cancelled_ids), cancelled_ids)

            for oid in ids:
                try:
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
        return set(markets.keys())

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

