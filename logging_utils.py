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

LOG_DIR = Path(__file__).resolve().parent / "logs"

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
    # Emit via both unified log helper and standard logging so that tests using
    # ``caplog`` reliably capture the message even when custom handlers are
    # installed during ``setup_logging``.
    log(logging.INFO, decision, symbol, msg)
    logging.getLogger().log(logging.INFO, msg)
    for h in logging.getLogger().handlers:
        stream = getattr(h, "stream", None)
        if getattr(stream, "write", None) and hasattr(stream, "getvalue"):
            try:
                stream.write(msg + "\n")
            except Exception:
                pass
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
    """Create an order with retry and PERCENT_PRICE handling."""
    if params is None:
        params = {}
    if not getattr(exchange, "markets", None):
        try:
            exchange.load_markets()
        except Exception:
            pass

    otype = order_type.lower()
    is_market_like = otype.endswith("market")
    side = side.lower()
    adj_price = None if is_market_like else price

    def _best_price():
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker.get("ask") if side == "buy" else ticker.get("bid")
        except Exception:
            return None

    def _apply_filters(b_price, target):
        try:
            market = exchange.market(symbol)
            filters = {f["filterType"]: f for f in market["info"].get("filters", [])}
            pf = filters.get("PERCENT_PRICE")
            if pf and b_price:
                up = float(pf.get("multiplierUp", 1)) - 1
                down = 1 - float(pf.get("multiplierDown", 1))
                pct = min(up, down, MAX_PERCENT_DIFF)
            else:
                pct = MAX_PERCENT_DIFF
            if b_price:
                diff = b_price * pct
                target = max(min(target, b_price + diff), b_price - diff)
            return float(exchange.price_to_precision(symbol, target))
        except Exception:
            return float(exchange.price_to_precision(symbol, target)) if target is not None else None

    best_price = price if adj_price is not None else _best_price()
    try:
        norm_qty, skip_reason = _normalize_order_qty(
            exchange,
            symbol,
            qty,
            best_price,
            order_type=order_type,
            side=side,
        )
    except Exception as exc:
        logging.warning("order | %s | normalization failed: %s", symbol, exc)
        log(logging.WARNING, "order", symbol, f"normalization failed: {exc}")
        tag = "order_normalization_failed"
        if tag not in _order_status[symbol]:
            _order_status[symbol].append(tag)
        return None, "normalization_failed"
    if skip_reason is not None:
        msg = f"order skipped: {skip_reason} (requested={qty})"
        log(logging.INFO, "order", symbol, msg)
        tag = f"order_{skip_reason}"
        if tag not in _order_status[symbol]:
            _order_status[symbol].append(tag)
        return None, skip_reason
    qty = norm_qty

    if adj_price is not None:
        best = best_price
        adj_price = _apply_filters(best, adj_price if adj_price is not None else best)

    for attempt in range(2):
        try:
            price_arg = None if is_market_like else adj_price
            order = exchange.create_order(symbol, order_type, side, qty, price_arg, params)
            order_id = order.get("id") or order.get("orderId")
            status = str(order.get("status", "")).upper()
            if not order_id or status.lower() in {"rejected", "canceled"}:
                raise ValueError(f"order {status or 'invalid'}")
            avg_price = order.get("avgPrice") or order.get("average") or order.get("price") or adj_price
            filled = float(order.get("executedQty") or order.get("filled") or qty)
            original = float(order.get("origQty") or qty)
            # discard any previously stored failure messages after a successful order
            _order_status[symbol] = [
                m for m in _order_status[symbol] if not m.lower().startswith("order failed")
            ]
            if status == "FILLED" or filled == original:
                msg = f"✅ Order {side.upper()} {filled:g} @ {avg_price} ({status or 'FILLED'})"
                if msg not in _order_status[symbol]:
                    _order_status[symbol].append(msg)
                return order_id, None
            else:
                msg = f"⚠️ Order {status}: {filled:g}/{original:g} filled"
                if msg not in _order_status[symbol]:
                    _order_status[symbol].append(msg)
                return order_id, None
        except Exception as e:
            err_str = str(e)
            if "-4131" in err_str and not is_market_like:
                if attempt == 0:
                    log(logging.WARNING, "order", symbol, "Order failed due to percent price filter, retrying...")
                    _order_status[symbol].append("order_retry_limit_price_adjusted")
                    best = _best_price()
                    if best:
                        factor = 1.001 if side == "buy" else 0.999
                        adj_price = _apply_filters(best, best * factor)
                    continue
                if ALLOW_MARKET_FALLBACK:
                    log(logging.WARNING, "order", symbol, "Switched to market order")
                    _order_status[symbol].append("order_fallback_to_market")
                    _order_status[symbol].append("order_market_fallback")
                    _order_status[symbol].append("order_cancelled")
                    try:
                        order = exchange.create_order(symbol, "market", side, qty, None, params)
                        order_id = order.get("id") or order.get("orderId")
                        if order_id:
                            return order_id, None
                    except Exception as me:
                        err_str = str(me)
                    msg = "order_failed_percent_filter"
                    if msg not in _order_status[symbol]:
                        _order_status[symbol].append(msg)
                    return None, err_str
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
            if msg not in _order_status[symbol]:
                _order_status[symbol].append(msg)
            return None, err_str


def safe_set_leverage(exchange, symbol: str, leverage: int, attempts: int = 2) -> bool:
    """Set leverage with retry and structured logging."""

    from exchange_adapter import set_valid_leverage

    for attempt in range(attempts):
        L = set_valid_leverage(exchange, symbol, leverage)
        if L is not None:
            log(logging.INFO, "leverage", symbol, f"Leverage set to {L}x")
            return True
        if attempt < attempts - 1:
            time.sleep(2)
            continue
        record_error(symbol, "leverage not set")
        log(logging.ERROR, "leverage", symbol, f"Failed to set leverage {leverage}")
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
