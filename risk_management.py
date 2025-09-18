"""Utility helpers for risk and trade management.

This module originally only contained a few functions that adjusted the
position volume based on past performance. For the trading experiments in
this kata we gradually extend it with more robust risk controls such as ATR
based stops, trailing stops and daily loss limits. None of the functions here
depend on network access which keeps unit tests fast and deterministic.
"""

import json
import os
import logging
import math
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, Optional

import pandas as pd

from exchange_adapter import ExchangeAdapter

MIN_NOTIONAL = getattr(sys.modules.get("main"), "MIN_NOTIONAL", 10.0)


def _default_time_stop_bars() -> int:
    try:
        value = int(os.getenv("TIME_STOP_BARS", "60"))
    except ValueError:
        value = 60
    return max(1, value)


# Maximum number of bars a position may stay open before a forced exit is
# triggered.  The default can be overridden with the ``TIME_STOP_BARS``
# environment variable to allow deployment-specific tuning.
time_stop_bars: int = _default_time_stop_bars()

# [ANCHOR:TRAIL_PARAMS]
TRAIL_ACTIVATE_R: float = 0.5
TRAIL_ACTIVATE_ATR: float = 0.75
TRAIL_OFFSET_ATR: float = 1.2
TRAIL_MIN_TICKS: int = 3
USE_BREAKEVEN_STEP: bool = True
BREAKEVEN_BUFFER_ATR: float = 0.2


CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'risk_config.json')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'risk_state.json')


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            data = json.load(f)
    else:
        data = {
            "max_trade_loss_pct": 0.1,
            "min_winrate_for_increase": 0.6,
            "increase_step": 0.2,
            "max_increase_factor": 2.0,
            "decrease_step": 0.2,
            "lookback_trades": 10,
            "losing_streak_limit": 4,
            "max_daily_loss_per_symbol": 0.05,
            "cool_down_bars": 5,
            "max_open_trades": 5,
        }

    data.setdefault("trades_path", "trades_log.csv")
    data.setdefault("profit_report_path", "profit_report.csv")
    data.setdefault("equity_curve_path", "equity_curve.csv")
    data.setdefault("max_open_trades", 5)
    data.setdefault("reserve_symbols", [])
    return data


@dataclass
class PairState:
    volume_factor: float = 1.0
    losing_streak: int = 0
    pending_qty: float = 0.0


def load_state() -> Dict[str, PairState]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {
            k: PairState(
                volume_factor=v.get("volume_factor", 1.0),
                losing_streak=v.get("losing_streak", 0),
                pending_qty=v.get("pending_qty", 0.0),
            )
            for k, v in data.items()
        }
    return {}


def save_state(state: Dict[str, PairState]) -> None:
    data = {k: asdict(v) for k, v in state.items()}
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def update_pair_stats(log_path: str = 'trades_log.csv', lookback: int = 10) -> dict:
    if not os.path.exists(log_path):
        return {}
    df = pd.read_csv(log_path)
    stats = {}
    for symbol, grp in df.groupby('symbol'):
        last = grp.tail(lookback)
        winrate = (last['profit'] > 0).mean() if not last.empty else 0.0
        avg_profit = last['profit'].mean() if not last.empty else 0.0
        streak = 0
        for p in reversed(last['profit'].tolist()):
            if p <= 0:
                streak += 1
            else:
                break
        stats[symbol] = {
            'winrate': float(winrate),
            'avg_profit': float(avg_profit),
            'losing_streak': streak,
        }
    return stats


def adjust_state_by_stats(state: Dict[str, PairState], stats: dict, config: dict) -> Dict[str, PairState]:
    for symbol, s in stats.items():
        st = state.get(symbol, PairState())
        if s['winrate'] >= config.get('min_winrate_for_increase', 0.6) and st.volume_factor < config.get('max_increase_factor', 2.0):
            st.volume_factor = min(
                config.get('max_increase_factor', 2.0),
                st.volume_factor + config.get('increase_step', 0.2),
            )
        elif s['winrate'] < 0.5:
            st.volume_factor = max(
                0.1,
                st.volume_factor - config.get('decrease_step', 0.2),
            )
        st.losing_streak = s['losing_streak']
        state[symbol] = st
    return state

def save_pair_report(stats: dict, path: str = "pair_report.csv") -> None:
    df = pd.DataFrame(stats).T
    df["timestamp"] = datetime.now(timezone.utc)
    write_header = not os.path.exists(path)
    df.reset_index().rename(columns={"index": "symbol"}).to_csv(path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# ATR based stop/target calculation
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the Average True Range for *df*.

    The dataframe must contain ``high``, ``low`` and ``close`` columns.  A
    simple moving average is used which keeps the implementation small and
    dependency free.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"].shift()
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def calc_sl_tp(
    entry_price: float,
    side: str,
    atr_value: float,
    atr_mult: float = 2.0,
    tp_mult: float = 4.0,
    tick_size: float | None = None,
) -> tuple[float, float, float]:
    """Return take-profit price, stop-loss price and stop percentage.

    ``side`` is either ``"LONG"`` or ``"SHORT"``.  ``atr_mult`` controls the
    stop distance while ``tp_mult`` sets the multiple for the take-profit
    distance.  ``tick_size`` is optional rounding precision for price values.
    The function returns ``(tp_price, sl_price, sl_pct)`` which can be used
    directly for order placement and position sizing.
    """

    if entry_price <= 0:
        return entry_price, entry_price, 0.0

    atr_pct = atr_value / entry_price if entry_price else 0.0
    sl_mult_adj = atr_mult
    tp_mult_adj = tp_mult
    if atr_pct:
        if atr_pct < 0.0025:
            sl_mult_adj *= 3.0
            tp_mult_adj *= 3.0
        elif atr_pct < 0.005:
            sl_mult_adj *= 2.0
            tp_mult_adj *= 2.0

    raw_sl_pct = (sl_mult_adj * atr_value) / entry_price if entry_price else 0.0
    raw_tp_pct = (tp_mult_adj * atr_value) / entry_price if entry_price else 0.0

    sl_pct = max(raw_sl_pct, 0.005)
    base_tp_pct = max(raw_tp_pct, 0.0)
    scale = tp_mult_adj / sl_mult_adj if sl_mult_adj else tp_mult_adj
    tp_pct = max(sl_pct * max(scale, 1.0), 0.01, base_tp_pct)

    side_up = side.upper()
    if side_up == "LONG":
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    if tick_size:
        sl_price = round(tick_size * round(sl_price / tick_size), 8)
        tp_price = round(tick_size * round(tp_price / tick_size), 8)

    return tp_price, sl_price, sl_pct


def compute_order_qty(
    adapter: ExchangeAdapter,
    symbol: str,
    side: str,
    balance: float,
    risk_pct: float,
    price: float | None = None,
) -> float | None:
    """Return quantity respecting exchange precision, ``min_qty`` and notional limits."""

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

    try:
        resolved_price = float(price)
        if not math.isfinite(resolved_price) or resolved_price <= 0:
            raise ValueError
    except Exception:
        resolved_price = None

    exchange = getattr(adapter, "exchange", None)
    if exchange is None:
        exchange = getattr(adapter, "x", None)
    if exchange is None:
        logging.warning("risk | %s | exchange unavailable for sizing", symbol)
        return None

    if not resolved_price or resolved_price <= 0:
        fetchers: list[Callable[[str], dict | None]] = []
        if hasattr(exchange, "fetch_ticker"):
            fetchers.append(exchange.fetch_ticker)
        if hasattr(adapter, "fetch_ticker"):
            fetchers.append(adapter.fetch_ticker)
        side_hint = (side or "").lower()
        if side_hint == "buy":
            preferred_keys = ("ask", "last", "close", "bid")
        elif side_hint == "sell":
            preferred_keys = ("bid", "last", "close", "ask")
        else:
            preferred_keys = ("last", "close", "ask", "bid")
        seen: set[Callable[[str], dict | None]] = set()
        for fetcher in fetchers:
            if fetcher in seen:
                continue
            seen.add(fetcher)
            try:
                ticker = fetcher(symbol)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning("risk | %s | fetch_ticker failed: %s", symbol, exc)
                continue
            if not isinstance(ticker, dict):
                continue
            for key in preferred_keys:
                candidate = _safe_float(ticker.get(key), 0.0)
                if candidate > 0:
                    resolved_price = candidate
                    break
            if (not resolved_price or resolved_price <= 0) and isinstance(ticker.get("info"), dict):
                info_block = ticker["info"]
                for key in ("lastPrice", "close", "markPrice", "price"):
                    candidate = _safe_float(info_block.get(key), 0.0)
                    if candidate > 0:
                        resolved_price = candidate
                        break
            if resolved_price and resolved_price > 0:
                break

    if not resolved_price or resolved_price <= 0:
        logging.warning("risk | %s | unable to determine valid price for sizing", symbol)
        return None

    try:
        market = exchange.market(symbol) or {}
    except Exception as exc:
        logging.warning("risk | %s | market lookup failed: %s", symbol, exc)
        market = {}

    precision = (market.get("precision") or {}) if isinstance(market, dict) else {}
    limits = (market.get("limits") or {}) if isinstance(market, dict) else {}
    amount_limits = (limits.get("amount") or {}) if isinstance(limits, dict) else {}
    cost_limits = (limits.get("cost") or {}) if isinstance(limits, dict) else {}

    amount_precision = _safe_int((precision or {}).get("amount"))
    amount_min = _safe_float(amount_limits.get("min"), 0.0)
    amount_step = _safe_float(amount_limits.get("step"), 0.0)
    if amount_step <= 0:
        amount_step = _safe_float(amount_limits.get("stepSize"), amount_step)

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

    cost_min = _safe_float(cost_limits.get("min"), 0.0)
    notional_min = cost_min
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
    fallback_notional = 10.0 if symbol_upper.endswith("USDT") else float(MIN_NOTIONAL or 10.0)
    required_notional = max(cost_min, float(MIN_NOTIONAL))
    notional_min = max(notional_min, required_notional, fallback_notional)

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
        bal_value = float(balance)
    except (TypeError, ValueError):
        bal_value = 0.0
    if not math.isfinite(bal_value):
        bal_value = 0.0

    try:
        risk_value = float(risk_pct)
    except (TypeError, ValueError):
        risk_value = 0.0
    if not math.isfinite(risk_value):
        risk_value = 0.0
    risk_value = max(risk_value, 0.0)

    resolved_price = max(float(resolved_price), 1e-9)

    base_qty = max((bal_value * risk_value) / resolved_price, 0.0)
    qty_value = max(base_qty, amount_min)
    adjustments: list[str] = []

    if base_qty < amount_min:
        adjustments.append(
            f"raised_to_min_amount(prev={base_qty}, new={qty_value}, min={amount_min})"
        )

    qty_value = _apply_precision(qty_value)

    need_qty = notional_min / resolved_price
    need_qty = max(need_qty, 0.0)
    need_qty = _ceil_step(max(need_qty, amount_min))

    if qty_value * resolved_price < notional_min:
        prev_qty = qty_value
        qty_value = _apply_precision(max(qty_value, need_qty, amount_min))
        adjustments.append(
            f"raised_to_min_notional(prev={prev_qty}, new={qty_value}, price={resolved_price}, minNotional={notional_min})"
        )

    if qty_value <= 0:
        prev_qty = qty_value
        qty_value = _apply_precision(max(amount_min, amount_step))
        adjustments.append(
            f"raised_to_min_amount(prev={prev_qty}, new={qty_value}, min={amount_min})"
        )

    if qty_value * resolved_price < notional_min:
        prev_qty = qty_value
        qty_value = _apply_precision(max(need_qty, amount_min))
        adjustments.append(
            f"raised_to_min_notional(prev={prev_qty}, new={qty_value}, price={resolved_price}, minNotional={notional_min})"
        )

    notional = qty_value * resolved_price
    if qty_value <= 0 or qty_value < amount_min or notional < notional_min:
        logging.info(
            "risk | %s | qty too small (qty=%.8f, min=%.8f, notional=%.4f, min_notional=%.4f)",
            symbol,
            qty_value,
            amount_min,
            notional,
            notional_min,
        )
        return None

    side_label = (side or "").upper() or "BUY"
    logging.debug(
        "risk | %s | compute_order_qty side=%s balance=%.4f risk_pct=%.4f price=%.4f qty=%.8f "
        "notional=%.4f min_qty=%.8f min_notional=%.4f step=%.8f adjustments=%s",
        symbol,
        side_label,
        bal_value,
        risk_value,
        resolved_price,
        qty_value,
        notional,
        amount_min,
        notional_min,
        amount_step,
        "; ".join(adjustments) if adjustments else "none",
    )

    return float(qty_value)

def calc_position_size(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    sl_pct: float,
    adapter: ExchangeAdapter,
    symbol: str,
    step_size: float | None = None,
) -> float:
    """Return quantity so that only ``risk_per_trade`` of equity is at risk.

    If ``step_size`` is provided the quantity is rounded to the exchange's
    precision requirements.
    """

    if sl_pct <= 0:
        raise ValueError("sl_pct must be > 0")
    qty = (equity * risk_per_trade) / (entry_price * sl_pct)
    min_qty = 0.0
    try:
        market = adapter.exchange.market(symbol)
        min_qty = float(
            market.get("limits", {}).get("amount", {}).get("min", 0.0) or 0.0
        )
    except Exception:
        min_qty = 0.0
    if step_size:
        # round to exchange precision
        qty = step_size * round(qty / step_size)
        # ensure the minimum quantity also respects step size
        if min_qty:
            min_qty = step_size * math.ceil(min_qty / step_size)
        qty = max(qty, min_qty)
        qty = round(qty, 8)
    else:
        qty = max(qty, min_qty)
    return qty


# [ANCHOR:TRAIL_ACTIVATE_FN]
def _should_activate_trailing(side: str, entry: float, last: float, r_value: float, atr: float) -> bool:
    """Активация трейлинга: moved_R >= 0.5R ИЛИ moved_ATR >= 0.75ATR."""
    import math
    try:
        rv = float(r_value)
        if not math.isfinite(rv) or rv <= 0:
            rv = 1e-9
    except Exception:
        rv = 1e-9
    try:
        atrv = float(atr)
        if not math.isfinite(atrv) or atrv <= 0:
            atrv = 1e-9
    except Exception:
        atrv = 1e-9

    moved_r = abs(float(last) - float(entry)) / rv
    moved_atr = abs(float(last) - float(entry)) / atrv
    return (moved_r >= TRAIL_ACTIVATE_R) or (moved_atr >= TRAIL_ACTIVATE_ATR)


def should_activate_trailing(side: str, entry: float, last: float, r_value: float, atr: float) -> bool:
    return _should_activate_trailing(side, entry, last, r_value, atr)


# [ANCHOR:TRAIL_UPDATE_FN]


def _tick_digits(tick: float) -> int:
    s = f"{tick:.8f}".rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0


class TickValue(float):
    """Float wrapper that formats according to tick size."""

    def __new__(cls, value: float, tick: float):
        obj = float.__new__(cls, value)
        obj._tick = tick
        return obj

    def __format__(self, spec: str) -> str:
        if spec == "@TICK":
            digits = _tick_digits(getattr(self, "_tick", 0.0))
            return f"{float(self):.{digits}f}"
        return float.__format__(self, spec)


def exchange_price_to_precision(symbol: str, price: float) -> float:
    """Return ``price`` rounded using the exchange's precision if available."""

    try:
        import main  # type: ignore

        exchange = getattr(main, "exchange", None)
        if exchange and hasattr(exchange, "price_to_precision"):
            return float(exchange.price_to_precision(symbol, price))
    except Exception:
        pass
    return float(price)


def _quantize(symbol: str | None, value: float, tick: float) -> TickValue:
    v = exchange_price_to_precision(symbol or "", value)
    if tick > 0:
        v = round(round(v / tick) * tick, 8)
    return TickValue(v, tick)


def _trail_levels(
    side: str,
    entry: float,
    last: float,
    atr: float,
    tick: float,
    breakeven_done: bool,
    current_sl: float | None = None,
    symbol: str | None = None,
) -> tuple[TickValue, bool]:
    """
    Возвращает (new_sl, new_breakeven_done).
    1) Если BE ещё не сделан и USE_BREAKEVEN_STEP=True → переносим в BE и отмечаем флаг.
    2) Иначе ведём по ATR-отступу (>= 3 тика), не расширяя риск относительно entry.
    В результирующем значении учитывается точность биржи и сетка тика.
    Обновление, которое ухудшает риск относительно ``current_sl``, игнорируется.
    """
    import math

    # --- Guards ---
    try:
        atr_safe = float(atr)
        if not math.isfinite(atr_safe) or atr_safe <= 0:
            atr_safe = 1e-9
    except Exception:
        atr_safe = 1e-9

    try:
        tick = float(tick)
        if not math.isfinite(tick) or tick <= 0:
            tick = 1e-9
    except Exception:
        tick = 1e-9

    off = max(TRAIL_OFFSET_ATR * atr_safe, TRAIL_MIN_TICKS * tick)

    # --- Breakeven (однократно) ---
    if not breakeven_done and USE_BREAKEVEN_STEP:
        be = float(entry) + (
            BREAKEVEN_BUFFER_ATR * atr_safe if side == "LONG" else -BREAKEVEN_BUFFER_ATR * atr_safe
        )
        be = _quantize(symbol, be, tick)
        if current_sl is not None:
            if side == "LONG" and be <= current_sl:
                return _quantize(symbol, current_sl, tick), True
            if side == "SHORT" and be >= current_sl:
                return _quantize(symbol, current_sl, tick), True
        return be, True

    # --- Ведение по ATR ---
    if side == "LONG":
        # стоп ниже текущей цены, но не ниже entry (не расширяем риск)
        candidate = max(float(last) - off, float(entry))
        if current_sl is not None and candidate <= current_sl:
            return _quantize(symbol, current_sl, tick), breakeven_done
        new_sl = _quantize(symbol, candidate, tick)
        if current_sl is not None and new_sl <= current_sl:
            return _quantize(symbol, current_sl, tick), breakeven_done
        return new_sl, breakeven_done
    else:
        # SHORT: стоп выше текущей цены, но не выше entry (не расширяем риск)
        candidate = min(float(last) + off, float(entry))
        if current_sl is not None and candidate >= current_sl:
            return _quantize(symbol, current_sl, tick), breakeven_done
        new_sl = _quantize(symbol, candidate, tick)
        if current_sl is not None and new_sl >= current_sl:
            return _quantize(symbol, current_sl, tick), breakeven_done
        return new_sl, breakeven_done


def trail_levels(
    side: str,
    entry: float,
    last: float,
    atr: float,
    tick: float,
    breakeven_done: bool,
    current_sl: float | None = None,
    symbol: str | None = None,
):
    return _trail_levels(side, entry, last, atr, tick, breakeven_done, current_sl, symbol)


class TrailingStop:
    """ATR based trailing stop with optional breakeven step."""

    def __init__(
        self,
        side: str,
        entry_price: float,
        initial_stop: float,
        atr: float,
        tick_size: float | None = None,
    ) -> None:
        self.side = side.upper()
        self.entry_price = entry_price
        self.stop_price = initial_stop
        self.initial_stop = initial_stop
        self.atr = atr
        self.tick_size = tick_size or 0.0
        self.active = False
        self._breakeven_done = False
        self._r_value = abs(entry_price - initial_stop)

    def update(self, current_price: float) -> float:
        """Update trailing stop and return new stop price."""

        if not self.active:
            if _should_activate_trailing(
                self.side, self.entry_price, current_price, self._r_value, self.atr
            ):
                self.active = True
        if self.active:
            new_sl, self._breakeven_done = _trail_levels(
                self.side,
                self.entry_price,
                current_price,
                self.atr,
                self.tick_size,
                self._breakeven_done,
                self.stop_price,
                None,
            )
            if self.side == "LONG" and new_sl > self.stop_price:
                self.stop_price = float(new_sl)
            elif self.side == "SHORT" and new_sl < self.stop_price:
                self.stop_price = float(new_sl)

        return self.stop_price


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


@dataclass
class DailyLossLimiter:
    """Track daily loss per symbol and disable trading once the threshold is hit."""

    max_daily_loss_per_symbol: float
    losses: Dict[str, float] = field(default_factory=dict)
    day: datetime = field(default_factory=lambda: datetime.now(timezone.utc).date())

    def _maybe_reset(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.losses.clear()

    def register(self, symbol: str, profit: float) -> None:
        self._maybe_reset()
        self.losses[symbol] = self.losses.get(symbol, 0.0) + profit

    def can_trade(self, symbol: str, equity: float) -> bool:
        self._maybe_reset()
        max_loss = equity * self.max_daily_loss_per_symbol
        return self.losses.get(symbol, 0.0) > -max_loss
    

def time_stop(open_index: int, current_index: int, time_limit: int | None = None) -> bool:
    """Return ``True`` if a position has been open for ``time_limit`` bars."""

    if time_limit is None:
        time_limit = time_stop_bars
    return (current_index - open_index) >= time_limit


@dataclass
class CoolDownManager:
    """Enforce a waiting period after a losing trade.

    ``cool_down_bars`` specifies how many bars must pass after a losing trade
    before a new trade on the same symbol is allowed.
    """

    cool_down_bars: int
    last_loss: Dict[str, int] = field(default_factory=dict)

    def register_loss(self, symbol: str, bar_index: int) -> None:
        self.last_loss[symbol] = bar_index

    def can_trade(self, symbol: str, current_index: int) -> bool:
        last = self.last_loss.get(symbol)
        if last is None:
            return True
        return (current_index - last) >= self.cool_down_bars


# ---------------------------------------------------------------------------
# Trend confirmation helpers
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return float(dx.ewm(alpha=1 / period).mean().iloc[-1])


# [ANCHOR:TREND_SAFE_UTILS]
def _safe_series_tail_value(s):
    """Вернёт последний элемент Series или None, если серия пуста/вся из NaN."""
    try:
        if s is None:
            return None
        s = s.dropna()
        if len(s) == 0:
            return None
        return s.iloc[-1]
    except Exception:
        return None


def _safe_ema(series, period: int):
    """Безопасный EMA: вернёт pd.Series или None, если ряд слишком короткий/некорректный."""
    try:
        if series is None:
            return None
        series = series.dropna()
        # требуем хотя бы period наблюдений
        if len(series) < period:
            return None
        return series.ewm(span=period, adjust=False).mean()
    except Exception:
        return None


def _valid_close(df):
    """Вернёт pd.Series 'close' или None, если df пуст/нет колонки."""
    try:
        if df is None or getattr(df, "empty", True):
            return None
        if "close" not in df.columns:
            return None
        return df["close"]
    except Exception:
        return None


def confirm_trend(
    data: Dict[str, pd.DataFrame],
    side: str,
    symbol: str | None = None,
    *,
    return_reason: bool = False,
):
    """Validate trend alignment across multiple timeframes using majority vote.

    Returns either a boolean (default) or ``(bool, reason)`` when
    ``return_reason`` is set to ``True``.  ``reason`` is ``"insufficient_data"``
    when guard-rails detect empty or too short dataframes.
    """

# [ANCHOR:TREND_SAFE_BODY]
    side = (side or "").upper()
    votes: list[str] = []  # "BULL" / "BEAR"
    reason: str | None = None

    ema_fast_len = 9
    ema_slow_len = 21
    adx_len = 14
    atr_len = 14
    needed = max(ema_fast_len, ema_slow_len, adx_len, atr_len)

    fetcher = globals().get("cached_fetch_ohlcv") or globals().get("fetch_ohlcv")

    # dfs ожидается dict { "5m": df, "15m": df, ... }
    for tf, df in (data or {}).items():
        close = _valid_close(df)
        if close is None or len(close) < needed:
            if fetcher and symbol:
                try:
                    df = fetcher(symbol, tf, limit=needed)
                    close = _valid_close(df)
                except Exception:
                    close = None
            if close is None or len(close) < needed:
                reason = "insufficient_data"
                if symbol:
                    try:
                        from logging_utils import log_decision

                        log_decision(symbol, reason)
                    except Exception:
                        pass
                return (False, reason) if return_reason else False

        ema_fast = _safe_ema(close, ema_fast_len)
        ema_slow = _safe_ema(close, ema_slow_len)
        v_fast = _safe_series_tail_value(ema_fast)
        v_slow = _safe_series_tail_value(ema_slow)

        if v_fast is None or v_slow is None:
            # короткий ряд — пропускаем ТФ
            continue

        bull = v_fast > v_slow

        adx_val = None
        try:
            adx_val = _adx(df)
            if pd.isna(adx_val):
                adx_val = None
        except Exception:
            adx_val = None
        if adx_val is not None and adx_val <= 20:
            # низкий ADX — пропускаем ТФ
            continue

        votes.append("BULL" if bull else "BEAR")

    if not votes:
        reason = "weak_trend"
        if symbol:
            try:
                from logging_utils import log_decision

                log_decision(symbol, reason)
            except Exception:
                pass
        logging.warning("confirm_trend: no valid TF votes; treating as weak_trend")
        return (True, reason) if return_reason else True

    bulls = votes.count("BULL")
    bears = votes.count("BEAR")

    # большинство по голосам
    majority_bull = bulls > bears
    majority_bear = bears > bulls

    if side == "LONG":
        result = majority_bull
    elif side == "SHORT":
        result = majority_bear
    else:
        # если side не задан — достаточно наличия большинства в любую сторону
        result = majority_bull or majority_bear

    return (result, reason) if return_reason else result


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def round_order(price: float, qty: float, tick_size: float, step_size: float) -> Dict[str, float]:
    """Round ``price`` and ``qty`` according to exchange precision limits."""

    def _round(value: float, step: float) -> float:
        return round(step * round(value / step), 8)

    return {
        "price": _round(price, tick_size),
        "qty": _round(qty, step_size),
    }


def maybe_invert_position(side: str, stop_hit: bool, trend_ok: bool) -> Optional[str]:
    """Return opposite side for potential position inversion."""

    if stop_hit and trend_ok:
        side_u = side.upper()
        if side_u == "LONG":
            return "SHORT"
        if side_u == "SHORT":
            return "LONG"
    return None


@dataclass
class StatsTracker:
    """Collect per-symbol trade statistics and temporary ban state."""

    stats: Dict[str, Dict[str, float | int | str | list | bool | None]] = field(
        default_factory=dict
    )

    def on_trade_close(
        self, symbol: str, profit: float, timestamp: datetime | None = None
    ) -> None:
        now = timestamp or datetime.now(timezone.utc)
        s = self.stats.setdefault(
            symbol,
            {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "equity": 0.0,
                "peak": 0.0,
                "max_drawdown": 0.0,
                "loss_streak": 0,
                "loss_streak_start": None,
                "banned_until": None,
                "reduced_risk": False,
            },
        )
        s["trades"] += 1
        s["pnl"] += profit
        s["equity"] += profit
        if profit > 0:
            s["wins"] += 1
            s["losses"] = s.get("losses", 0)
            s["loss_streak"] = 0
            s["loss_streak_start"] = None
            s["banned_until"] = None
            s["reduced_risk"] = False
        else:
            s["losses"] = s.get("losses", 0) + 1
            start = s.get("loss_streak_start")
            if not start:
                s["loss_streak_start"] = now.isoformat()
                s["loss_streak"] = 1
            else:
                start_dt = datetime.fromisoformat(start)
                if now - start_dt > timedelta(minutes=60):
                    s["loss_streak_start"] = now.isoformat()
                    s["loss_streak"] = 1
                else:
                    s["loss_streak"] = s.get("loss_streak", 0) + 1
            if (
                s["loss_streak"] >= 2
                and now - datetime.fromisoformat(s["loss_streak_start"]) <= timedelta(minutes=60)
            ):
                # [ANCHOR:SYMBOL_BAN_RULE]
                s["banned_until"] = (now + timedelta(minutes=60)).isoformat()
        s["peak"] = max(s["peak"], s["equity"])
        dd = s["equity"] - s["peak"]
        s["max_drawdown"] = min(s["max_drawdown"], dd)

    def register(
        self, symbol: str, profit: float, timestamp: datetime | None = None
    ) -> None:
        """Backward compatible wrapper for ``on_trade_close``."""
        self.on_trade_close(symbol, profit, timestamp)

    def is_banned(self, symbol: str, now: datetime | None = None) -> bool:
        s = self.stats.get(symbol)
        if not s:
            return False
        until = s.get("banned_until")
        if not until:
            return False
        now = now or datetime.now(timezone.utc)
        if now < datetime.fromisoformat(until):
            return True
        s["banned_until"] = None
        s["loss_streak"] = 0
        s["loss_streak_start"] = None
        s["reduced_risk"] = True
        return False

    def clear_ban(self, symbol: str) -> None:
        s = self.stats.get(symbol)
        if s:
            s["banned_until"] = None
            s["loss_streak"] = 0
            s["loss_streak_start"] = None
            s["reduced_risk"] = True

    def pop_soft_risk(self, symbol: str) -> bool:
        """Return ``True`` while reduced risk mode is active for ``symbol``."""
        s = self.stats.get(symbol)
        return bool(s and s.get("reduced_risk"))

    # Convenience accessors -------------------------------------------------
    def trades(self, symbol: str) -> int:
        return int(self.stats.get(symbol, {}).get("trades", 0))

    def win_rate(self, symbol: str) -> Optional[float]:
        s = self.stats.get(symbol)
        if not s or not s.get("trades"):
            return None
        return s["wins"] / s["trades"]

    def avg_profit(self, symbol: str) -> Optional[float]:
        s = self.stats.get(symbol)
        if not s or not s.get("trades"):
            return None
        return s["pnl"] / s["trades"]

    def max_drawdown(self, symbol: str) -> Optional[float]:
        s = self.stats.get(symbol)
        if not s:
            return None
        return s["max_drawdown"]



def load_risk_state(config: Dict, path: str = STATE_FILE):
    """Load extended risk state including pair info and risk managers.

    ``config`` is used to derive default values for the risk managers.  The
    original :func:`load_state` only restored volume factors.  To make the
    trading loop fully restartable we also persist ``DailyLossLimiter``,
    ``CoolDownManager`` and ``StatsTracker``.  The returned tuple contains
    ``(pair_state, limiter, cool_down, stats)``.
    """

    max_daily_loss_per_symbol = config.get("max_daily_loss_per_symbol", 0.02)
    cool_down_bars = config.get("cool_down_bars", 5)

    pair_state: Dict[str, PairState] = {}
    limiter = DailyLossLimiter(max_daily_loss_per_symbol)
    cool = CoolDownManager(cool_down_bars)
    stats = StatsTracker()

    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

        pair_data = data.get("pairs", data)
        pair_state = {
            k: PairState(
                volume_factor=v.get("volume_factor", 1.0),
                losing_streak=v.get("losing_streak", 0),
                pending_qty=v.get("pending_qty", 0.0),
            )
            for k, v in pair_data.items()
        }

        if "daily" in data:
            day = datetime.fromisoformat(data["daily"].get("day")).date()
            limiter = DailyLossLimiter(
                data["daily"].get(
                    "max_daily_loss_per_symbol", max_daily_loss_per_symbol
                ),
                losses=data["daily"].get("losses", {}),
                day=day,
            )

        if "cool" in data:
            cool = CoolDownManager(
                data["cool"].get("cool_down_bars", cool_down_bars),
                last_loss=data["cool"].get("last_loss", {}),
            )

        if "stats" in data:
            stats = StatsTracker(stats=data.get("stats", {}))

    return pair_state, limiter, cool, stats


def save_risk_state(
    pair_state: Dict[str, PairState],
    limiter: DailyLossLimiter,
    cool: CoolDownManager,
    stats: StatsTracker,
    path: str = STATE_FILE,
) -> None:
    """Persist pair state together with risk manager information."""

    data = {
        "pairs": {k: asdict(v) for k, v in pair_state.items()},
        "daily": {
            "max_daily_loss_per_symbol": limiter.max_daily_loss_per_symbol,
            "losses": limiter.losses,
            "day": str(limiter.day),
        },
        "cool": {
            "cool_down_bars": cool.cool_down_bars,
            "last_loss": cool.last_loss,
        },
        "stats": stats.stats,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


