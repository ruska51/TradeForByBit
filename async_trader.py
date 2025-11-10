import asyncio
import logging
from typing import List, Dict

import ccxt
import pandas as pd
from datetime import datetime, timezone

from async_utils import fetch_multi_ohlcv_async
from logging_utils import (
    log,
    log_once,
    record_pattern,
    safe_create_order,
    safe_fetch_balance,
    safe_set_leverage,
    SOFT_ORDER_ERRORS,
    setup_logging,
)
from metrics_utils import backtest_metrics
from pattern_detector import detect_pattern
from main import predict_signal, log_trade
from exchange_adapter import ExchangeAdapter
from main import ADAPTER, API_KEY, API_SECRET, SANDBOX_MODE  # type: ignore
from model_utils import load_global_bundle
import risk_management
from volume_utils import _safe_vol_ratio, VOL_WINDOW

# Selected strong patterns influencing decision making
BULLISH_PATTERNS = {
    "bull_flag",
    "ascending_triangle",
    "ascending_channel",
    "cup_and_handle",
    "inverse_head_and_shoulders",
    "double_bottom",
    "triple_bottom",
    "hammer",
    "dragonfly_doji",
}

BEARISH_PATTERNS = {
    "bear_flag",
    "descending_triangle",
    "descending_channel",
    "head_and_shoulders",
    "double_top",
    "triple_top",
    "hanging_man",
    "dark_cloud_cover",
}


_GLOBAL_MODEL, _GLOBAL_SCALER, _GLOBAL_FEATURES, _GLOBAL_CLASSES = load_global_bundle()


def _make_exchange():
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if API_KEY:
        params["apiKey"] = API_KEY
    if API_SECRET:
        params["secret"] = API_SECRET
    exchange = ccxt.bybit(params)
    if hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(SANDBOX_MODE)
    return exchange


async def fetch_ticker(symbol: str) -> Dict:
    exchange = _make_exchange()
    try:
        return await asyncio.to_thread(exchange.fetch_ticker, symbol)
    finally:
        if hasattr(exchange, "close"):
            await asyncio.to_thread(exchange.close)


async def run_trade(
    symbol: str,
    side: str,
    mode: str,
    data: Dict[str, pd.DataFrame],
    pair_state: Dict[str, risk_management.PairState],
    config: Dict,
    limiter: risk_management.DailyLossLimiter,
    cool: risk_management.CoolDownManager,
    stats: risk_management.StatsTracker,
    start_index: int,
    risk_factor: float = 1.0,
) -> None:
    """Execute a trade using the full risk management stack."""

    # Guard against accidental trades against the prevailing trend.
    if not risk_management.confirm_trend(data, side):
        log(logging.INFO, "trade", symbol, "trend not confirmed")
        return

    if not cool.can_trade(symbol, start_index):
        log(logging.INFO, "trade", symbol, "cooldown active")
        return

    exchange = _make_exchange()
    try:
        await asyncio.to_thread(exchange.load_markets)
        market = exchange.market(symbol)
        tick_size = 1 / 10 ** market["precision"]["price"]
        step_size = 1 / 10 ** market["precision"]["amount"]

        price = await asyncio.to_thread(lambda: exchange.fetch_ticker(symbol)["last"])
        atr = risk_management.calc_atr(data["5m"])
        mode_params = {
            "sl_mult": config.get("atr_mult", 2.0),
            "tp_mult": config.get("tp_mult", 4.0),
        }
        tp_price, sl_price, sl_pct = risk_management.calc_sl_tp(
            price,
            atr,
            mode_params,
            side.lower(),
            tick_size=tick_size,
        )
        min_tick = tick_size if tick_size and tick_size > 0 else 1e-6
        if side.upper() == "SHORT":
            if sl_price >= price * 1.9:
                log_once(
                    "warning",
                    f"trade | {symbol} | skip: short stop %.4f too far above entry %.4f"
                    % (sl_price, price),
                    window_sec=30.0,
                )
                return
            if tp_price <= min_tick:
                log_once(
                    "warning",
                    f"trade | {symbol} | skip: short take-profit %.4f below minimum %.6f"
                    % (tp_price, min_tick),
                    window_sec=30.0,
                )
                return

        balance = await asyncio.to_thread(safe_fetch_balance, exchange, {"type": "future"})
        equity = balance["total"].get("USDT", 0.0)
        if not limiter.can_trade(symbol, equity):
            log(logging.INFO, "trade", symbol, "daily loss limit reached")
            return

        pair = pair_state.get(symbol, risk_management.PairState())
        base_risk = config.get("max_trade_loss_pct", 0.01)
        risk_per_trade = base_risk * pair.volume_factor * risk_factor
        qty = risk_management.calc_position_size(
            equity,
            risk_per_trade,
            price,
            sl_pct,
            ADAPTER,
            symbol,
            step_size,
        )
        qty += getattr(pair, "pending_qty", 0.0)
        qty = float(exchange.amount_to_precision(symbol, qty))
        min_qty = float(
            market.get("limits", {}).get("amount", {}).get("min", 0.0) or 0.0
        )
        if qty < min_qty:
            pair.pending_qty = qty
            pair_state[symbol] = pair
            risk_management.save_risk_state(pair_state, limiter, cool, stats)
            log(logging.INFO, "trade", symbol, "qty below min, accumulating")
            return
        pair.pending_qty = 0.0
        pair_state[symbol] = pair
        sl_price = float(exchange.price_to_precision(symbol, sl_price))
        tp_price = float(exchange.price_to_precision(symbol, tp_price))
        if side.upper() == "SHORT" and tp_price <= min_tick:
            log_once(
                "warning",
                f"trade | {symbol} | skip: precision rounded TP %.4f below minimum %.6f"
                % (tp_price, min_tick),
                window_sec=30.0,
            )
            return

        side_ccxt = "buy" if side == "LONG" else "sell"
        order_id, err = await asyncio.to_thread(
            safe_create_order, exchange, symbol, "market", side_ccxt, qty
        )
        if err in SOFT_ORDER_ERRORS:
            log(logging.INFO, "order", symbol, f"skipped: {err}")
            return
        if order_id is None:
            log(logging.ERROR, "trade", symbol, f"order failed: {err}")
            return

        log(
            logging.INFO,
            "trade",
            symbol,
            f"{side} {qty:.4f} @ {price:.5f} | SL={sl_price:.5f} TP={tp_price:.5f}",
        )

        exit_side = "sell" if side == "LONG" else "buy"

        def _trigger_direction(*, is_tp: bool) -> int:
            if side.upper() == "LONG":
                return 1 if is_tp else 2
            return 2 if is_tp else 1

        async def _place_stop_order(stop_value: float) -> str | None:
            trigger = float(exchange.price_to_precision(symbol, stop_value))
            params = {
                "category": "linear",
                "triggerPrice": trigger,
                "triggerDirection": _trigger_direction(is_tp=False),
                "triggerBy": "LastPrice",
                "reduceOnly": True,
                "closeOnTrigger": True,
                "tpSlMode": "Full",
            }
            try:
                response = await asyncio.to_thread(
                    exchange.create_order,
                    symbol,
                    "STOP_MARKET",
                    exit_side,
                    qty,
                    None,
                    params,
                )
            except Exception as exc:
                log_once(
                    "warning",
                    f"order | {symbol} | stop setup failed: {exc}",
                    window_sec=10.0,
                )
                return None
            order_id = None
            if isinstance(response, dict):
                order_id = response.get("id") or response.get("orderId")
            log(logging.INFO, "order", symbol, f"stop set @ {trigger:.5f}")
            return order_id

        async def _place_take_profit_order(target_value: float) -> str | None:
            trigger = float(exchange.price_to_precision(symbol, target_value))
            params = {
                "category": "linear",
                "triggerPrice": trigger,
                "triggerDirection": _trigger_direction(is_tp=True),
                "triggerBy": "LastPrice",
                "reduceOnly": True,
                "closeOnTrigger": True,
                "tpSlMode": "Full",
            }
            try:
                response = await asyncio.to_thread(
                    exchange.create_order,
                    symbol,
                    "TAKE_PROFIT_MARKET",
                    exit_side,
                    qty,
                    None,
                    params,
                )
            except Exception as exc:
                log_once(
                    "warning",
                    f"order | {symbol} | target setup failed: {exc}",
                    window_sec=10.0,
                )
                return None
            order_id = None
            if isinstance(response, dict):
                order_id = response.get("id") or response.get("orderId")
            log(logging.INFO, "order", symbol, f"target set @ {trigger:.5f}")
            return order_id

        async def _replace_stop_order(stop_value: float, current_id: str | None) -> str | None:
            if current_id:
                try:
                    await asyncio.to_thread(
                        exchange.cancel_order,
                        current_id,
                        symbol,
                        {"category": "linear"},
                    )
                except Exception:
                    pass
            return await _place_stop_order(stop_value)

        sl_order_id = None
        if sl_price > 0:
            sl_order_id = await _place_stop_order(sl_price)
        if tp_price > 0:
            await _place_take_profit_order(tp_price)

        trail = risk_management.TrailingStop(
            side, price, sl_price, atr, tick_size
        )

        open_idx = start_index
        cur_idx = start_index
        stop_hit = False
        trail_logged = False

        while True:
            await asyncio.sleep(1)
            ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
            last = ticker["last"]
            prev_stop = trail.stop_price
            stop_price = trail.update(last)

            if trail.active and not trail_logged:
                log(logging.INFO, "trail", symbol, f"activated @ {stop_price:.5f}")
                trail_logged = True

            if trail.active:
                eps = tick_size if tick_size > 0 else 1e-6
                moved = False
                if side == "LONG":
                    moved = stop_price > (prev_stop + eps)
                else:
                    moved = stop_price < (prev_stop - eps)
                if moved:
                    log(logging.INFO, "trail", symbol, f"stop moved to {stop_price:.5f}")
                    if stop_price > 0:
                        sl_order_id = await _replace_stop_order(stop_price, sl_order_id)

            if side == "LONG":
                if last <= stop_price:
                    stop_hit = True
                    exit_reason = "STOP"
                    break
                if last >= tp_price:
                    exit_reason = "TARGET"
                    break
            else:
                if last >= stop_price:
                    stop_hit = True
                    exit_reason = "STOP"
                    break
                if last <= tp_price:
                    exit_reason = "TARGET"
                    break

            cur_idx += 1
            if risk_management.time_stop(open_idx, cur_idx):
                exit_reason = "TIME"
                break

        close_side = "sell" if side == "LONG" else "buy"
        exit_price = last
        _, err = await asyncio.to_thread(
            safe_create_order, exchange, symbol, "market", close_side, qty
        )
        if err in SOFT_ORDER_ERRORS:
            log(logging.INFO, "order", symbol, f"skipped: {err}")
        await asyncio.to_thread(ADAPTER.cancel_open_orders, symbol)

        commission = 0.0006
        gross = (
            (exit_price - price) * qty
            if side == "LONG"
            else (price - exit_price) * qty
        )
        fee = commission * (price + exit_price) * qty
        profit = gross - fee
        exit_type = {
            "TARGET": "TAKE_PROFIT",
            "STOP": "TRAILING_STOP",
            "TIME": "TIME",
        }.get(exit_reason, "MANUAL")
        log_trade(
            datetime.now(timezone.utc),
            symbol,
            side,
            price,
            exit_price,
            qty,
            profit,
            exit_type,
        )
        limiter.register(symbol, profit)
        stats.register(symbol, profit)
        if profit < 0:
            cool.register_loss(symbol, cur_idx)

        risk_management.save_risk_state(pair_state, limiter, cool, stats)

        if stats.trades(symbol) % config.get("lookback_trades", 10) == 0:
            stats_dict = risk_management.update_pair_stats()
            risk_management.adjust_state_by_stats(pair_state, stats_dict, config)
            risk_management.save_pair_report(stats_dict)
            risk_management.save_risk_state(pair_state, limiter, cool, stats)

        if stop_hit:
            trend_ok = risk_management.confirm_trend(
                data, "SHORT" if side == "LONG" else "LONG"
            )
            new_side = risk_management.maybe_invert_position(side, True, trend_ok)
            if new_side:
                await run_trade(
                    symbol,
                    new_side,
                    mode,
                    data,
                    pair_state,
                    config,
                    limiter,
                    cool,
                    stats,
                    cur_idx,
                    0.5,
                )
    finally:
        if hasattr(exchange, "close"):
            await asyncio.to_thread(exchange.close)


def determine_mode(df: pd.DataFrame) -> str:
    """Lightweight trade mode classification used by the async demo."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    vol = df["volume"].rolling(14).mean().iloc[-1]
    vol_ratio = _safe_vol_ratio(df["volume"], VOL_WINDOW) or 0.0
    atr_ratio = atr / close.iloc[-1] if close.iloc[-1] else 0.0

    score = 0.4 * (atr_ratio / 0.01) + 0.3 * vol_ratio
    score += 0.3 * (close.pct_change(fill_method=None).rolling(5).std().iloc[-1] / 0.01)

    if score > 1.5:
        return "scalp"
    if score > 0.8:
        return "intraday"
    return "swing"


async def process_symbol(
    symbol: str,
    pair_state: Dict[str, risk_management.PairState],
    limiter: risk_management.DailyLossLimiter,
    cool: risk_management.CoolDownManager,
    stats: risk_management.StatsTracker,
    config: Dict,
) -> None:
    try:
        data_raw = await fetch_multi_ohlcv_async(
            symbol, ["5m", "15m", "30m", "1h", "4h", "1d"], limit=200
        )
        if not data_raw:
            log(logging.ERROR, "data", symbol, "no data")
            return
        if not data_raw.get("5m") and data_raw.get("15m"):
            data_raw["5m"] = data_raw.get("15m")
        if not data_raw.get("15m") and data_raw.get("5m"):
            data_raw["15m"] = data_raw.get("5m")
        if not data_raw.get("30m") and data_raw.get("15m"):
            data_raw["30m"] = data_raw.get("15m")

        data = {
            tf: pd.DataFrame(v, columns=["ts", "open", "high", "low", "close", "volume"])
            for tf, v in data_raw.items()
        }

        df_1h = data.get("1h", pd.DataFrame())
        if df_1h.empty or len(df_1h) < 2:
            log(logging.ERROR, "scan", symbol, "insufficient data")
            return

        roi = df_1h["close"].iloc[-1] / df_1h["close"].iloc[0] - 1
        metrics = backtest_metrics(df_1h["close"])
        sharpe = metrics["sharpe"]
        drawdown = metrics["max_drawdown"]
        log(
            logging.INFO,
            "scan",
            symbol,
            f"ROI={roi:.2%}, Sharpe={sharpe:.2f}, Drawdown={drawdown:.2%}",
        )
        if roi <= 0.005 or sharpe <= 0.3 or drawdown <= -0.05:
            return

        # --- multi timeframe signal detection ---------------------------------
        async def _predict(tf: str):
            pinfo = await detect_pattern(symbol, data.get(tf, pd.DataFrame()))
            record_pattern(symbol, pinfo["pattern_name"])
            sig, conf = await asyncio.to_thread(
                predict_signal,
                symbol,
                proba_filter=0.5,
                pattern_name=pinfo["pattern_name"],
                pattern_source=pinfo["source"],
                pattern_confidence=pinfo["confidence"],
            )
            side = None
            threshold = 0.6
            if sig == "long":
                if pinfo["pattern_name"] in BULLISH_PATTERNS:
                    threshold = 0.5
                elif pinfo["pattern_name"] in BEARISH_PATTERNS:
                    threshold = 0.7
                if conf >= threshold:
                    side = "LONG"
            elif sig == "short":
                if pinfo["pattern_name"] in BEARISH_PATTERNS:
                    threshold = 0.5
                elif pinfo["pattern_name"] in BULLISH_PATTERNS:
                    threshold = 0.7
                if conf >= threshold:
                    side = "SHORT"
            return side, pinfo

        side_15, pinfo_15 = await _predict("15m")
        side_30, pinfo_30 = await _predict("30m")
        if not side_15 or not side_30 or side_15 != side_30:
            return
        side = side_15
        pattern_info = (
            pinfo_15 if pinfo_15["confidence"] >= pinfo_30["confidence"] else pinfo_30
        )

        df_5m = data.get("5m", pd.DataFrame())
        mode = determine_mode(df_5m)
        if not risk_management.confirm_trend(data, side):
            return
        current_index = len(df_5m)
        lev = 10 if mode == "scalp" else 20
        pair = pair_state.get(symbol, risk_management.PairState())
        pair_state[symbol] = pair
        if not pair.leverage_ready:
            ex = _make_exchange()
            try:
                await asyncio.to_thread(ex.load_markets)
                success = await asyncio.to_thread(safe_set_leverage, ex, symbol, lev)
            finally:
                if hasattr(ex, "close"):
                    await asyncio.to_thread(ex.close)
            if not success:
                log(logging.INFO, "leverage", symbol, "skip leverage change, using cross")
            pair.leverage_ready = True
            pair_state[symbol] = pair
        tp_mult = config.get("tp_mult", 2.0)
        if side == "LONG" and pattern_info["pattern_name"] == "bull_flag":
            tp_mult = 3.0
        elif side == "SHORT" and pattern_info["pattern_name"] == "head_and_shoulders":
            tp_mult = 3.0
        local_cfg = dict(config, tp_mult=tp_mult)
        await run_trade(
            symbol,
            side,
            mode,
            data,
            pair_state,
            local_cfg,
            limiter,
            cool,
            stats,
            current_index,
        )
    except Exception as e:
        log(logging.ERROR, "process", symbol, str(e))


async def main() -> None:
    setup_logging()
    config = risk_management.load_config()
    pair_state, limiter, cool, stats = risk_management.load_risk_state(config)

    symbols = [
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "SUI/USDT",
        "TON/USDT",
    ]
    await asyncio.gather(
        *[
            process_symbol(s, pair_state, limiter, cool, stats, config)
            for s in symbols
        ]
    )
    stats_dict = risk_management.update_pair_stats()
    risk_management.adjust_state_by_stats(pair_state, stats_dict, config)
    risk_management.save_pair_report(stats_dict)
    risk_management.save_risk_state(pair_state, limiter, cool, stats)


if __name__ == "__main__":
    asyncio.run(main())
