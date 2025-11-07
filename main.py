# === ПРОКАЧАННЫЙ ТОРГОВЫЙ БОТ ===

from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# make stdout/stderr line-buffered so console shows live logs even without -u
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # pragma: no cover - best effort
    pass

from logging_utils import setup_logging
setup_logging()
print("===== BOT START =====", flush=True)
import logging
logging.info("boot | cwd=%s | root=%s", os.getcwd(), ROOT)

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import asyncio
import math
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, asdict
import optuna
try:
    from colorama import Fore, Style, init
except Exception:  # pragma: no cover - optional dependency fallback
    class _NoColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = _NoColor()  # type: ignore

    def init(*args, **kwargs):  # type: ignore
        return None
import json
import joblib
import matplotlib
import csv

init(autoreset=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
from model_utils import (
    load_global_bundle,
    BUNDLE_PATH,
    SimpleScaler,
)
from sklearn.metrics import f1_score
from collections import Counter, defaultdict
from uuid import uuid4
from utils.data_prep import fetch_and_prepare_training_data

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
except Exception:  # pragma: no cover - optional dependency
    import torch_stub as torch
    import torch_stub.nn as nn
    from torchvision_stub import transforms
    from torchvision_stub.datasets import ImageFolder
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore
import requests
import zipfile
import io
import shutil

try:
    import mlcroissant as mlc
except Exception:  # pragma: no cover - optional dependency
    mlc = None
from asset_scanner import scan_symbols

# === Trade analysis utilities ===
from pathlib import Path
import logging_utils
from logging_utils import (
    colorize,
    log,
    log_decision,
    record_candle_status,
    record_no_data,
    clear_no_data,
    record_backtest,
    record_pattern,
    record_error,
    safe_create_order,
    place_conditional_exit,
    enter_ensure_filled,
    wait_position_after_entry,
    has_pending_entry,
    has_open_position,
    get_position_entry_price,
    get_last_price,
    SOFT_ORDER_ERRORS,
    flush_symbol_logs,
    flush_cycle_logs,
    log_prediction_error,
    record_summary,
    emit_summary,
    log_once,
    _is_bybit_exchange,
    detect_market_category,
    _normalize_bybit_symbol,
    _price_qty_to_precision,
    _round_qty,
)
from retrain_utils import retrain_global_model
from fallback import fallback_signal
from metrics_utils import backtest_metrics
import risk_management
from risk_management import (
    load_config,
    update_pair_stats,
    adjust_state_by_stats,
    save_pair_report,
    PairState,
    time_stop,
    time_stop_bars,
    load_risk_state,
    save_risk_state,
    StatsTracker,
    confirm_trend,
)
from memory_utils import memory_manager, normalize_param_keys
from pattern_detector import (
    REAL_PATTERNS,
    detect_pattern,
    detect_pattern_image,
    BULLISH_PATTERNS as DETECTOR_BULLISH_PATTERNS,
    BEARISH_PATTERNS as DETECTOR_BEARISH_PATTERNS,
    self_test as pattern_self_test,
)

# Bybit requires explicit trigger direction values when placing conditional
# orders.  The exchange expects the ``triggerDirection`` argument to be the
# string ``"ascending"`` when the trigger fires as the price moves up, and
# ``"descending"`` when it fires while the price moves down.  Keeping the
# mapping in one place prevents drift between the trading helpers and the
# official API semantics.
BYBIT_TRIGGER_DIRECTIONS = {
    "rising": "ascending",
    "falling": "descending",
}

# Классификация паттернов по направлению
BULLISH_PATTERNS = {
    "bull_flag",
    "double_bottom",
    "triple_bottom",
    "ascending_triangle",
    "cup_and_handle",
    "inverse_head_and_shoulders",
}
BULLISH_PATTERNS |= set(DETECTOR_BULLISH_PATTERNS)

BEARISH_PATTERNS = {
    "bear_flag",
    "double_top",
    "triple_top",
    "descending_triangle",
    "head_and_shoulders",
}
BEARISH_PATTERNS |= set(DETECTOR_BEARISH_PATTERNS)

NEUTRAL_PATTERNS = {"rectangle", "sym_triangle"}


def get_pattern_direction(pattern_name: str) -> str | None:
    """Возвращает направление ('long'/'short') для распознанного паттерна."""

    name = (pattern_name or "").lower()
    if name in BULLISH_PATTERNS:
        return "long"
    if name in BEARISH_PATTERNS:
        return "short"
    return None


def _retrain_checked(*args, **kwargs):
    """Wrapper around :func:`retrain_global_model` ensuring bundle persistence."""

    model, scaler, feats, classes = retrain_global_model(*args, **kwargs)
    if not BUNDLE_PATH.exists():
        logging.error(
            "main | retrain completed but bundle missing at %s", BUNDLE_PATH
        )
        raise RuntimeError("retrain did not produce model bundle")
    return model, scaler, feats, classes
import volume_utils
from volume_utils import VOL_WINDOW, VOL_RATIO_MAX, safe_vol_ratio, volume_reason, safe_atr

# --- CSV helpers -------------------------------------------------------


def touch_csv(path: str, headers: list[str]) -> None:
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    """Append ``row`` to ``path`` ensuring ``fieldnames`` header exists."""

    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})

# [ANCHOR:IMPORTS_INTEGRATION]
from exchange_adapter import (
    ExchangeAdapter,
    AdapterInitError,
    AdapterOHLCVUnavailable,
    safe_fetch_closed_orders,
)
from typing import Dict, Tuple
from logging_utils import (
    ensure_trades_csv_header,
    ensure_report_schema,
    TRADES_CSV_HEADER,
    log_entry,  # не менять сигнатуру, возвращает trade_id
    log_exit_from_order,
    safe_fetch_balance,
    safe_set_leverage,
    detect_market_category,
)
from reporting import build_profit_report, build_equity_curve
from risk_management import (
    trail_levels,
    should_activate_trailing,
)
from symbol_utils import filter_supported_symbols

STRONG_BULL_PATTERNS = {"bull_flag", "cup_and_handle", "triple_bottom"}
STRONG_BEAR_PATTERNS = {"bear_flag", "triple_top"}

# [ANCHOR:INTEGRATION_SWITCH]
ENABLE_ATR_TRAIL = True
ENABLE_CLOSE_ORDERING = True
ENABLE_REPORTS_BUILDER = True
ENABLE_SYMBOL_BAN = True
ENABLE_VOL_RATIO_FIX = True

# [ANCHOR:STATE_INIT]
touch_csv("trades_log.csv", TRADES_CSV_HEADER)
touch_csv(
    "profit_report.csv",
    [
        "timestamp",
        "symbol",
        "pnl_net",
        "cum_pnl",
        "winrate",
        "avg_win",
        "avg_loss",
        "sharpe",
        "max_dd",
    ],
)
touch_csv("equity_curve.csv", ["timestamp", "equity"])
touch_csv("decision_log.csv", ["timestamp", "symbol", "signal", "reason"])

# лог-пути
trade_log_path = str(ROOT / "trades_log.csv")
profit_report_path = str(ROOT / "profit_report.csv")
equity_curve_path = str(ROOT / "equity_curve.csv")
pair_report_path = str(ROOT / "pair_report.csv")
ensure_trades_csv_header(trade_log_path)
ensure_report_schema(
    profit_report_path,
    [
        "timestamp",
        "symbol",
        "pnl_net",
        "cum_pnl",
        "winrate",
        "avg_win",
        "avg_loss",
        "sharpe",
        "max_dd",
    ],
)
ensure_report_schema(equity_curve_path, ["timestamp", "equity"])
ensure_report_schema(
    pair_report_path,
    ["symbol", "winrate", "avg_profit", "losing_streak", "timestamp"],
)

# кэш и сервисные структуры
if "ohlcv_cache" not in globals():
    ohlcv_cache = {}  # {(symbol, tf): (ts, df)}
if "_processed_order_ids" not in globals():
    _processed_order_ids = set()
if "pair_state" not in globals():
    pair_state = {}
# [ANCHOR:STATE_INIT_MARKETS_CACHE]
if "markets_cache" not in globals():
    markets_cache = {"loaded": False, "ts": 0.0, "by_name": set(), "by_id": set()}


def load_trades(csv_file: str = "trades_log.csv", create: bool = False) -> pd.DataFrame:
    """Load a CSV file with trade logs, optionally creating it if missing."""
    path = Path(csv_file).expanduser()
    if not path.exists():
        alt_path = Path(__file__).resolve().parent / csv_file
        if alt_path.exists():
            path = alt_path
        elif create:
            path = alt_path
            cols = [
                "timestamp",
                "symbol",
                "side",
                "entry_price",
                "exit_price",
                "volume",
                "profit",
                "exit_type",
            ]
            pd.DataFrame(columns=cols).to_csv(path, index=False)
        else:
            raise FileNotFoundError(csv_file)
    return pd.read_csv(path)


def _coerce_exit_type(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    if "exit_type" in coerced.columns:
        coerced["exit_type"] = coerced["exit_type"].astype(str)
    return coerced


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic metrics from trade data."""
    df = _coerce_exit_type(df)
    if "exit_type" in df.columns:
        df["exit_type"] = df["exit_type"].astype(str)
        df = df[df["exit_type"].str.upper() != "ENTRY"]
    sl_rate = 0.0
    tp_rate = 0.0
    if "sl_hit" in df.columns:
        sl_rate = df["sl_hit"].mean()
    elif "exit_type" in df.columns:
        sl_rate = df["exit_type"].str.contains("STOP", case=False).mean()

    if "tp_hit" in df.columns:
        tp_rate = df["tp_hit"].mean()
    elif "exit_type" in df.columns:
        tp_rate = df["exit_type"].str.contains("TAKE_PROFIT", case=False).mean()

    if "entry" in df.columns:
        avg_roi = (df["profit"] / df["entry"]).mean()
    elif "entry_price" in df.columns:
        avg_roi = (df["profit"] / df["entry_price"]).mean()
    else:
        avg_roi = 0.0

    avg_duration = df["duration"].mean() if "duration" in df.columns else 0.0

    metrics = {
        "average_pnl": df["profit"].mean(),
        "total_trades": len(df),
        "winrate": (df["profit"] > 0).mean(),
        "sl_rate": sl_rate,
        "tp_rate": tp_rate,
        "average_roi": avg_roi,
        "average_duration": avg_duration,
    }
    return metrics


def analyze_errors(df: pd.DataFrame) -> List[str]:
    """Identify common mistakes based on statistics."""
    df = _coerce_exit_type(df)
    issues = []
    if "exit_type" in df.columns:
        df = df[df["exit_type"].str.upper() != "ENTRY"]
    if "sl_hit" in df.columns:
        sl_rate = df["sl_hit"].mean()
    elif "exit_type" in df.columns:
        sl_rate = df["exit_type"].str.contains("STOP", case=False).mean()
    else:
        sl_rate = 0.0

    if sl_rate > 0.5:
        issues.append("High stop loss hit rate detected")
    if (df["profit"] <= 0).mean() > 0.5:
        issues.append("More than half of trades are losing")
    if "duration" in df.columns and df["duration"].mean() > df["duration"].median() * 2:
        issues.append("Some trades take unusually long")
    return issues


def recommend_parameters(df: pd.DataFrame) -> Dict[str, float]:
    """Provide naive parameter recommendations from historical data."""
    df = _coerce_exit_type(df)
    if "exit_type" in df.columns:
        df = df[df["exit_type"].str.upper() != "ENTRY"]
    entry_col = "entry" if "entry" in df.columns else "entry_price"
    sl_pct = df.loc[df["profit"] < 0, "profit"].abs().mean() / df[entry_col].mean()
    tp_pct = df.loc[df["profit"] > 0, "profit"].mean() / df[entry_col].mean()
    threshold = (df["profit"].mean() / df[entry_col].mean()) / 2
    horizon = int(df["duration"].median()) if "duration" in df.columns else 0
    proba_filter = max(0.4, min(0.8, (df["profit"] > 0).mean()))
    return {
        "SL_PCT": round(sl_pct, 4),
        "TP_PCT": round(tp_pct, 4),
        "THRESHOLD": round(threshold, 4),
        "HORIZON": horizon,
        "PROBA_FILTER": round(proba_filter, 2),
        "ADX_THRESHOLD": ADX_THRESHOLD,
        "RSI_OVERBOUGHT": RSI_OVERBOUGHT,
        "RSI_OVERSOLD": RSI_OVERSOLD,
    }


def default_backtest(
    df: pd.DataFrame,
    sl_pct: float,
    tp_pct: float,
    threshold: float,
    horizon: int,
    proba_filter: float,
    adx_threshold: float,
    rsi_overbought: float,
    rsi_oversold: float,
) -> float:
    if sl_pct + tp_pct == 0:
        return 0.0
    pnl_scale = tp_pct / (sl_pct + tp_pct)
    bonus = (proba_filter - 0.4) + (30 - adx_threshold) * 0.01
    return float(df["profit"].sum() * pnl_scale + bonus)


def run_optuna(
    df: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame, float, float, float, int, float, float, float, float], float] | None = None,
    n_trials: int = 50,
    ranges: dict | None = None,
) -> optuna.study.Study:
    """Optimize strategy parameters with Optuna."""
    if backtest_func is None:
        backtest_func = default_backtest
    if ranges is None:
        ranges = OPTUNA_RANGES

    def objective(trial: optuna.trial.Trial) -> float:
        sl_pct = trial.suggest_float("sl_pct", *ranges["sl_pct"])
        tp_pct = trial.suggest_float("tp_pct", *ranges["tp_pct"])
        threshold = trial.suggest_float("threshold", *ranges["threshold"])
        horizon = trial.suggest_int("horizon", *ranges["horizon"])
        proba = trial.suggest_float("proba_filter", *ranges["proba_filter"])
        adx = trial.suggest_float("adx_threshold", *ranges["adx_threshold"])
        rsi_ob = trial.suggest_float("rsi_overbought", *ranges["rsi_overbought"])
        rsi_os = trial.suggest_float("rsi_oversold", *ranges["rsi_oversold"])
        return backtest_func(df, sl_pct, tp_pct, threshold, horizon, proba, adx, rsi_ob, rsi_os)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    last = memory_manager.last_best_params()
    if last:
        try:
            study.enqueue_trial(last)
        except Exception:
            pass
    study.optimize(objective, n_trials=n_trials)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    return study


def run_trade_analysis(csv_file: str = "trades_log.csv", n_trials: int = 50, ranges: dict | None = None) -> None:
    """Print trade statistics and run parameter optimization."""
    try:
        trades = load_trades(csv_file)
    except FileNotFoundError:
        logging.warning(f"Trade analysis skipped: {csv_file} not found")
        return
    metrics = compute_metrics(trades)
    logging.info("=== Trade Metrics ===")
    for key, val in metrics.items():
        logging.info(f"{key}: {val}")

    issues = analyze_errors(trades)
    if issues:
        logging.info("=== Detected Issues ===")
        for issue in issues:
            logging.info(f"- {issue}")

    recs = recommend_parameters(trades)
    logging.info("=== Parameter Recommendations ===")
    for key, val in recs.items():
        logging.info(f"{key}: {val}")

    mem_stats = memory_manager.trade_stats()
    if mem_stats["count"]:
        logging.info("=== Memory Trade Stats ===")
        for k, v in mem_stats.items():
            logging.info(f"{k}: {v}")

    study = load_optuna_study()
    if study is None:
        logging.info("=== Running Optuna Optimization ===")
        study = run_optuna(trades, n_trials=n_trials, ranges=ranges)
        save_optuna_study(study)
    else:
        logging.info("Loaded existing Optuna study")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")
    best = normalize_param_keys(study.best_params)
    best_params_cache["GLOBAL"] = best
    DEFAULT_PARAMS.update(best)
    apply_params(best)
    save_param_cache()
    memory_manager.add_event(
        "optimize",
        {
            "best_params": best,
            "best_value": study.best_value,
            "metrics": metrics,
        },
    )



# --- OHLCV Fetching -------------------------------------------------

def fetch_multi_ohlcv(
    symbol: str,
    timeframes: list[str],
    limit: int = 300,
    warn: bool = True,
) -> pd.DataFrame | None:
    """Synchronously fetch OHLCV data via :class:`ExchangeAdapter`.

    On ``AdapterOHLCVUnavailable`` a skip is logged and an empty ``DataFrame``
    is returned so that the caller can gracefully skip the symbol.
    """

    dfs: dict[str, pd.DataFrame] = {}
    raw_dfs: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        ohlcv = None
        for attempt in range(3):
            try:
                ohlcv = ADAPTER.fetch_ohlcv(symbol, tf, limit=limit)
                break
            except AdapterOHLCVUnavailable as exc:
                if attempt == 2:
                    log_once(
                        "warning",
                        f"data | {symbol} | {tf} ohlcv unavailable: {exc}",
                    )
                time.sleep(0.1 * (attempt + 1))
            except Exception as exc:  # pragma: no cover - network errors
                if attempt == 2:
                    log_once(
                        "warning",
                        f"data | {symbol} | {tf} fetch failed: {exc}",
                    )
                time.sleep(0.1 * (attempt + 1))
        if not ohlcv:
            log_once(
                "warning",
                f"data | {symbol} | {tf} returned no candles",
            )
            continue
        df_raw = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
        raw_dfs[tf] = df_raw
        df = df_raw.rename(columns=lambda c: f"{c}_{tf}")
        dfs[tf] = df
    if not dfs:
        message = f"data | {symbol} | no OHLCV for required timeframes; skipping"
        if warn:
            log_once("warning", message)
        else:
            log_once("info", message)
        record_no_data(symbol, "multi", "no_timeframes")
        log_decision(symbol, "ohlcv_unavailable")
        return None

    ordered = [tf for tf in timeframes if tf in dfs]
    base_tf = ordered[0]
    result = dfs[base_tf]
    for tf in ordered[1:]:
        result = pd.merge_asof(
            result.sort_values(f"timestamp_{base_tf}"),
            dfs[tf].sort_values(f"timestamp_{tf}"),
            left_on=f"timestamp_{base_tf}",
            right_on=f"timestamp_{tf}",
            direction="backward",
        )

    missing = [tf for tf in timeframes if tf not in dfs]
    if missing:
        msg = "data | %s | partial OHLCV; missing %s" % (
            symbol,
            ",".join(missing),
        )
        if warn:
            log_once("warning", msg)
        else:
            log_once("info", msg)
        result.attrs["reduced"] = True

    required_timeframes: list[str] = []
    if "15m" in timeframes:
        required_timeframes.append("15m")
    elif timeframes:
        required_timeframes.append(timeframes[0])
    expected_cols = [f"close_{tf}" for tf in required_timeframes]
    missing_cols = [col for col in expected_cols if col not in result.columns]
    if result is None or result.empty:
        log_once("warning", f"data | {symbol} | empty OHLCV result")
        record_no_data(symbol, "multi", "empty_result")
        log_decision(symbol, "ohlcv_empty")
        return None
    if missing_cols:
        level = "warning" if warn else "info"
        log_once(
            level,
            f"data | {symbol} | missing columns: {', '.join(missing_cols)}",
        )
        record_no_data(symbol, "multi", "missing_columns")
        log_decision(symbol, "ohlcv_missing_columns")
        return None

    clear_no_data(symbol, "multi")

    try:
        result.attrs["sources"] = raw_dfs
    except Exception:  # pragma: no cover - defensive attr assignment
        pass

    return result


def _health_check(symbols: list[str]) -> None:
    """Verify model availability and basic data fetch before trading."""

    fatal_issues: list[str] = []
    data_issues: list[str] = []
    if (
        GLOBAL_MODEL is None
        or not hasattr(GLOBAL_MODEL, "classes_")
        or len(getattr(GLOBAL_MODEL, "classes_", [])) < 3
    ):
        fatal_issues.append("model_unavailable")

    if timeframes:
        required_col = "close_15m" if "15m" in timeframes else f"close_{timeframes[0]}"
    else:
        required_col = "close_15m"

    for sym in symbols:
        try:
            df = fetch_multi_ohlcv(sym, timeframes, limit=5, warn=False)
        except Exception as exc:  # pragma: no cover - defensive
            data_issues.append(f"{sym} ({exc})")
            continue

        if df is None or df.empty:
            data_issues.append(f"{sym} (empty)")
            continue
        if required_col not in df.columns:
            data_issues.append(f"{sym} (missing {required_col})")

    if fatal_issues:
        msg = "health check failed: " + ", ".join(fatal_issues)
        logging.error(msg)
        raise RuntimeError(msg)

    if data_issues:
        msg = "health check degraded: data unavailable for " + ", ".join(data_issues)
        log_once("warning", msg, window_sec=300)

# === Подгружаем обученную модель CNN ===
# Если файл отсутствует, создаём небольшую обучающую выборку
# с примитивными изображениями паттернов и тренируем простую
# модель, способную различать эти паттерны.

# Pattern classes from the crypto price chart patterns dataset
pattern_classes = [
    "none",
    "ascending_triangle",
    "descending_triangle",
    "double_top",
    "double_bottom",
    "rising_wedge",
    "falling_wedge",
    "symmetrical_triangle",
    # Дополнительные паттерны, создаваемые синтетически
    "flag",
    "pennant",
    "wedge_up",
    "wedge_down",
    "triangle_sym",
    "triangle_asc",
    "triangle_desc",
    "head_and_shoulders",
    "inverse_head_and_shoulders",
]


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)


model_path = os.path.join(os.path.dirname(__file__), "pattern_model.pt")
# Директория с реальными изображениями паттернов, если доступна
PATTERN_DATA_DIR = os.getenv("PATTERN_DATA_DIR", os.path.join(os.path.dirname(__file__), "pattern_data"))

# Dataset with labelled crypto chart pattern images
PATTERN_DATA_URL = "https://www.kaggle.com/datasets/suluharif/crypto-price-chart-patterns/croissant/download"


def ensure_patterns_dataset(data_dir: str) -> None:
    """Download and prepare the stock chart pattern dataset if missing."""
    if os.path.isdir(data_dir) and any(os.scandir(data_dir)):
        return

    os.makedirs(data_dir, exist_ok=True)
    try:
        dataset = mlc.Dataset(PATTERN_DATA_URL)
        archive = next((fo for fo in dataset.metadata.file_objects if fo.name == "archive.zip"), None)
        if not archive:
            logging.error("Pattern dataset archive not found in metadata")
            return
        logging.info("Downloading real pattern dataset...")
        resp = requests.get(archive.content_url)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(data_dir)

        csv_path = os.path.join(data_dir, "Patterns.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for cls_name, group in df.groupby("ClassName"):
                folder = os.path.join(data_dir, cls_name.lower().replace(" ", "_"))
                os.makedirs(folder, exist_ok=True)
                for rel_path in group["Path"]:
                    src = os.path.join(data_dir, rel_path.replace("/", os.sep))
                    if os.path.isfile(src):
                        shutil.copy(src, os.path.join(folder, os.path.basename(rel_path)))
        else:
            dataset_root = os.path.join(data_dir, "DATASET")
            if os.path.isdir(dataset_root):
                for name in os.listdir(dataset_root):
                    src_dir = os.path.join(dataset_root, name)
                    if not os.path.isdir(src_dir):
                        continue
                    folder = os.path.join(data_dir, name.lower())
                    os.makedirs(folder, exist_ok=True)
                    for img in os.listdir(src_dir):
                        if img.lower().endswith((".png", ".jpg", ".jpeg")):
                            shutil.copy(os.path.join(src_dir, img), os.path.join(folder, img))
        logging.info(f"Pattern dataset downloaded to {data_dir}")
    except Exception as e:
        logging.error(f"Failed to download pattern dataset: {e}")


def _series_from_points(points, length=60):
    xp = np.linspace(0, 1, len(points))
    x = np.linspace(0, 1, length)
    return np.interp(x, xp, points)


_base_patterns = {
    "ascending_triangle": [0.2, 0.8, 0.6, 0.8, 0.7, 0.8],
    "descending_triangle": [0.8, 0.3, 0.5, 0.3, 0.4, 0.3],
    "symmetrical_triangle": [0.2, 0.9, 0.4, 0.8, 0.6, 0.7, 0.65],
    "rising_wedge": [0.2, 0.4, 0.55, 0.65, 0.7, 0.72],
    "falling_wedge": [0.8, 0.6, 0.45, 0.35, 0.3, 0.28],
    "double_top": [0.2, 0.8, 0.4, 0.8, 0.2],
    "double_bottom": [0.8, 0.2, 0.6, 0.2, 0.8],
    # Новые синтетические паттерны
    "flag": [0.2, 0.85, 0.8, 0.82, 0.81, 0.83],
    "pennant": [0.2, 0.85, 0.7, 0.8, 0.75, 0.78],
    "wedge_up": [0.2, 0.4, 0.55, 0.65, 0.7, 0.72],
    "wedge_down": [0.8, 0.6, 0.45, 0.35, 0.3, 0.28],
    "triangle_sym": [0.2, 0.9, 0.4, 0.8, 0.6, 0.7, 0.65],
    "triangle_asc": [0.2, 0.8, 0.6, 0.8, 0.7, 0.8],
    "triangle_desc": [0.8, 0.3, 0.5, 0.3, 0.4, 0.3],
    "head_and_shoulders": [0.2, 0.8, 0.4, 0.9, 0.5, 0.8, 0.3],
    "inverse_head_and_shoulders": [0.8, 0.2, 0.6, 0.1, 0.5, 0.2, 0.7],
}


def _generate_series(name, length=60):
    if name == "none" or name not in _base_patterns:
        return np.cumsum(np.random.normal(0, 0.05, length))
    base = _base_patterns[name]
    series = _series_from_points(base, length)
    noise = np.random.normal(0, 0.02, length)
    return series + noise


def _series_to_image(series):
    if Image is None:
        raise RuntimeError("Pillow not installed")

    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), "black")
    draw = ImageDraw.Draw(img)
    series = (series - series.min()) / (series.max() - series.min() + 1e-8)
    points = [(int(i / (len(series) - 1) * 223), 223 - int(v * 223)) for i, v in enumerate(series)]
    draw.line(points, fill="white", width=2)
    return transforms.ToTensor()(img)


def _generate_dataset(num_per_class: int = 50, length: int = 60, classes=None):
    """Generate a synthetic dataset for the given pattern ``classes``.

    The returned dataset uses the global ``pattern_classes`` indexing so that
    synthetic samples can be seamlessly combined with real images.
    """
    from torch.utils.data import TensorDataset

    if classes is None:
        classes = pattern_classes

    images = []
    labels = []
    for name in classes:
        target_idx = pattern_classes.index(name)
        for _ in range(num_per_class):
            series = _generate_series(name, length)
            img = _series_to_image(series)
            images.append(img)
            labels.append(target_idx)

    x = torch.stack(images)
    y = torch.tensor(labels)
    return TensorDataset(x, y)


def train_pattern_model(path: str, data_dir: str | None = None) -> torch.nn.Module:
    """Train a simple CNN model. If ``data_dir`` contains real pattern
    images arranged in subfolders per class, they will be used. Otherwise
    synthetic data is generated."""
    from torch.utils.data import DataLoader, ConcatDataset

    datasets_list = []
    present_classes: set[str] = set()

    if data_dir and os.path.isdir(data_dir):
        try:

            class PatchedImageFolder(ImageFolder):
                def __init__(self, root, transform=None):
                    super().__init__(root, transform=transform)
                    valid_samples = []
                    self.present: set[str] = set()
                    for path, target in self.samples:
                        cls_name = self.classes[target]
                        if cls_name in pattern_classes:
                            self.present.add(cls_name)
                            new_idx = pattern_classes.index(cls_name)
                            valid_samples.append((path, new_idx))
                    self.samples = valid_samples
                    self.targets = [s[1] for s in valid_samples]
                    self.classes = pattern_classes
                    self.class_to_idx = {c: i for i, c in enumerate(pattern_classes)}

                def __getitem__(self, index):
                    img, target = super().__getitem__(index)
                    return img, torch.tensor(target)

            real_ds = PatchedImageFolder(
                data_dir,
                transform=transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ]
                ),
            )
            if len(real_ds) > 0:
                datasets_list.append(real_ds)
                present_classes.update(real_ds.present)
                logging.info(f"Loaded real pattern dataset from {data_dir}")
        except Exception as e:
            logging.error(f"Failed to load real dataset: {e}")

    missing = [c for c in pattern_classes if c not in present_classes]
    if missing:
        logging.info(f"Using synthetic data for classes: {', '.join(missing)}")
        datasets_list.append(_generate_dataset(classes=missing))

    if not datasets_list:
        # No real images and no synthetic classes for some reason
        logging.info("Using synthetic pattern data")
        datasets_list.append(_generate_dataset())

    if len(datasets_list) > 1:
        dataset = ConcatDataset(datasets_list)
    else:
        dataset = datasets_list[0]

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN(len(pattern_classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(3):  # несколько эпох для лучшей сходимости
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model, path)
    return model


pattern_model = None
pattern_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

if os.environ.get("LOAD_PATTERN_MODEL") == "1":
    if not os.path.exists(model_path):
        logging.info(f"Model file not found: {model_path}. Training pattern model...")
        ensure_patterns_dataset(PATTERN_DATA_DIR)
        pattern_model = train_pattern_model(model_path, PATTERN_DATA_DIR)
    else:
        pattern_model = torch.load(model_path, map_location="cpu", weights_only=False)
    pattern_model.eval()

# API credentials
from credentials import API_KEY, API_SECRET

# Leverage used for all pairs
LEVERAGE = int(os.getenv("LEVERAGE", 20))

SANDBOX_MODE = True

capital_fraction = 0.10  # базовый размер позиции: 10% от баланса
MAX_POSITION_PCT = 0.10  # абсолютный лимит позиции: 10% от капитала
# Минимальный нотационал сделки в USDT
MIN_NOTIONAL = 10.0
# Минимальное отклонение для установки trailing stop без мгновенного срабатывания
# Используется как базовое значение, фактический оффсет выбирается случайно в
# диапазоне 0.4–0.7% при постановке трейлинга.
TRAIL_ACTIVATION_OFFSET = 0.005  # 0.5%
# === PATCH: КЛЮЧЕВЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
# Сниженные пороги для более активной торговли
THRESHOLD = 0.001  # 0.1% минимальное движение для сделки
SL_PCT = 0.02  # 2% стоп-лосс
TP_PCT = 0.05  # 5% тейк-профит для RR ≈ 2.5
MIN_RR = 1.5  # минимальное соотношение TP/SL


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


# Базовые и динамические фильтры вероятности/ADX
BASE_PROBA_FILTER = 0.25
PROBA_FILTER = _env_float("PROBA_FILTER", BASE_PROBA_FILTER)  # динамическое значение
# [ANCHOR:DYNA_THRESH_CONSTS]
MIN_PROBA_FILTER = _env_float(
    "MIN_PROBA_FILTER", min(0.4, float(PROBA_FILTER))
)
# Стратегия допускает сделки только при умеренном тренде
BASE_ADX_THRESHOLD = 15.0
ADX_THRESHOLD = _env_float("ADX_THRESHOLD", BASE_ADX_THRESHOLD)  # минимальный ADX для сделки
MIN_ADX_THRESHOLD = _env_float(
    "MIN_ADX_THRESHOLD", min(12.0, float(ADX_THRESHOLD))
)
RSI_OVERBOUGHT = _env_float("RSI_OVERBOUGHT", 75.0)  # порог перекупленности для long
RSI_OVERSOLD = _env_float("RSI_OVERSOLD", 25.0)  # порог перепроданности для short
RSI_OVERBOUGHT_MAX = max(RSI_OVERBOUGHT, _env_float("RSI_OVERBOUGHT_MAX", 80.0))
RSI_OVERSOLD_MIN = min(RSI_OVERSOLD, _env_float("RSI_OVERSOLD_MIN", 20.0))
PRED_HORIZON = 3  # число свечей вперёд для прогноза и бэктеста
MAX_LOSS_ROI = 0.10  # допустимый убыток по позиции (10% ROI)
ROI_TARGET_PCT = 1.5  # целевой ROI для автофиксации прибыли (в процентах)
ALLOW_FALLBACK_ENTRY = True  # allow entering trades when only fallback signal confirms
ALLOW_MARKET_FALLBACK = True  # switch to market order if limit rejected
MAX_PERCENT_DIFF = 0.0015  # max deviation from best price for limit orders
RISK_PER_TRADE = 0.03  # 3% of equity risked per trade
# базовые пороги объёма для фильтра ликвидности
VOLUME_RATIO_MIN = float(os.getenv("VOLUME_RATIO_MIN", "1.0"))
VOLUME_RATIO_ENTRY = float(os.getenv("VOLUME_RATIO_ENTRY", "1.0"))

# PATCH NOTES:
# Что изменено:
# 1) RISK_PER_TRADE по умолчанию снижен до 3%, а VOLUME_RATIO_MIN/VOLUME_RATIO_ENTRY теперь >= 1.0.
# 2) run_trade снижает риск при vol_missing и пропускает сделки при vol_low перед расчётом объёма.
# Почему безопасно:
# 1) Пороговые значения переопределяются ENV/конфигом, а log_once защищает от спама при частых проверках.
# Критерии приёмки:
# - vol_low приводит к log_decision("volume_low") и возврату False до отправки ордера.
# - vol_reason == "vol_missing" уменьшает risk_factor до 50% перед расчётом qty.
# - qty_target рассчитывается с effective_risk_factor >= 0.

# Minimum number of agreeing lower timeframe signals required for entry
# разрешаем вход даже без совпадений по младшим таймфреймам
MIN_LOWER_TF_MATCHES = int(os.getenv("MIN_LOWER_TF_MATCHES", "0"))

# [ANCHOR:TRAIL_PARAMS]
TRAIL_ACTIVATE_R = 0.5
TRAIL_ACTIVATE_ATR = 0.75
TRAIL_OFFSET_ATR = 1.2
TRAIL_MIN_TICKS = 3
USE_BREAKEVEN_STEP = True
BREAKEVEN_BUFFER_ATR = 0.2

# Inactivity adaptation thresholds (hours)
INACTIVITY_ADAPT_HOURS = 0.5
INACTIVITY_CONDITIONAL_HOURS = 1

# PATCH NOTES:
# - Обновлены базовые TP/SL, ADX, RSI и PRED_HORIZON.
# - Безопасно: значения можно переопределить через ENV, MIN_RR контролирует TP.
# - Критерии: TP/SL >= 1.5, ADX-фильтр логирует пропуски, модель использует горизонт 3.
INACTIVITY_FALLBACK_HOURS = 2
BOT_CYCLE_MINUTES = _env_float("BOT_CYCLE_MINUTES", 5.0)
INACTIVITY_ADAPT_CYCLES = int(os.getenv("INACTIVITY_ADAPT_CYCLES", "6"))
INACTIVITY_PROBA_STEP = _env_float("INACTIVITY_PROBA_STEP", 0.05)
INACTIVITY_ADX_STEP = _env_float("INACTIVITY_ADX_STEP", 1.0)
INACTIVITY_RSI_STEP = _env_float("INACTIVITY_RSI_STEP", 1.0)
FALLBACK_MODE_ENABLED = os.getenv("FALLBACK_MODE", "0").lower() in {"1", "true", "yes"}
NO_DATA_RETRY_SEC = int(os.getenv("NO_DATA_RETRY_SEC", "180"))

# --- No-trade telemetry ---
from collections import defaultdict, deque

SUMMARY_CYCLES = 20  # print summary every N bot cycles
PATTERN_HIT_LOG_CYCLES = 50  # log pattern hit rate every N cycles
_event_counters: defaultdict[str, float] = defaultdict(float)
_cycle_counter = 0


def _inc_event(name: str, value: float = 1.0) -> None:
    """Increase telemetry counter ``name`` by ``value``."""
    _event_counters[name] += value


def _summarize_events() -> None:
    """Log current no-trade telemetry and reset counters."""
    if _event_counters:
        logging.info("no-trade summary: %s", dict(_event_counters))
        _event_counters.clear()

recent_hits: deque[bool] = deque(maxlen=50)

# Count consecutive bad prediction lengths to trigger fallback usage
recent_trade_times: deque[datetime] = deque(maxlen=50)
pattern_hit_rates: dict[str, float] = {}

logging_utils.ALLOW_MARKET_FALLBACK = ALLOW_MARKET_FALLBACK
logging_utils.MAX_PERCENT_DIFF = MAX_PERCENT_DIFF


def update_dynamic_thresholds() -> None:
    """Adapt dynamic PROBA/ADX thresholds and volume filter."""
    global PROBA_FILTER, VOLUME_RATIO_MIN, ADX_THRESHOLD
    hit_rate = sum(recent_hits) / len(recent_hits) if recent_hits else 0
    if len(recent_hits) >= 20 and hit_rate < 0.25:
        PROBA_FILTER = min(0.9, PROBA_FILTER + 0.03)
        ADX_THRESHOLD = min(30, ADX_THRESHOLD + 2)
    trades_per_hour = 0.0
    if recent_trade_times:
        span = (datetime.now(timezone.utc) - recent_trade_times[0]).total_seconds() / 3600
        if span > 0:
            trades_per_hour = len(recent_trade_times) / span
    if trades_per_hour < 1:
        PROBA_FILTER = max(MIN_PROBA_FILTER, PROBA_FILTER - 0.03)
        ADX_THRESHOLD = max(MIN_ADX_THRESHOLD, ADX_THRESHOLD - 2)
        VOLUME_RATIO_MIN = max(1.0, VOLUME_RATIO_MIN)
    else:
        VOLUME_RATIO_MIN = max(1.0, VOLUME_RATIO_MIN)
    PROBA_FILTER = max(MIN_PROBA_FILTER, PROBA_FILTER)
    ADX_THRESHOLD = max(MIN_ADX_THRESHOLD, ADX_THRESHOLD)

# Thresholds for trade mode selection
ATR_THRESHOLD = 0.005
VOLUME_THRESHOLD = 1_000_000
STRENGTH_THRESHOLD = 0.5
VOLATILITY_5M_THRESHOLD = 0.01

risk_config = load_config()
risk_state, limiter, cool, stats = load_risk_state(risk_config)
MAX_LOSS_ROI = risk_config.get("max_trade_loss_pct", MAX_LOSS_ROI)
def _resolve_max_open_trades(config: dict) -> int:
    try:
        env_val = int(os.getenv("MAX_OPEN_TRADES", "0"))
    except ValueError:
        env_val = 0
    if env_val > 0:
        return env_val
    try:
        return int(config.get("max_open_trades", 5))
    except (TypeError, ValueError):
        return 5


MAX_OPEN_TRADES = _resolve_max_open_trades(risk_config)

# Убедимся, что тейк-профит не меньше, чем стоп-лосс с учётом минимального RR
if TP_PCT / SL_PCT < MIN_RR:
    TP_PCT = round(SL_PCT * MIN_RR, 4)
    logging.info(f"params | GLOBAL | TP_PCT adjusted to {TP_PCT:.4f} to satisfy RR >= {MIN_RR}")

# Параметры и их диапазоны для подбора Optuna
OPTUNA_RANGES = {
    "sl_pct": (0.005, 0.05),  # 0.5-5%
    "tp_pct": (0.01, 0.1),  # 1-10%
    "threshold": (0.0005, 0.01),  # 0.05-1%
    "horizon": (1, 30),
    "proba_filter": (0.4, 0.8),
    "adx_threshold": (10, 30),
    "rsi_overbought": (60, 80),
    "rsi_oversold": (20, 40),
}

# Сохраняем параметры по умолчанию и словарь оптимальных параметров
DEFAULT_PARAMS = dict(
    THRESHOLD=THRESHOLD,
    SL_PCT=SL_PCT,
    TP_PCT=TP_PCT,
    HORIZON=PRED_HORIZON,
    # используем базовые константы (0.25, 2.0, 65/35)
    PROBA_FILTER=BASE_PROBA_FILTER,
    ADX_THRESHOLD=BASE_ADX_THRESHOLD,
    RSI_OVERBOUGHT=RSI_OVERBOUGHT,
    RSI_OVERSOLD=RSI_OVERSOLD,
)


@dataclass
class StrategyParams:
    """Container for strategy settings for a single symbol."""

    THRESHOLD: float
    SL_PCT: float
    TP_PCT: float
    HORIZON: int
    PROBA_FILTER: float
    ADX_THRESHOLD: float
    RSI_OVERBOUGHT: float
    RSI_OVERSOLD: float

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "StrategyParams":
        d = normalize_param_keys(data)
        return cls(
            THRESHOLD=d.get("THRESHOLD", THRESHOLD),
            SL_PCT=d.get("SL_PCT", SL_PCT),
            TP_PCT=d.get("TP_PCT", TP_PCT),
            HORIZON=d.get("HORIZON", PRED_HORIZON),
            PROBA_FILTER=d.get("PROBA_FILTER", PROBA_FILTER),
            ADX_THRESHOLD=d.get("ADX_THRESHOLD", ADX_THRESHOLD),
            RSI_OVERBOUGHT=d.get("RSI_OVERBOUGHT", RSI_OVERBOUGHT),
            RSI_OVERSOLD=d.get("RSI_OVERSOLD", RSI_OVERSOLD),
        )


# Файл для хранения лучших параметров (итоги grid search)
PARAM_CACHE_FILE = os.path.join(os.path.dirname(__file__), "best_params.json")
# File for persisting Optuna optimization studies
OPTUNA_STUDY_FILE = os.path.join(os.path.dirname(__file__), "optuna_study.pkl")


def load_param_cache() -> dict:
    """Load cached parameter grid search results from disk."""
    if os.path.exists(PARAM_CACHE_FILE):
        try:
            with open(PARAM_CACHE_FILE, "r") as f:
                params = json.load(f)
            logging.info("Loaded cached parameters")
            return params
        except Exception as e:
            logging.error(f"Failed to load param cache: {e}")
    return {}


# Словарь результатов оптимизации (загружается один раз при старте)
best_params_cache = load_param_cache()


def get_symbol_params(symbol: str) -> StrategyParams:
    """Return parameters for a symbol as a dataclass."""
    data = best_params_cache.get(symbol)
    if data is None:
        data = best_params_cache.get("GLOBAL", DEFAULT_PARAMS)
    return StrategyParams.from_dict(data)


def save_param_cache():
    """Persist best parameter settings to disk."""
    try:
        with open(PARAM_CACHE_FILE, "w") as f:
            json.dump(best_params_cache, f)
    except Exception as e:
        logging.error(f"Failed to save param cache: {e}")


def save_optuna_study(study: optuna.study.Study, path: str = OPTUNA_STUDY_FILE) -> None:
    """Persist Optuna study to a file for later inspection or resuming."""
    try:
        joblib.dump(study, path)
    except Exception as e:
        logging.error(f"Failed to save Optuna study: {e}")


def load_optuna_study(path: str = OPTUNA_STUDY_FILE) -> optuna.study.Study | None:
    """Load a previously saved Optuna study if it exists."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Failed to load Optuna study: {e}")
    return None


# === Inactivity tracking ===
symbol_activity: Dict[str, datetime] = {}
open_trade_ctx: Dict[str, Dict[str, Any]] = {}
_last_exit_qty: Dict[str, float] = {}
exit_orders_fetch_guard: Dict[str, Dict[str, bool]] = {}
fallback_cooldown: Dict[str, int] = {}
tf_skip_counters: Dict[str, int] = defaultdict(int)
_entry_guard: Dict[str, Dict[str, Any]] = {}
TF_SKIP_THRESHOLD = 3


def inc_tf_skip(symbol: str) -> None:
    """Increment consecutive timeframe skip counter for ``symbol``."""
    tf_skip_counters[symbol] += 1
    memory_manager.add_event("tf_skip", {"symbol": symbol, "count": tf_skip_counters[symbol]})


def reset_tf_skip(symbol: str) -> None:
    """Reset timeframe skip counter for ``symbol`` if set."""
    if tf_skip_counters.get(symbol):
        tf_skip_counters[symbol] = 0
        memory_manager.add_event("tf_skip_reset", {"symbol": symbol})


def load_last_trade_times(
    log_path: str = os.path.join(os.path.dirname(__file__), "trades_log.csv")
) -> Dict[str, datetime]:
    """Load last trade timestamp for each symbol from trade log."""
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if "timestamp" in df.columns and "symbol" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                return df.groupby("symbol")["timestamp"].max().to_dict()
        except Exception as e:
            logging.error(f"Failed to load trade log for activity: {e}")
    return {}


symbol_activity = load_last_trade_times()


def adjust_filters_for_inactivity(
    symbol: str,
) -> tuple[float, float, bool, bool, float, float, float]:
    """Return adjusted filter thresholds based on symbol inactivity."""

    last = symbol_activity.get(symbol)
    if last is None:
        symbol_activity[symbol] = datetime.now(timezone.utc)
        inactivity = 0.0
    else:
        delta = datetime.now(timezone.utc) - pd.to_datetime(last, utc=True)
        inactivity = max(delta.total_seconds(), 0.0) / 3600

    allow_conditional = False
    use_fallback = False

    cycle_minutes = max(BOT_CYCLE_MINUTES, 1.0)
    cycles = int((inactivity * 60.0) / cycle_minutes)
    adapt_cycles = max(INACTIVITY_ADAPT_CYCLES, 1)
    steps = max(cycles // adapt_cycles, 0)
    if inactivity < INACTIVITY_ADAPT_HOURS:
        steps = 0

    adj_proba = max(
        MIN_PROBA_FILTER,
        float(PROBA_FILTER) - float(steps) * float(INACTIVITY_PROBA_STEP),
    )
    adj_adx = max(
        float(MIN_ADX_THRESHOLD),
        float(ADX_THRESHOLD) - float(steps) * float(INACTIVITY_ADX_STEP),
    )
    adj_rsi_overbought = min(
        float(RSI_OVERBOUGHT_MAX),
        float(RSI_OVERBOUGHT) + float(steps) * float(INACTIVITY_RSI_STEP),
    )
    adj_rsi_oversold = max(
        float(RSI_OVERSOLD_MIN),
        float(RSI_OVERSOLD) - float(steps) * float(INACTIVITY_RSI_STEP),
    )

    if inactivity >= INACTIVITY_CONDITIONAL_HOURS:
        allow_conditional = True
    if inactivity >= INACTIVITY_FALLBACK_HOURS:
        use_fallback = True

    # [ANCHOR:DYNA_THRESH_LOG]
    logging.info(
        "adj | %s | inactive %.1fh → PROBA %.2f, ADX %.1f, RSI %.1f/%.1f",
        symbol,
        inactivity,
        adj_proba,
        adj_adx,
        adj_rsi_overbought,
        adj_rsi_oversold,
    )
    return (
        adj_proba,
        adj_adx,
        allow_conditional,
        use_fallback,
        inactivity,
        adj_rsi_overbought,
        adj_rsi_oversold,
    )


# stop_loss_pct = 0.008  # === больше не используется, см. выше ===


def apply_pattern_proba_bonus(adj_proba: float, pattern_conf: float, trend_ok: bool) -> float:
    """Сильные паттерны заметно снижают требуемую вероятность."""

    if pattern_conf >= 0.7 and trend_ok:
        return max(MIN_PROBA_FILTER, adj_proba - 0.20)
    return adj_proba

ADAPTER_READY = False
try:
    ADAPTER = ExchangeAdapter(
        config={
            "sandbox": SANDBOX_MODE,
            "futures": True,
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "exchange_id": "bybit",
        }
    )
    exchange = ADAPTER.x  # legacy name to reduce changes in the rest of the file
    try:
        ADAPTER.load_markets()  # via ExchangeAdapter
    except Exception as e:  # pragma: no cover - log only
        logging.warning("adapter | load_markets skipped: %s", e)
    ADAPTER_READY = bool(exchange)
except AdapterInitError as e:  # pragma: no cover - init diagnostics
    logging.error("main | adapter init failed: %s", e)
    ADAPTER = ExchangeAdapter.__new__(ExchangeAdapter)
    ADAPTER.backend = "ccxt"
    ADAPTER.x = None
    ADAPTER.sdk = None
    ADAPTER.last_warn_at = {}
    ADAPTER.markets_loaded_at = 0.0
    ADAPTER.config = {}
    ADAPTER.sandbox = SANDBOX_MODE
    ADAPTER.futures = True
    ADAPTER_READY = False
    exchange = None
except Exception as e:  # pragma: no cover - unexpected init failure
    logging.exception("main | unexpected adapter init failure")
    ADAPTER = ExchangeAdapter.__new__(ExchangeAdapter)
    ADAPTER.backend = "ccxt"
    ADAPTER.x = None
    ADAPTER.sdk = None
    ADAPTER.last_warn_at = {}
    ADAPTER.markets_loaded_at = 0.0
    ADAPTER.config = {}
    ADAPTER.sandbox = SANDBOX_MODE
    ADAPTER.futures = True
    ADAPTER_READY = False
    exchange = None

# PATCH NOTES:
# - Sandbox startup now tracks adapter readiness and aborts trading when unavailable.
# - Keeps ccxt stub fallback for tests while surfacing detailed init errors in logs.
# - Criteria: logs include sandbox status; main() exits if adapter missing.


def initialize_symbols() -> list[str]:
    """Return list of tradable pairs. Market scanning is currently disabled."""
    default = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "APT/USDT",
        "AVAX/USDT",
        "DOGE/USDT",
        "LINK/USDT",
        "NEAR/USDT",
        "OP/USDT",
    ]
    # Удаляем возможные дубликаты, сохраняя порядок
    seen: set[str] = set()
    unique = []
    for s in default:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    # PATCH NOTES:
    # - Расширен список ликвидных линейных контрактов (BTC, ETH, SOL, BNB, ADA + APT, AVAX, DOGE, LINK, NEAR, OP).
    # - Безопасно: filter_supported_symbols убирает неподдерживаемые рынки и обновляет BASE_SYMBOL_COUNT.
    # - Критерии: initialize_symbols содержит только активные линейные пары без контрактов без OHLCV.
    # Scanning for new symbols is temporarily disabled.
    return unique


# Торгуемые пары
symbols = initialize_symbols()
# [ANCHOR:SYMBOLS_FILTER_APPLY]
symbols, _removed, degraded = filter_supported_symbols(ADAPTER, symbols, markets_cache)
if degraded:
    logging.info("filter | degraded mode enabled; proceeding with original symbols to avoid idle")
BASE_SYMBOL_COUNT = len(symbols)
reserve_symbols, _res_removed, _res_degraded = filter_supported_symbols(
    ADAPTER, risk_config.get("reserve_symbols", []), markets_cache
)

SYMBOL_CATEGORIES = {s: i for i, s in enumerate(symbols)}
PATTERN_LABELS = {name: i for i, name in enumerate(pattern_classes)}
PATTERN_SOURCE_MAP = {"cnn": 1, "manual": 2, "real": 3, "synthetic": 4, "none": 0}


def calculate_indicators(df):
    """Enrich OHLCV dataframe with indicators while keeping rows usable.

    Historically this helper relied on :meth:`pandas.DataFrame.dropna` which
    caused the caller to receive an empty dataframe whenever a division by
    zero (for example during flat candles) introduced NaNs.  The trading loop
    would then emit ``"Chart data unavailable"`` warnings and skip pattern
    recognition altogether.  The updated implementation keeps computations
    numerically stable by using ``min_periods=1`` windows, guarding every
    division and normalising problematic values instead of discarding the
    entire dataset.
    """

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["sma_100"] = df["close"].rolling(window=100, min_periods=1).mean()

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    loss_zero = avg_loss == 0
    gain_zero = avg_gain == 0
    valid_mask = ~(loss_zero & gain_zero)
    rs.loc[valid_mask] = avg_gain.loc[valid_mask] / avg_loss.loc[valid_mask].replace(0, np.nan)
    rs.loc[loss_zero & ~gain_zero] = np.inf
    df["rsi"] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan)))
    df.loc[loss_zero & ~gain_zero, "rsi"] = 100.0
    df.loc[gain_zero & ~loss_zero, "rsi"] = 0.0
    df.loc[gain_zero & loss_zero, "rsi"] = 50.0

    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["volatility"] = df["close"].pct_change(fill_method=None).rolling(50, min_periods=1).std(ddof=0)
    df["atr"] = (df["high"] - df["low"]).rolling(14, min_periods=1).mean()

    high_roll = df["high"].rolling(14, min_periods=1)
    low_roll = df["low"].rolling(14, min_periods=1)
    range_span = high_roll.max() - low_roll.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["willr"] = np.where(
            range_span == 0,
            0.0,
            (high_roll.max() - df["close"]) / range_span * -100,
        )

    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_mean = tp.rolling(20, min_periods=1).mean()
    tp_std = tp.rolling(20, min_periods=1).std(ddof=0)
    denom = 0.015 * tp_std
    df["cci"] = np.where(denom == 0, 0.0, (df["close"] - tp_mean) / denom)

    df["candle_range"] = df["high"] - df["low"]
    df["volume_ema"] = df["volume"].ewm(span=20, adjust=False).mean()
    df["trend_strength"] = (df["ema_fast"] - df["ema_slow"]).abs()
    df["ema_200_slope"] = df["ema_200"].diff()
    df["trend_direction"] = np.sign(df["ema_fast"] - df["ema_slow"])
    df["trend_persistence"] = df["trend_direction"].rolling(20, min_periods=1).sum().fillna(0)

    df["roc"] = df["close"].pct_change(periods=10, fill_method=None)
    df["mom"] = df["close"] - df["close"].shift(10)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    close_mean_20 = df["close"].rolling(20, min_periods=1).mean()
    close_std_20 = df["close"].rolling(20, min_periods=1).std(ddof=0)
    bb_denom = 2 * close_std_20
    df["bb_b"] = np.where(bb_denom == 0, 0.0, (df["close"] - close_mean_20) / bb_denom)

    low_min = df["low"].rolling(14, min_periods=1).min()
    high_max = df["high"].rolling(14, min_periods=1).max()
    stoch_denom = high_max - low_min
    df["stoch_k"] = np.where(
        stoch_denom == 0,
        0.0,
        100 * (df["close"] - low_min) / stoch_denom,
    )
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean()

    plus_dm = (df["high"].diff()).clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_adx = tr.rolling(14, min_periods=1).mean()
    atr_adx = atr_adx.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).mean() / atr_adx)
    adx_denom = (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = 100 * (plus_di - minus_di).abs() / adx_denom
    df["adx"] = df["adx"].fillna(0.0)

    lag_periods = [1, 2, 3]
    for lag in lag_periods:
        df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
        df[f"macd_lag{lag}"] = df["macd"].shift(lag)
        df[f"ema_fast_lag{lag}"] = df["ema_fast"].shift(lag)

    for window in [5, 10, 20, 50]:
        df[f"close_rolling_mean_{window}"] = df["close"].rolling(window, min_periods=1).mean()
        df[f"close_rolling_std_{window}"] = df["close"].rolling(window, min_periods=1).std(ddof=0)
        df[f"volume_rolling_mean_{window}"] = df["volume"].rolling(window, min_periods=1).mean()
        df[f"volume_rolling_std_{window}"] = df["volume"].rolling(window, min_periods=1).std(ddof=0)
        df[f"price_range_{window}"] = (
            df["high"].rolling(window, min_periods=1).max()
            - df["low"].rolling(window, min_periods=1).min()
        )

    df["vol_div_avg20"] = df["volume"] / (df["volume"].rolling(20, min_periods=1).mean() + 1e-8)

    warmup = max(10, max(lag_periods))
    if len(df) > warmup:
        df = df.iloc[warmup:].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)

    return df


def log_candle_status(symbol: str, tf: str, count: int | None) -> None:
    """Collect candle load status for later output."""
    record_candle_status(symbol, tf, count)


# [ANCHOR:FETCH_GUARDS]
def _safe_df(data) -> pd.DataFrame | None:
    try:
        if data is None:
            return None
        cleaned: list[list[float]] = []
        for row in data:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) < 6:
                continue
            try:
                ts = int(float(row[0]))
                open_, high, low, close, volume = (
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            except (TypeError, ValueError):
                continue
            cleaned.append([ts, open_, high, low, close, volume])
        if not cleaned:
            return None
        df = pd.DataFrame(
            cleaned,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df.dropna(inplace=True)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df.dropna(inplace=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


FETCH_CACHE: dict[tuple[int, str, str, int], pd.DataFrame | None] = {}
_NO_DATA_RETRY: dict[tuple[str, str], float] = {}


def _fetch_ohlcv(symbol, tf, limit=10000):
    """Fetch OHLCV with a short-lived in-cycle cache."""
    cycle = int(time.time() // 300)
    # purge old cache entries
    to_remove = [k for k in FETCH_CACHE if k[0] != cycle]
    for k in to_remove:
        FETCH_CACHE.pop(k, None)
    key = (cycle, symbol, tf, limit)
    if key in FETCH_CACHE:
        return FETCH_CACHE[key]

    now = time.time()
    retry_after = _NO_DATA_RETRY.get((symbol, tf))
    if retry_after and now < retry_after:
        log_candle_status(symbol, tf, None)
        FETCH_CACHE[key] = None
        return None

    try:
        ohlcv = ADAPTER.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    except Exception as exc:
        ohlcv = None
        record_no_data(symbol, tf, f"exception: {exc}")
        _NO_DATA_RETRY[(symbol, tf)] = now + NO_DATA_RETRY_SEC
    if not ohlcv:
        log_candle_status(symbol, tf, None)
        if (symbol, tf) not in _NO_DATA_RETRY:
            _NO_DATA_RETRY[(symbol, tf)] = now + NO_DATA_RETRY_SEC
        record_no_data(symbol, tf, "empty_fetch")
        FETCH_CACHE[key] = None
        return None

    df = _safe_df(ohlcv)
    count = len(df) if df is not None else None
    if count is None or count == 0:
        log_candle_status(symbol, tf, None)
        _NO_DATA_RETRY[(symbol, tf)] = now + NO_DATA_RETRY_SEC
        record_no_data(symbol, tf, "empty_df")
        FETCH_CACHE[key] = None
        return None
    log_candle_status(symbol, tf, count)
    _NO_DATA_RETRY.pop((symbol, tf), None)
    clear_no_data(symbol, tf)
    result = calculate_indicators(df)
    FETCH_CACHE[key] = result
    return result


CACHE_TTL_SEC = 300


def cached_fetch_ohlcv(symbol: str, tf: str, limit: int = 200):
    key = (symbol, tf)
    now = time.time()
    ts_df = ohlcv_cache.get(key)
    if ts_df and now - ts_df[0] < CACHE_TTL_SEC:
        return ts_df[1]
    df = _fetch_ohlcv(symbol, tf, limit=limit)
    if df is not None and not df.empty:
        ohlcv_cache[key] = (now, df)
    return df


# make cached fetch the default
fetch_ohlcv = cached_fetch_ohlcv


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV dataframe to a different timeframe."""
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Pandas deprecated the 'T' alias for minutes.
    # Convert any usage to the recommended 'min' unit.
    rule = rule.replace("T", "min")
    df = df.set_index("timestamp").resample(rule).agg(ohlc).dropna().reset_index()
    return calculate_indicators(df)


features_used = [
    "ema_fast",
    "ema_slow",
    "ema_200",
    "sma_100",
    "rsi",
    "macd",
    "macd_signal",
    "atr",
    "volatility",
    "willr",
    "cci",
    "roc",
    "mom",
    "obv",
    "bb_b",
    "stoch_k",
    "stoch_d",
    "adx",
    "hour",
    "dayofweek",
    "candle_range",
    "volume_ema",
    "trend_strength",
    "ema_200_slope",
    "trend_direction",
    "trend_persistence",
    "rsi_lag1",
    "rsi_lag2",
    "rsi_lag3",
    "macd_lag1",
    "macd_lag2",
    "macd_lag3",
    "ema_fast_lag1",
    "ema_fast_lag2",
    "ema_fast_lag3",
    "pattern_name",
    "pattern_source",
    "pattern_confidence",
    "price_change_5m",
    "volume_ratio",
    "ema_slope",
    "volatility_30m",
    "symbol_strength",
]

# ---- Мультифрейм! ----
# include 30m and 4h timeframes for broader signal analysis
timeframes = ["5m", "15m", "30m", "1h", "4h", "12h", "1d"]
try:
    GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = load_global_bundle()
    logging.info("main | loaded global bundle | classes=%s", GLOBAL_CLASSES)
except Exception as e:
    logging.info("main | bundle not found yet: %s; retraining...", e)
    try:
        df_features, df_target, feature_cols = fetch_and_prepare_training_data(
            ADAPTER, symbols
        )
        GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = _retrain_checked(
            df_features, df_target, feature_cols
        )
    except Exception as e2:
        logging.error("main | retrain failed: %s", e2)
        GLOBAL_MODEL = GLOBAL_SCALER = GLOBAL_FEATURES = GLOBAL_CLASSES = None

LAST_TRAIN_FINGERPRINT: dict[str, str] = {}
_NO_DATA_RETRAIN_COUNT = 0
_RETRAIN_COOLDOWN_UNTIL = 0.0


class _HoldModel:
    """Deterministic hold model used when training data is unavailable."""

    def __init__(self):
        self.classes_ = np.array([0, 1, 2])

    def predict(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.zeros(arr.shape[0], dtype=int)

    def predict_proba(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        probs = np.zeros((arr.shape[0], 3), dtype=float)
        probs[:, 0] = 1.0
        return probs


def _set_hold_fallback() -> None:
    """Populate global model state with a safe hold-only predictor."""

    global GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES
    GLOBAL_MODEL = _HoldModel()
    scaler = SimpleScaler().fit(np.zeros((1, 1), dtype=float))
    GLOBAL_SCALER = scaler
    GLOBAL_FEATURES = ["fallback_bias"]
    GLOBAL_CLASSES = np.array([0, 1, 2])


def _maybe_retrain_global() -> None:
    try:
        load_global_bundle()
    except Exception:
        pass


def ensure_model_loaded(adapter, symbols):
    global GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES
    global _NO_DATA_RETRAIN_COUNT, _RETRAIN_COOLDOWN_UNTIL
    if GLOBAL_MODEL is not None and GLOBAL_SCALER is not None:
        if not hasattr(GLOBAL_MODEL, "predict_proba"):
            logging.debug(
                "ensure_model_loaded | existing model lacks predict_proba; relying on fallback predict"
            )
        if GLOBAL_FEATURES is None:
            GLOBAL_FEATURES = []
        if GLOBAL_CLASSES is None:
            GLOBAL_CLASSES = np.array([0, 1, 2])
        return
    now = time.time()
    if now < _RETRAIN_COOLDOWN_UNTIL:
        return
    try:
        GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = load_global_bundle()
        _NO_DATA_RETRAIN_COUNT = 0
        _RETRAIN_COOLDOWN_UNTIL = 0.0
    except FileNotFoundError:
        logging.info(
            "ensure_model_loaded | global_model.joblib missing; retraining once"
        )
        try:
            df_features, df_target, feature_cols = fetch_and_prepare_training_data(
                adapter, symbols
            )
            GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = _retrain_checked(
                df_features, df_target, feature_cols
            )
            _NO_DATA_RETRAIN_COUNT = 0
            _RETRAIN_COOLDOWN_UNTIL = 0.0
        except ValueError as e:
            logging.error("ensure_model_loaded | retrain failed: %s", e)
            _set_hold_fallback()
            _NO_DATA_RETRAIN_COUNT += 1
            GLOBAL_MODEL = GLOBAL_MODEL  # explicitly set by fallback
        except Exception as e:
            logging.error("ensure_model_loaded | retrain failed: %s", e)
            _set_hold_fallback()
            _NO_DATA_RETRAIN_COUNT += 1
    except Exception:
        try:
            df_features, df_target, feature_cols = fetch_and_prepare_training_data(
                adapter, symbols
            )
            GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = _retrain_checked(
                df_features, df_target, feature_cols
            )
            _NO_DATA_RETRAIN_COUNT = 0
            _RETRAIN_COOLDOWN_UNTIL = 0.0
        except ValueError as e:
            if "no training data collected" in str(e):
                _NO_DATA_RETRAIN_COUNT += 1
                logging.warning(
                    "ensure_model_loaded | retrain failed (%d): %s",
                    _NO_DATA_RETRAIN_COUNT,
                    e,
                )
                _set_hold_fallback()
                if _NO_DATA_RETRAIN_COUNT >= 3:
                    logging.warning(
                        "ensure_model_loaded | no training data after %d attempts; waiting for next window",
                        _NO_DATA_RETRAIN_COUNT,
                    )
                    _RETRAIN_COOLDOWN_UNTIL = now + 900
            else:
                logging.warning("ensure_model_loaded | retrain failed: %s", e)
                _set_hold_fallback()
        except Exception as e:
            logging.warning("ensure_model_loaded | retrain failed: %s", e)
            _set_hold_fallback()


def safe_f1_macro(y_true, y_pred):
    """Return macro F1 score handling probability predictions."""
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="macro")


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional engineered features used by the model."""
    # Процентное изменение цены за последние 5 минут
    if "close_5m" in df.columns:
        df["price_change_5m"] = df["close_5m"].pct_change(fill_method=None)
    else:
        df["price_change_5m"] = 0.0
    # Отношение объема к среднему за час (12 свечей по 5m)
    if "volume_5m" in df.columns:
        df["volume_ratio"] = df["volume_5m"] / (
            df["volume_5m"].rolling(12).mean() + 1e-8
        )
    else:
        df["volume_ratio"] = 0.0
    # Наклон EMA на 15m таймфрейме
    if "close_15m" in df.columns:
        ema = df["close_15m"].ewm(span=20).mean()
        df["ema_slope"] = ema.diff()
    else:
        df["ema_slope"] = 0.0
    # Волатильность за 30 минут
    if "close_5m" in df.columns:
        df["volatility_30m"] = (
            df["close_5m"].pct_change(fill_method=None).rolling(6).std()
        )
    else:
        df["volatility_30m"] = 0.0
    # Простейшая метрика силы символа: различие RSI между 15m и 1h
    if "rsi_15m" in df.columns and "rsi_1h" in df.columns:
        df["symbol_strength"] = df["rsi_15m"] - df["rsi_1h"]
    else:
        df["symbol_strength"] = 0.0
    return df


def prepare_profit_dataset(symbol: str, profit_csv: str = "profit_report.csv") -> pd.DataFrame:
    """Load historical trades and build a training set from profitable ones."""
    path = Path(profit_csv)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        trades = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in trades.columns:
        trades = trades[trades["symbol"] == symbol]
    if trades.empty:
        return pd.DataFrame()
    # Фильтруем плохие сделки
    trades = trades[(trades["pnl_pct"] > -3) & (trades["duration"] >= 120)]
    if trades.empty:
        return pd.DataFrame()
    # Определяем три класса: прибыльная сделка, убыточная и нейтральная
    trades["target"] = np.select(
        [trades["pnl_pct"] > 0.5, trades["pnl_pct"] < -0.5],
        [1, 2],
        default=0,
    )
    ohlcv = fetch_multi_ohlcv(symbol, timeframes, limit=5000, warn=False)
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, row in trades.iterrows():
        ts = pd.to_datetime(row.get("timestamp") or row.get("entry_time"))
        if pd.isna(ts):
            continue
        snap = ohlcv[ohlcv["timestamp_15m"] <= ts].tail(1)
        if snap.empty:
            continue
        snap = add_custom_features(snap.copy())
        snap["target"] = row["target"]
        rows.append(snap.iloc[0])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.dropna(inplace=True)
    return df


def train_optuna_model(symbol: str, n_trials: int = 20):
    """Train a single global model using historical trade outcomes."""
    logging.info(f"model | training for {symbol}…")

    df = prepare_profit_dataset(symbol)
    if df.empty:
        df = pd.DataFrame()

    # Build price-based dataset if profit dataset is missing or lacks all classes
    if df.empty or not all(cls in df.get("target", pd.Series()).unique() for cls in [0, 1, 2]):
        horizon = PRED_HORIZON
        threshold = THRESHOLD
        limit_val = 5000
        for _ in range(5):
            df = fetch_multi_ohlcv(symbol, timeframes, limit=limit_val, warn=False)
            if df is None or df.empty:
                logging.error(f"model | {symbol} | Missing required timeframe data")
                return None
            for lag in range(1, 7):
                df[f"close_lag{lag}"] = df["close_15m"].shift(lag)
                df[f"volume_lag{lag}"] = df["volume_15m"].shift(lag)
            df["delta"] = df["close_15m"].pct_change(
                periods=horizon, fill_method=None
            ).shift(-horizon)
            df["target"] = np.select(
                [df["delta"] > threshold, df["delta"] < -threshold],
                [1, 2],
                default=0,
            )
            df = add_custom_features(df)
            df.dropna(inplace=True)
            class_counts = df["target"].value_counts()
            if all(class_counts.get(c, 0) >= 30 for c in [0, 1, 2]):
                break
            if limit_val < 10000:
                limit_val *= 2
            else:
                threshold *= 0.8
        if not all(class_counts.get(c, 0) >= 30 for c in [0, 1, 2]):
            if all(class_counts.get(c, 0) > 0 for c in [0, 1, 2]):
                logging.warning(
                    f"model | {symbol} | proceeding with imbalanced classes {class_counts.to_dict()}"
                )
            else:
                missing = [c for c in [0, 1, 2] if class_counts.get(c, 0) == 0]
                logging.warning(
                    f"model | {symbol} | adding synthetic samples for classes {missing}"
                )
                for cls in missing:
                    df = pd.concat([df, df.iloc[:2].assign(target=cls)], ignore_index=True)
                class_counts = df["target"].value_counts()
    else:
        # ensure lag features exist for profit-based dataset
        for lag in range(1, 7):
            if "close_15m" in df.columns:
                df[f"close_lag{lag}"] = df["close_15m"].shift(lag)
            if "volume_15m" in df.columns:
                df[f"volume_lag{lag}"] = df["volume_15m"].shift(lag)
        df.dropna(inplace=True)
        class_counts = df["target"].value_counts()
        if not all(class_counts.get(c, 0) >= 30 for c in [0, 1, 2]):
            logging.error(
                f"model | {symbol} | profit dataset lacks samples for all classes"
            )
            return None

    if len(df) > 3000:
        df = df.tail(3000)
    if df.empty:
        logging.error(f"model | {symbol} | no data for training")
        return None

    ts_cols = [c for c in df.columns if c.startswith("timestamp")]
    if ts_cols:
        fingerprint = f"{df[ts_cols[0]].iloc[0]}_{df[ts_cols[0]].iloc[-1]}_{len(df)}"
    else:
        fingerprint = f"{df.index[0]}_{df.index[-1]}_{len(df)}"
    model_path = str(BUNDLE_PATH)
    if os.path.exists(model_path) and LAST_TRAIN_FINGERPRINT.get(symbol) == fingerprint:
        logging.info(f"model | {symbol} | data window unchanged, skipping training")
        return GLOBAL_MODEL
    LAST_TRAIN_FINGERPRINT[symbol] = fingerprint

    feature_cols = [
        c
        for c in df.columns
        if not c.startswith("timestamp") and c not in ["target", "delta"]
    ]
    # [ANCHOR:FEATURE_SHAPE_GUARD]
    df_features = df[feature_cols].fillna(0).copy()
    if df_features.shape[1] != len(feature_cols):
        logging.warning("feature | shape_mismatch")
        return None
    df_target = df["target"].astype(int)
    if len(np.unique(df_target)) < 2:
        logging.error(f"model | {symbol} | only one class present; aborting")
        return None

    try:
        model, scaler, features, classes = retrain_global_model(
            df_features, df_target, feature_cols
        )
    except Exception as exc:
        logging.error("model | %s | retrain failed: %s", symbol, exc)
        return None

    try:
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "features": list(features),
                "classes": classes,
            },
            BUNDLE_PATH,
        )
    except Exception as exc:
        logging.warning("model | %s | bundle dump failed: %s", symbol, exc)

    globals()["GLOBAL_MODEL"] = model
    globals()["GLOBAL_SCALER"] = scaler
    globals()["GLOBAL_FEATURES"] = list(features)
    globals()["GLOBAL_CLASSES"] = classes
    logging.info(
        "model | %s | ✅ model retrained (%d features)",
        symbol,
        len(features),
    )
    return model


def train_model(symbol):
    """Backward compatibility wrapper calling Optuna-based training."""
    return train_optuna_model(symbol)


def predict_signal(
    symbol: str,
    X_last: pd.DataFrame,
    adx: float,
    rsi_cross_from_extreme: bool,
    returns_1h: float,
) -> tuple[str, float]:
    model = GLOBAL_MODEL
    scaler = GLOBAL_SCALER
    features = list(GLOBAL_FEATURES or [])
    ts = datetime.now(timezone.utc).isoformat()

    if not model or not scaler or not features:
        log(logging.WARNING, "predict", symbol, "model/scaler/features missing; fallback")
        append_csv(
            "decision_log.csv",
            {"timestamp": ts, "symbol": symbol, "signal": "hold", "reason": "missing_model"},
            ["timestamp", "symbol", "signal", "reason"],
        )
        return "hold", 0.0

    X = X_last.reindex(columns=features, fill_value=0.0)
    if X.shape[1] != len(features):
        log(logging.ERROR, "predict", symbol, f"feature mismatch {X.shape[1]} != {len(features)}; fallback")
        append_csv(
            "decision_log.csv",
            {"timestamp": ts, "symbol": symbol, "signal": "hold", "reason": "feature_mismatch"},
            ["timestamp", "symbol", "signal", "reason"],
        )
        return "hold", 0.0
    Xs = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
    elif hasattr(model, "predict") and getattr(model, "objective", "") == "multi:softprob":
        raw = model.predict(Xs)
        proba = np.asarray(raw).reshape(1, -1)
    else:
        log(logging.ERROR, "predict", symbol, "model lacks predict_proba; fallback")
        append_csv(
            "decision_log.csv",
            {"timestamp": ts, "symbol": symbol, "signal": "hold", "reason": "no_predict_proba"},
            ["timestamp", "symbol", "signal", "reason"],
        )
        return "hold", 0.0

    classes = GLOBAL_CLASSES
    if classes is None or len(classes) < 3:
        log(logging.ERROR, "predict", symbol, f"global classes invalid {classes}; fallback")
        append_csv(
            "decision_log.csv",
            {"timestamp": ts, "symbol": symbol, "signal": "hold", "reason": "bad_classes"},
            ["timestamp", "symbol", "signal", "reason"],
        )
        return "hold", 0.0

    i_hold = int(np.where(classes == 0)[0][0])
    i_long = int(np.where(classes == 1)[0][0])
    i_short = int(np.where(classes == 2)[0][0])

    p_hold = float(proba[0, i_hold])
    p_long = float(proba[0, i_long])
    p_short = float(proba[0, i_short])
    conf = max(p_long, p_short)

    proba_filter_adj = PROBA_FILTER
    if adx >= 18 or abs(returns_1h) > 0.01:
        proba_filter_adj = max(0.35, PROBA_FILTER - 0.10)
    if conf < proba_filter_adj and adx >= 20 and rsi_cross_from_extreme:
        conf += 0.05

    if conf < proba_filter_adj:
        append_csv(
            "decision_log.csv",
            {"timestamp": ts, "symbol": symbol, "signal": "hold", "reason": "low_conf"},
            ["timestamp", "symbol", "signal", "reason"],
        )
        return "hold", conf

    signal = "long" if p_long >= p_short else "short"
    append_csv(
        "decision_log.csv",
        {"timestamp": ts, "symbol": symbol, "signal": signal, "reason": "model"},
        ["timestamp", "symbol", "signal", "reason"],
    )
    return signal, conf


def backtest(symbol, tf="15m", horizon: int | None = None):
    """Run a simple backtest and return metrics.

    The returned dictionary contains ``return`` percentage, ``sharpe`` ratio,
    ``max_drawdown`` and the ``equity_curve`` used for the calculations.  If no
    data is available the function returns an empty dict and logs the reason.

    If ``horizon`` is not provided, the global ``PRED_HORIZON`` value is used to
    ensure tuned parameters via ``apply_params`` take effect.
    """
    global GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES
    if horizon is None:
        horizon = PRED_HORIZON
    report_path = os.path.join(os.path.dirname(__file__), "profit_report.csv")
    model = GLOBAL_MODEL
    scaler = GLOBAL_SCALER
    if model is None or scaler is None:
        _maybe_retrain_global()
        model = GLOBAL_MODEL
        scaler = GLOBAL_SCALER
    if (
        model is None
        or scaler is None
        or not hasattr(model, "classes_")
        or len(getattr(model, "classes_", [])) < 3
    ):
        msg = "global model unavailable"
        log(logging.ERROR, "backtest", symbol, msg)
        raise RuntimeError(msg)
    limit_val = 300
    df = fetch_multi_ohlcv(symbol, timeframes, limit=limit_val)
    if df is None:
        log_candle_status(symbol, tf, None)
        logging.warning("backtest | %s | data unavailable", symbol)
        return {"mode": "unavailable"}
    reduced = bool(df.attrs.get("reduced", False))
    log_candle_status(symbol, tf, len(df))
    if df.empty:
        logging.warning("backtest | %s | data unavailable", symbol)
        return {"mode": "unavailable"}
    if reduced and "close_15m" in df.columns:
        returns = df["close_15m"].pct_change(fill_method=None).fillna(0)
        equity = (1 + returns).cumprod()
        metrics = backtest_metrics(equity)
        metrics["data_mode"] = "reduced"
        total_return = equity.iloc[-1] - 1 if not equity.empty else 0.0
        return {
            "return": float(total_return),
            "equity_curve": equity.tolist(),
        } | metrics
    if len(df) < 50:
        reduced = True
        log(
            logging.INFO,
            "backtest",
            symbol,
            f"rows={len(df)} < 50, running reduced backtest",
        )
    # === ДОБАВЛЯЕМ LAG-ФИЧИ, как в train_model ===
    for lag in range(1, 7):
        df[f"close_lag{lag}"] = df["close_15m"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume_15m"].shift(lag)

    feature_cols = list(GLOBAL_FEATURES or [])
    if not feature_cols:
        log(logging.ERROR, "backtest", symbol, "GLOBAL_FEATURES missing")
        return {"mode": "fallback"}
    df["pattern_name"] = -1
    df["pattern_source"] = 0
    df["pattern_confidence"] = 0.0
    df_features = df.reindex(columns=feature_cols, fill_value=0.0)
    log(
        logging.INFO,
        "backtest",
        symbol,
        f"X.shape={df_features.shape}, features={feature_cols}, rows={len(df_features)}",
    )
    if df_features.empty:
        log(
            logging.WARNING,
            "backtest",
            symbol,
            f"X.shape={df_features.shape}, backtest skipped due to no data",
        )
        return {"mode": "unavailable"}
    if df_features.shape[0] < 20:
        reduced = True
        log(
            logging.INFO,
            "backtest",
            symbol,
            f"X.shape={df_features.shape}, running reduced backtest",
        )

    if df_features.shape[1] != len(feature_cols):
        log(
            logging.ERROR,
            "backtest",
            symbol,
            f"feature mismatch {df_features.shape[1]} != {len(feature_cols)}",
        )
        return {"mode": "fallback"}
    try:
        if scaler is not None:
            X_arr = scaler.transform(df_features)
            X_bt = pd.DataFrame(X_arr, columns=feature_cols, index=df_features.index)
        else:
            X_bt = df_features.copy()
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_bt)
            if (
                proba is None
                or len(proba) == 0
                or len(proba) != len(X_bt)
            ):
                log_prediction_error(
                    "backtest", symbol, len(X_bt), 0 if proba is None else len(proba)
                )
                fallback_signal(symbol)
                return {"mode": "fallback"}
            pred = np.argmax(proba, axis=1)
            maxp = np.max(proba, axis=1)
            if float(maxp[-20:].mean()) == 0.0:
                log(logging.ERROR, "backtest", symbol, "mean proba 0; reloading model")
                try:
                    GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = load_global_bundle()
                except Exception:
                    pass
                try:
                    df_features, df_target, feature_cols = fetch_and_prepare_training_data(
                        ADAPTER, symbols
                    )
                    GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = _retrain_checked(
                        df_features, df_target, feature_cols
                    )
                except Exception as e:
                    logging.error("backtest | %s | retrain failed: %s", symbol, e)
        else:
            pred = model.predict(X_bt)
            if pred is None or len(pred) == 0 or len(pred) != len(X_bt):
                log_prediction_error(
                    "backtest", symbol, len(X_bt), len(pred) if pred is not None else None
                )
                fallback_signal(symbol)
                return {"mode": "fallback"}
    except Exception as e:
        log(logging.ERROR, "backtest", symbol, f"model error: {e}")
        fallback_signal(symbol)
        return {"mode": "fallback"}
    df["pred"] = pred
    df["returns"] = df["close_15m"].pct_change(periods=horizon, fill_method=None).shift(-horizon)
    df["strategy"] = np.where(df["pred"] == 1, df["returns"], np.where(df["pred"] == 2, -df["returns"], 0))
    df["equity"] = (1 + df["strategy"].fillna(0)).cumprod()
    total_return = df["equity"].iloc[-1] - 1
    record_backtest(symbol, total_return)
    metrics = backtest_metrics(df["equity"])
    if reduced:
        metrics["data_mode"] = "reduced"
    sharpe = metrics.get("sharpe")
    if reduced:
        return_str = "NA (reduced)"
        sharpe_str = "NA"
    else:
        return_str = f"{total_return:.2%}"
        sharpe_str = "NA" if np.isnan(sharpe) else f"{sharpe:.2f}"
    logging.info(
        f"backtest | {symbol} | Return: {return_str}, Sharpe={sharpe_str}, DD={metrics.get('max_drawdown', float('nan')):.2%}"
    )
    header = not os.path.exists(report_path)
    if not header:
        prev = pd.read_csv(report_path)
        if "return" in prev.columns:
            cumulative = (prev["return"] + 1).prod() - 1
        else:
            cumulative = 0.0
    else:
        cumulative = 0.0
    row = {
        "timestamp": datetime.now(timezone.utc),
        "symbol": symbol,
        "return": total_return,
        "cumulative_return": cumulative,
    } | metrics
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=header, index=False)
    metrics_with_equity = {
        "return": float(total_return),
        "equity_curve": df["equity"].tolist(),
    } | metrics
    return metrics_with_equity


def log_trade(
    timestamp: datetime,
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    volume: float,
    profit: float,
    exit_type: str,
    log_path: str | None = None,
) -> None:
    """Append closed trade information to ``trades_log.csv``.

    # [ANCHOR:LOG_SCHEMA_ENFORCE]
    The file is created with headers if it does not exist or is empty.
    """
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), "trades_log.csv")

    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    ctx = open_trade_ctx.pop(symbol, {})
    entry_time_str = ctx.get("entry_time")
    if entry_time_str:
        entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
    else:
        entry_dt = timestamp
        entry_time_str = timestamp.isoformat().replace("+00:00", "Z")
    duration = (timestamp - entry_dt).total_seconds() / 60
    profit_pct = (profit / (entry_price * volume)) * 100 if entry_price * volume else 0.0

    row = {
        "trade_id": ctx.get("trade_id", str(uuid4())),
        "timestamp_close": timestamp.isoformat().replace("+00:00", "Z"),
        "symbol": symbol,
        "side": side.upper(),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "qty": float(volume),
        "profit": float(profit),
        "profit_pct": float(profit_pct),
        "exit_type": exit_type,
        "entry_time": entry_time_str,
        "duration_min": float(duration),
        "stop_loss_triggered": exit_type == "STOP_MARKET",
        "take_profit_triggered": exit_type == "TP",
        "trailing_profit_used": exit_type == "TRAIL_STOP",
        "source": ctx.get("source", "live"),
        "reduced_risk": bool(ctx.get("reduced_risk", False)),
        "order_id": ctx.get("order_id", ""),
    }

    df = pd.DataFrame([row], columns=logging_utils.LOG_EXIT_FIELDS)
    df.to_csv(log_path, mode="a", header=write_header, index=False)
    symbol_activity[symbol] = timestamp

    if ctx:
        ctx |= {
            "exit_price": float(exit_price),
            "profit": float(profit),
            "exit_type": exit_type,
        }
        memory_manager.add_trade_close(ctx)


def register_trade_result(symbol: str, profit: float, log_path: str) -> None:
    """Record trade outcome and persist updated risk state."""

    global risk_state
    stats.register(symbol, profit)
    limiter.register(symbol, profit)
    bar_index = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
    if profit < 0:
        cool.register_loss(symbol, bar_index)
    # [ANCHOR:SAVE_RISK_STATE_AFTER_TRADE]
    try:
        save_risk_state(risk_state, limiter, cool, stats)
    except Exception as e:
        logging.exception("save_risk_state failed: %s", e)

    lookback = risk_config.get("lookback_trades", 10)
    if stats.trades(symbol) % lookback == 0:
        updated = update_pair_stats(log_path, lookback)
        risk_state = adjust_state_by_stats(risk_state, updated, risk_config)
        save_pair_report(updated)
        save_risk_state(risk_state, limiter, cool, stats)


## log_exit_from_order imported from logging_utils


# [ANCHOR:EXTRACT_COMMISSION]
def _extract_commission(order) -> float:
    try:
        fee = order.get("fee") or order.get("info", {}).get("fee") or order.get("info", {}).get("cum_fee")
        if isinstance(fee, dict):
            return float(fee.get("cost") or fee.get("value") or 0.0)
        if fee is not None:
            return float(fee)
    except Exception:
        pass
    return 0.0


# [ANCHOR:NORMALIZE_PREV_SIDE]
def _normalize_prev_side(order) -> str | None:
    """
    Возвращает 'LONG' или 'SHORT' из метаданных ордера.
    Поддерживает варианты: LONG/SHORT, BUY/SELL (переводит в LONG/SHORT).
    Если определить невозможно — вернёт None.
    """
    raw = None
    info = order.get("info", {}) or {}
    # приоритет: prev_side из info, затем side из order/info
    raw = info.get("prev_side") or order.get("side") or info.get("side")
    if not raw:
        return None
    s = str(raw).strip().upper()
    if s in ("LONG", "SHORT"):
        return s
    if s == "BUY":
        return "LONG"
    if s == "SELL":
        return "SHORT"
    return None


def apply_params(best: dict | StrategyParams | None):
    """Safely apply strategy parameters.

    Значения берутся из ``best`` c откатом к ``DEFAULT_PARAMS`` при отсутствии,
    ``None`` или ``NaN``. Параметры помещаются в глобальное пространство имён
    модуля, чтобы сохранить обратную совместимость с существующим кодом.
    """

    if isinstance(best, StrategyParams):
        p = asdict(best)
    else:
        p = dict(best or {})
    p = normalize_param_keys(p)
    for k, v in DEFAULT_PARAMS.items():
        globals()[k] = (
            v
            if (
                k not in p
                or p[k] is None
                or (isinstance(p[k], float) and p[k] != p[k])
            )
            else p[k]
        )
    param_str = " ".join(f"{k}={globals()[k]}" for k in DEFAULT_PARAMS)
    log(logging.INFO, "__main__", "params", "applied", param_str)


def select_trade_mode(symbol: str, df_trend: pd.DataFrame) -> tuple[str, dict, str]:
    """Classify market regime and derive trade parameters.

    Returns ``(mode, params, data_mode)`` where ``params`` holds ATR-based
    multipliers for SL/TP and leverage. ``data_mode`` is ``'reduced'`` when
    lower timeframes had to be upsampled.
    """

    df_5m = fetch_ohlcv(symbol, "5m", limit=50)
    data_mode = "normal"
    if df_5m is None or df_5m.empty:
        df_15m = fetch_ohlcv(symbol, "15m", limit=50)
        if df_15m is not None and not df_15m.empty:
            df_5m = resample_ohlcv(df_15m, "5min")
            data_mode = "reduced"
        else:
            df_1h = fetch_ohlcv(symbol, "1h", limit=50)
            if df_1h is not None and not df_1h.empty:
                df_5m = resample_ohlcv(df_1h, "5min")
                data_mode = "reduced"

    if "atr" in df_trend.columns:
        atr = float(df_trend["atr"].iloc[-1])
    else:
        atr = 0.0
    close = float(df_trend["close"].iloc[-1]) if not df_trend.empty else 1.0
    vol_series = df_trend["volume"] if "volume" in df_trend.columns else None
    vol_ratio = safe_vol_ratio(vol_series, VOL_WINDOW, key=symbol) or 0.0
    atr_ratio = atr / close if close else 0.0

    params: dict
    if atr_ratio >= 0.01 and vol_ratio >= 1.2:
        mode = "scalp"
        params = {
            "sl_mult": 2.0,
            "tp_mult": 4.0,
            "lev": 40,
            "horizon": 6,
            "trailing_start": 0.004,
            "partial_tp": 0.3,
            "partial_tp_mult": 1.0,
        }
    elif atr_ratio < 0.01:
        mode = "swing"
        params = {
            "sl_mult": 2.5,
            "tp_mult": 5.0,
            "lev": 10,
            "horizon": 25,
            "partial_tp": 0.5,
            "partial_tp_mult": 1.2,
        }
    else:
        mode = "intraday"
        params = {
            "sl_mult": 2.2,
            "tp_mult": 4.5,
            "lev": 20,
            "horizon": 10,
            "partial_tp": 0.4,
            "partial_tp_mult": 1.1,
        }

    logging.info(
        f"mode | {symbol} | selected {mode.upper()} atr={atr_ratio:.4f} vol_ratio={vol_ratio:.2f}"
    )
    return mode, params, data_mode


def best_entry_moment(
    symbol: str,
    signal: str,
    timeframe: str = "15m",
    limit: int = 250,
    *,
    source: str = "model",
    mode: str = "SWING",
    confidence: float | None = None,
    adx: float | None = None,
    trend_ok: bool = False,
) -> bool:
    """Return ``True`` if short-term indicators confirm entry direction."""
    relaxed = (
        confidence is not None
        and adx is not None
        and confidence > PROBA_FILTER + 0.05
        and adx >= ADX_THRESHOLD
    )
    lookback = 3 if relaxed else 1
    price_tol = 0.005 if relaxed else 0.0025
    df = fetch_ohlcv(symbol, timeframe, limit=limit)
    if df is None or df.empty or len(df) < lookback + 1:
        fallback = fetch_ohlcv(symbol, "5m", limit=max(limit * 3, 10))
        if fallback is not None and len(fallback) >= 3:
            df = resample_ohlcv(fallback, "15min")
        else:
            log(
                logging.WARNING,
                "entry",
                symbol,
                f"not enough candles for {timeframe} confirmation - proceeding",
            )
            return True
    ema_fast = df["close"].ewm(span=9).mean()
    ema_slow = df["close"].ewm(span=21).mean()
    rsi = df["rsi"]
    vol_ratio = safe_vol_ratio(df["volume"], VOL_WINDOW, key=symbol)
    if len(df) < lookback + 1:
        log(
            logging.WARNING,
            "entry",
            symbol,
            f"not enough candles for {timeframe} confirmation - proceeding",
        )
        return True
    if vol_ratio is not None:
        thr = 0.8 if source == "fallback" else 1.0
        if vol_ratio < thr:
            return False
        if source == "fallback" and trend_ok and vol_ratio >= 0.8:
            log(logging.INFO, "entry_timing_relaxed", symbol, "fallback volume ok")
            return True

    if source == "fallback" or mode in {"SCALP", "INTRADAY"}:
        if signal == "long":
            cross = any(
                ema_fast.iloc[-i - 2] <= ema_slow.iloc[-i - 2]
                and ema_fast.iloc[-i - 1] > ema_slow.iloc[-i - 1]
                for i in range(lookback)
            )
            retest = df["close"].iloc[-1] >= ema_fast.iloc[-1] and (
                (df["close"].iloc[-1] - ema_fast.iloc[-1]) / ema_fast.iloc[-1] <= price_tol
            )
            ok = cross or retest
        else:
            cross = any(
                ema_fast.iloc[-i - 2] >= ema_slow.iloc[-i - 2]
                and ema_fast.iloc[-i - 1] < ema_slow.iloc[-i - 1]
                for i in range(lookback)
            )
            retest = df["close"].iloc[-1] <= ema_fast.iloc[-1] and (
                (ema_fast.iloc[-1] - df["close"].iloc[-1]) / ema_fast.iloc[-1] <= price_tol
            )
            ok = cross or retest
    else:
        if signal == "long":
            cross = any(
                ema_fast.iloc[-i - 2] <= ema_slow.iloc[-i - 2]
                and ema_fast.iloc[-i - 1] > ema_slow.iloc[-i - 1]
                for i in range(lookback)
            )
            ok = cross and rsi.iloc[-1] > 50
        else:
            cross = any(
                ema_fast.iloc[-i - 2] >= ema_slow.iloc[-i - 2]
                and ema_fast.iloc[-i - 1] < ema_slow.iloc[-i - 1]
                for i in range(lookback)
            )
            ok = cross and rsi.iloc[-1] < 50
    if ok and relaxed:
        log(
            logging.INFO,
            "entry_timing_relaxed",
            symbol,
            f"confidence={confidence:.2f} adx={adx:.1f}",
        )
    return ok
def fetch_positions_soft(symbol: str) -> list[dict]:
    """Fetch positions while gracefully handling missing category support."""

    cat = detect_market_category(exchange, symbol) or "linear"
    cat = str(cat or "").lower()
    if not cat or cat == "swap":
        cat = "linear"
    if cat == "spot":
        cat = "linear"
    norm = _normalize_bybit_symbol(exchange, symbol, cat)
    try:
        return exchange.fetch_positions([norm], {"category": cat})
    except Exception as exc:
        log(
            logging.WARNING,
            "positions",
            symbol,
            f"fallback to generic fetch_positions: {exc}",
        )
    try:
        return exchange.fetch_positions([norm])
    except Exception as exc:
        record_error(symbol, f"positions fetch failed: {exc}")
        logging.error(f"Failed to fetch positions for {symbol}: {exc}")
        return []


def _max_affordable_amount(
    exchange,
    symbol,
    side,
    leverage: int,
    price: float,
    min_notional: float = 10.0,
) -> float:
    """Return maximum position size affordable with current free balance."""

    try:
        bal = safe_fetch_balance(exchange, {"type": "future"})
    except Exception:
        return 0.0

    free_raw = 0.0
    if isinstance(bal, dict):
        account = bal.get("USDT")
        if isinstance(account, dict):
            free_raw = float(account.get("free", 0.0) or 0.0)
        if free_raw <= 0:
            free_raw = float(bal.get("free", 0.0) or 0.0)
    try:
        free = float(free_raw)
    except Exception:
        free = 0.0
    if free <= 0:
        return 0.0

    max_notional = free * float(leverage) * 0.95
    if max_notional < min_notional:
        return 0.0

    safe_price = max(float(price or 0.0), 1e-12)
    max_amount = max_notional / safe_price
    _, max_amount = _price_qty_to_precision(exchange, symbol, price=None, amount=max_amount)
    try:
        return float(max_amount)
    except Exception:
        return 0.0


def _adjust_qty_for_margin(
    exchange,
    symbol: str,
    qty: float,
    price: float,
    leverage: float,
    available_margin: float,
    min_qty: float,
) -> tuple[float | None, str | None]:
    """Return quantity adjusted for available margin or ``None`` if impossible."""

    effective_leverage = max(float(leverage) or 0.0, 1.0)
    adjusted_qty = float(exchange.amount_to_precision(symbol, qty))
    attempts = 0
    available = max(float(available_margin or 0.0), 0.0)

    while adjusted_qty > 0:
        required_margin = (adjusted_qty * price) / effective_leverage if price > 0 else float("inf")
        if required_margin <= available:
            if adjusted_qty * price < 10:
                return None, "below_min_notional"
            return adjusted_qty, None
        if attempts >= 2 or adjusted_qty <= max(min_qty, 0.0):
            break
        attempts += 1
        new_qty = max(min_qty, adjusted_qty / 2)
        new_qty = float(exchange.amount_to_precision(symbol, new_qty))
        if new_qty == adjusted_qty:
            break
        log(
            logging.WARNING,
            "order",
            symbol,
            f"reduced qty to {new_qty} due to insufficient balance (needs {required_margin:.4f} > has {available:.4f})",
        )
        adjusted_qty = new_qty

    required_margin = (adjusted_qty * price) / effective_leverage if price > 0 else float("inf")
    if required_margin <= available:
        if adjusted_qty * price < 10:
            return None, "below_min_notional"
        return adjusted_qty, None
    return None, "insufficient_balance"


def _compute_entry_qty(
    symbol: str,
    side: str,
    price: float,
    leverage: int | float,
    balance: float,
    available_margin: float,
    risk_factor: float,
) -> float:
    """Return order size limited by configured risk and available margin."""

    if price <= 0:
        return 0.0

    qty = risk_management.compute_order_qty(
        ADAPTER,
        symbol,
        side,
        balance,
        RISK_PER_TRADE * risk_factor,
        price=price,
    )
    if qty is None or qty <= 0:
        return 0.0

    try:
        lev_int = int(float(leverage))
    except Exception:
        lev_int = 1
    lev_int = max(lev_int, 1)

    effective_margin = available_margin if available_margin and available_margin > 0 else balance
    if effective_margin <= 0:
        return 0.0

    qty_balance = (effective_margin * lev_int) / price
    max_notional = balance * MAX_POSITION_PCT if balance > 0 else 0.0
    if max_notional > 0:
        qty_balance = min(qty_balance, max_notional / price)

    if qty_balance > 0:
        qty = min(qty, qty_balance)

    return max(float(qty), 0.0)


def run_trade(
    symbol: str,
    signal: str,
    df_trend=None,
    stats=None,
    state=None,
    entry_ctx: Dict[str, Any] | None = None,
    *,
    sl_mult: float = 2.0,
    tp_mult: float = 4.0,
    trailing_start: float | None = None,
    partial_tp: float | None = None,
    partial_tp_mult: float | None = None,
    leverage: int | None = None,
    risk_factor: float = 1.0,
) -> bool:
    """Execute a trade with SL/TP and optional trailing stop."""
    if df_trend is None:
        df_trend = fetch_ohlcv(symbol, "1h", limit=250)
    if stats is None:
        stats = {}
    if state is None:
        state = risk_state.get(symbol, PairState())

    price: float | None = None
    ticker: dict | None = None
    try:
        ticker = exchange.fetch_ticker(symbol)
    except Exception as exc:
        log_once("warning", f"trade | {symbol} | fetch_ticker failed: {exc}")
        ticker = None
    price_candidates: list[float | None] = []
    if isinstance(ticker, dict):
        price_candidates.extend(
            [
                ticker.get("last"),
                ticker.get("close"),
                ticker.get("ask"),
                ticker.get("bid"),
            ]
        )
    for candidate in price_candidates:
        if candidate:
            try:
                price = float(candidate)
            except (TypeError, ValueError):
                continue
            if price > 0:
                break
            price = None
    if price is None or price <= 0:
        try:
            price = float(df_trend["close"].iloc[-1])
        except Exception:
            price = None
    if price is None or price <= 0:
        log(logging.ERROR, "trade", symbol, "unable to determine valid price; skipping order")
        log_decision(symbol, "price_unavailable")
        return False
    detected_category = detect_market_category(exchange, symbol)
    category = str(detected_category or "").lower()
    if category == "swap":
        category = "linear"
    if category != "linear":
        log_decision(
            symbol,
            "no_futures_contract",
            detail=(
                f"entry | {symbol} | skip: unsupported market category "
                f"{category or 'unknown'}"
            ),
        )
        return False
    want_side = "buy" if signal == "long" else "sell"
    qty_signed, qty_abs = has_open_position(exchange, symbol, category)
    if qty_abs > 0 and (
        (want_side == "buy" and qty_signed > 0)
        or (want_side == "sell" and qty_signed < 0)
    ):
        log_decision(
            symbol,
            "position_already_open",
            detail=f"entry | {symbol} | skip: position already open (qty={qty_signed:.4f})",
        )
        return False
    if has_pending_entry(exchange, symbol, want_side, category):
        log_decision(
            symbol,
            "pending_entry_exists",
            detail=f"entry | {symbol} | skip: pending entry exists ({want_side})",
        )
        return False
    now_bar5 = int(time.time() // (5 * 60))
    guard_state = _entry_guard.get(symbol) or {}
    if guard_state.get("bar") == now_bar5:
        log_decision(
            symbol,
            "entry_guard_active",
            detail=f"entry | {symbol} | skip: already entered this bar",
        )
        return False

    lev = LEVERAGE if leverage is None else leverage
    atr_val = safe_atr(df_trend["atr"] if "atr" in df_trend.columns else None, key=symbol) or 0.0
    if entry_ctx is None:
        entry_ctx = {}
    balance_info = safe_fetch_balance(exchange, {"type": "future"})
    totals = balance_info.get("total") if isinstance(balance_info, dict) else None
    frees = balance_info.get("free") if isinstance(balance_info, dict) else None
    balance = float((totals or {}).get("USDT", 0.0))
    available_margin = float((frees or totals or {}).get("USDT", 0.0))
    market = exchange.market(symbol)
    precision = market.get("precision", {}).get("amount", 0)
    precision_step = 1 / (10**precision) if precision else 1.0
    min_qty = market.get("limits", {}).get("amount", {}).get("min", 0.0) or 0.0
    price_precision = market.get("precision", {}).get("price", 0)
    tick_size = 1 / (10**price_precision) if price_precision else None
    atr_pct = atr_val / price if price else 0.0
    sl_adj = sl_mult
    tp_adj = tp_mult
    if atr_pct > 0.01:
        sl_adj *= 1.2
        tp_adj *= 1.2

    mode_params = {"sl_mult": float(sl_adj), "tp_mult": float(tp_adj)}
    tp_price_raw, sl_price_raw, sl_pct = risk_management.calc_sl_tp(
        price,
        atr_val,
        mode_params,
        "long" if signal == "long" else "short",
        tick_size=tick_size,
    )
    sl_price = float(exchange.price_to_precision(symbol, sl_price_raw))
    tp_price = float(exchange.price_to_precision(symbol, tp_price_raw))

    sizing_price = price
    if ticker is not None:
        for key in ("last", "close", "ask", "bid"):
            val = ticker.get(key) if isinstance(ticker, dict) else None
            if val is None:
                continue
            try:
                candidate = float(val)
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                sizing_price = candidate
                break

    vol_series_entry = df_trend["volume"] if "volume" in df_trend.columns else None
    vol_ratio_entry = safe_vol_ratio(vol_series_entry, VOL_WINDOW, key=f"{symbol}_entry_main")
    vol_reason_entry = volume_reason(vol_series_entry, VOLUME_RATIO_MIN, VOL_WINDOW)

    if entry_ctx is not None:
        entry_ctx["entry_vol_ratio"] = vol_ratio_entry
        entry_ctx["entry_vol_reason"] = vol_reason_entry

    effective_risk_factor = float(risk_factor)
    if vol_reason_entry == "vol_low":
        if vol_ratio_entry is not None:
            detail = (
                f"entry | {symbol} | skip: volume ratio {vol_ratio_entry:.2f} "
                f"below minimum {VOLUME_RATIO_MIN:.2f}"
            )
        else:
            detail = (
                f"entry | {symbol} | skip: volume ratio below minimum {VOLUME_RATIO_MIN:.2f}"
            )
        log_once("info", detail, window_sec=120.0)
        log_decision(symbol, "volume_low", detail=detail)
        return False
    if vol_reason_entry == "vol_missing":
        effective_risk_factor = max(0.0, risk_factor * 0.5)
        log_once(
            "info",
            f"entry | {symbol} | missing volume data; risk_factor cut to {effective_risk_factor:.2f}",
            window_sec=300.0,
        )

    symbol_norm = _normalize_bybit_symbol(ADAPTER.x, symbol, category)
    qty_target = _compute_entry_qty(
        symbol,
        want_side,
        sizing_price,
        lev,
        balance,
        available_margin,
        effective_risk_factor,
    )
    if qty_target <= 0:
        log_decision(
            symbol,
            "qty_insufficient",
            detail=f"entry | {symbol} | skip: qty_target too small",
        )
        return False

    try:
        leverage_int = int(float(lev))
    except Exception:
        leverage_int = int(LEVERAGE)
    leverage_int = max(leverage_int, 1)
    affordable_qty = _max_affordable_amount(
        exchange,
        symbol,
        want_side,
        leverage_int,
        price,
        MIN_NOTIONAL,
    )
    if affordable_qty <= 0:
        log_decision(
            symbol,
            "insufficient_balance",
            detail=f"order | {symbol} | skipped: insufficient balance for entry",
        )
        return False
    qty_target = min(qty_target, affordable_qty)

    qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
    if qty_target <= 0:
        log_decision(
            symbol,
            "qty_insufficient",
            detail=f"entry | {symbol} | skip: qty_target too small after rounding",
        )
        return False

    adjusted_qty, margin_reason = _adjust_qty_for_margin(
        exchange,
        symbol,
        qty_target,
        price,
        lev,
        available_margin,
        min_qty,
    )
    if adjusted_qty is None:
        log_decision(symbol, margin_reason or "insufficient_balance")
        return False
    qty_target = adjusted_qty

    max_pos_qty = get_max_position_qty(symbol, lev, price)
    if max_pos_qty:
        qty_target = min(qty_target, max_pos_qty)

    qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
    if qty_target <= 0:
        log_decision(
            symbol,
            "qty_insufficient",
            detail=f"entry | {symbol} | skip: qty_target too small",
        )
        return False

    logging.info(
        colorize(
            f"trade | {symbol} | Opening {want_side.upper()} position | qty={qty_target} | price={price}",
            "open",
        )
    )
    order_id = None
    try:
        filled_qty = enter_ensure_filled(
            ADAPTER.x,
            symbol,
            want_side,
            qty_target,
            category=category,
        )
    except Exception as exc:
        logging.warning("entry | %s | ensure_filled failed: %s", symbol, exc)
        log_decision(symbol, "order_failed")
        return False
    if (filled_qty or 0.0) <= 0:
        log_decision(symbol, "order_failed")
        return False
    _entry_guard[symbol] = {"bar": now_bar5, "side": want_side}

    detected_qty = wait_position_after_entry(
        ADAPTER.x, symbol, category=category, timeout_sec=3.0
    )
    if detected_qty <= 0:
        log_once(
            "warning",
            f"entry | {symbol} | filled order but position not visible yet; exits postponed",
            window_sec=60.0,
        )

    _pos_signed_after, pos_abs_after = has_open_position(exchange, symbol, category)
    entry_price = get_position_entry_price(exchange, symbol, category) or price

    want_long = signal == "long"
    try:
        sl_pct_eff = abs((sl_price / entry_price) - 1) if entry_price else SL_PCT
    except Exception:
        sl_pct_eff = SL_PCT
    if not sl_pct_eff or not math.isfinite(sl_pct_eff):
        sl_pct_eff = SL_PCT
    try:
        tp_pct_eff = abs((tp_price / entry_price) - 1) if entry_price else TP_PCT
    except Exception:
        tp_pct_eff = TP_PCT
    if not tp_pct_eff or not math.isfinite(tp_pct_eff):
        tp_pct_eff = TP_PCT

    if pos_abs_after > 0:
        try:
            _, err = place_conditional_exit(
                ADAPTER.x,
                symbol,
                "buy" if want_long else "sell",
                entry_price,
                price,
                sl_pct_eff,
                category,
                is_tp=False,
            )
            if err:
                log_once("warning", f"Failed to set SL for {symbol}: {err}")
        except RuntimeError as exc:
            log_once("warning", f"Failed to set SL for {symbol}: {exc}")
        except Exception as exc:
            log_once("warning", f"Failed to set SL for {symbol}: {exc}")

        try:
            _, err = place_conditional_exit(
                ADAPTER.x,
                symbol,
                "buy" if want_long else "sell",
                entry_price,
                price,
                tp_pct_eff,
                category,
                is_tp=True,
            )
            if err:
                log_once("warning", f"Failed to set TP for {symbol}: {err}")
        except RuntimeError as exc:
            log_once("warning", f"Failed to set TP for {symbol}: {exc}")
        except Exception as exc:
            log_once("warning", f"Failed to set TP for {symbol}: {exc}")

    qty = float(pos_abs_after or detected_qty or filled_qty)

    current_bar_index = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ctx = {
        "symbol": symbol,
        "side": signal.upper(),
        "entry_price": float(entry_price),
        "entry_time": now_iso,
        "qty": float(qty),
        "open_bar_index": current_bar_index,
        "trailing_profit_used": False,  # [ANCHOR:REMOVE_LEGACY_TRAIL]
        "order_id": order_id,
        "source": entry_ctx.get("source", "live"),
        "reduced_risk": bool(entry_ctx.get("reduced_risk", False)),
        "atr": float(atr_val),
        "tick_size": float(tick_size or 0.0),
        "sl_price": float(sl_price) if sl_price is not None else None,
        "tp_price": float(tp_price) if tp_price is not None else None,
        **{k: v for k, v in entry_ctx.items() if v is not None},
    }
    ctx["sl_mult"] = float(sl_adj)
    ctx["tp_mult"] = float(tp_adj)
    ctx["sl_pct"] = float(sl_pct)
    ctx["atr_pct"] = float(atr_pct)
    open_trade_ctx[symbol] = ctx
    memory_manager.add_trade_open(ctx)
    # [ANCHOR:ENTRY_LOGGING_CALLSITE]
    trade_id = log_entry(symbol, ctx, trade_log_path)
    pair_state.setdefault(symbol, {})["trade_id"] = trade_id
    symbol_activity[symbol] = datetime.now(timezone.utc)
    log_decision(symbol, entry_ctx.get("reason", "model_confirmed"), decision="entry")
    try:
        ensure_exit_orders(
            ADAPTER,
            symbol,
            "long" if signal == "long" else "short",
            qty,
            ctx.get("sl_price"),
            ctx.get("tp_price"),
        )
    except Exception as exc:
        logging.warning("exit_guard | %s | ensure_exit_orders failed: %s", symbol, exc)
    return True


def attempt_direct_market_entry(
    symbol: str,
    direction: str,
    *,
    ctx: dict,
    df_trend: pd.DataFrame | None,
    multi_df: pd.DataFrame | None,
    mode_params: dict,
    risk_factor: float,
    atr_value: float,
    leverage: int,
    price_hint: float | None = None,
) -> bool:
    """Fallback execution path to place a market order when ``run_trade`` fails."""

    side = "buy" if direction == "long" else "sell"

    price_candidates: list[float] = []
    if price_hint is not None:
        price_candidates.append(price_hint)
    if multi_df is not None and not multi_df.empty:
        for col in ("close_15m", "close_5m", "close"):
            if col in multi_df.columns:
                series = multi_df[col].dropna()
                if not series.empty:
                    price_candidates.append(series.iloc[-1])
                    break
    if df_trend is not None and not df_trend.empty and "close" in df_trend.columns:
        price_candidates.append(df_trend["close"].iloc[-1])

    last_price: float | None = None
    for candidate in price_candidates:
        try:
            cand = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cand) and cand > 0:
            last_price = cand
            break

    if last_price is None or last_price <= 0:
        ticker = None
        try:
            ticker = exchange.fetch_ticker(symbol)
        except Exception as exc:
            log_once(
                "warning",
                f"fallback trade | {symbol} | fetch_ticker failed: {exc}",
            )
        if isinstance(ticker, dict):
            for key in ("last", "close", "ask", "bid"):
                try:
                    candidate = float(ticker.get(key) or 0.0)
                except (TypeError, ValueError):
                    candidate = 0.0
                if math.isfinite(candidate) and candidate > 0:
                    last_price = candidate
                    break
            if (last_price is None or last_price <= 0) and isinstance(ticker.get("info"), dict):
                info = ticker["info"]
                for key in ("lastPrice", "close", "markPrice", "price"):
                    try:
                        candidate = float(info.get(key) or 0.0)
                    except (TypeError, ValueError):
                        candidate = 0.0
                    if math.isfinite(candidate) and candidate > 0:
                        last_price = candidate
                        break

    if last_price is None or last_price <= 0:
        log_decision(symbol, "price_unavailable")
        return False

    category = detect_market_category(ADAPTER.x, symbol) or "linear"
    category = str(category or "").lower()
    if category in ("", "swap"):
        category = "linear"
    if category not in {"linear", "inverse"}:
        log_decision(
            symbol,
            "no_futures_contract",
            detail=f"entry | {symbol} | skip: unsupported market category {category or 'unknown'}",
        )
        return False
    side_norm = str(side or "").lower()
    qty_signed, qty_abs = has_open_position(ADAPTER.x, symbol, category)
    if qty_abs > 0 and (
        (side_norm == "buy" and qty_signed > 0)
        or (side_norm == "sell" and qty_signed < 0)
    ):
        log_decision(
            symbol,
            "position_already_open",
            detail=f"entry | {symbol} | skip: position already open (qty={qty_signed:.4f})",
        )
        return False
    if has_pending_entry(ADAPTER.x, symbol, side, category):
        log_decision(
            symbol,
            "pending_entry_exists",
            detail=f"entry | {symbol} | skip: pending entry exists ({side})",
        )
        return False
    now_bar5 = int(time.time() // (5 * 60))
    guard_state = _entry_guard.get(symbol) or {}
    if guard_state.get("bar") == now_bar5:
        log_decision(
            symbol,
            "entry_guard_active",
            detail=f"entry | {symbol} | skip: already entered this bar",
        )
        return False

    balance = 0.0
    available_margin = 0.0
    try:
        bal_info = safe_fetch_balance(exchange, {"type": "future"})
        totals = bal_info.get("total") if isinstance(bal_info, dict) else None
        frees = bal_info.get("free") if isinstance(bal_info, dict) else None
        balance = float((totals or {}).get("USDT", 0.0))
        available_margin = float((frees or totals or {}).get("USDT", 0.0))
    except Exception as exc:
        log_once(
            "warning",
            f"fallback trade | {symbol} | fetch_balance failed: {exc}",
        )
    try:
        market = exchange.market(symbol) or {}
    except Exception as exc:
        log_once(
            "warning",
            f"fallback trade | {symbol} | market lookup failed: {exc}",
        )
        market = {}
    min_qty = float(
        ((market.get("limits") or {}).get("amount") or {}).get("min", 0.0) or 0.0
    )

    symbol_norm = _normalize_bybit_symbol(ADAPTER.x, symbol, category)
    qty_target = _compute_entry_qty(
        symbol,
        side,
        last_price,
        leverage,
        balance,
        available_margin,
        risk_factor,
    )
    if qty_target <= 0:
        log_decision(symbol, "qty_insufficient")
        return False

    affordable_qty = _max_affordable_amount(
        exchange,
        symbol,
        side,
        max(int(leverage or 1), 1),
        last_price,
        MIN_NOTIONAL,
    )
    if affordable_qty <= 0:
        log_decision(symbol, "insufficient_balance")
        return False
    qty_target = min(qty_target, affordable_qty)

    qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
    if qty_target <= 0:
        log_decision(symbol, "qty_insufficient")
        return False

    adjusted_qty, margin_reason = _adjust_qty_for_margin(
        exchange,
        symbol,
        qty_target,
        last_price,
        leverage,
        available_margin,
        min_qty,
    )
    if adjusted_qty is None or adjusted_qty <= 0:
        log_decision(symbol, margin_reason or "insufficient_balance")
        return False
    qty_target = adjusted_qty

    max_pos_qty = get_max_position_qty(symbol, leverage, last_price)
    if max_pos_qty:
        qty_target = min(qty_target, max_pos_qty)

    qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
    if qty_target <= 0:
        log_decision(symbol, "qty_insufficient")
        return False

    precision = market.get("precision") or {}
    price_precision = 0
    if isinstance(precision, dict):
        try:
            price_precision = int(precision.get("price") or 0)
        except (TypeError, ValueError):
            price_precision = 0
    tick_size = 1 / (10 ** price_precision) if price_precision else None

    atr_val = float(atr_value or 0.0)
    atr_pct = atr_val / last_price if last_price else 0.0
    sl_mult = mode_params.get("sl_mult", 2.0)
    tp_mult = mode_params.get("tp_mult", 4.0)
    if atr_pct > 0.01:
        sl_mult *= 1.2
        tp_mult *= 1.2

    mode_params_calc = {"sl_mult": float(sl_mult), "tp_mult": float(tp_mult)}
    tp_price_raw, sl_price_raw, _ = risk_management.calc_sl_tp(
        last_price,
        atr_val,
        mode_params_calc,
        "long" if direction == "long" else "short",
        tick_size=tick_size,
    )

    try:
        sl_price = float(exchange.price_to_precision(symbol, sl_price_raw))
    except Exception:
        sl_price = float(sl_price_raw)
    try:
        tp_price = float(exchange.price_to_precision(symbol, tp_price_raw))
    except Exception:
        tp_price = float(tp_price_raw)

    order_id = None
    try:
        filled_qty = enter_ensure_filled(
            ADAPTER.x,
            symbol,
            side,
            qty_target,
            category=category,
        )
    except Exception as exc:
        log_once(
            "warning",
            f"fallback trade | {symbol} | ensure_filled failed: {exc}",
        )
        log_decision(symbol, "order_failed")
        return False
    if (filled_qty or 0.0) <= 0:
        log_decision(symbol, "order_failed")
        return False
    _entry_guard[symbol] = {"bar": now_bar5, "side": side_norm}

    detected_qty = wait_position_after_entry(
        ADAPTER.x, symbol, category=category, timeout_sec=3.0
    )
    if detected_qty <= 0:
        log_once(
            "warning",
            f"fallback trade | {symbol} | filled order but position not visible yet; exits postponed",
            window_sec=60.0,
        )

    _pos_after, pos_abs_after = has_open_position(ADAPTER.x, symbol, category)
    entry_price = get_position_entry_price(ADAPTER.x, symbol, category) or last_price

    want_long = direction == "long"
    try:
        sl_pct_eff = abs((sl_price / entry_price) - 1) if entry_price else SL_PCT
    except Exception:
        sl_pct_eff = SL_PCT
    if not sl_pct_eff or not math.isfinite(sl_pct_eff):
        sl_pct_eff = SL_PCT
    try:
        tp_pct_eff = abs((tp_price / entry_price) - 1) if entry_price else TP_PCT
    except Exception:
        tp_pct_eff = TP_PCT
    if not tp_pct_eff or not math.isfinite(tp_pct_eff):
        tp_pct_eff = TP_PCT

    if pos_abs_after > 0:
        try:
            _, err = place_conditional_exit(
                ADAPTER.x,
                symbol,
                "buy" if want_long else "sell",
                entry_price,
                last_price,
                sl_pct_eff,
                category,
                is_tp=False,
            )
            if err:
                log_once("warning", f"fallback trade | {symbol} | Failed to set SL: {err}")
        except RuntimeError as exc:
            log_once("warning", f"fallback trade | {symbol} | Failed to set SL: {exc}")
        except Exception as exc:
            log_once("warning", f"fallback trade | {symbol} | Failed to set SL: {exc}")

        try:
            _, err = place_conditional_exit(
                ADAPTER.x,
                symbol,
                "buy" if want_long else "sell",
                entry_price,
                last_price,
                tp_pct_eff,
                category,
                is_tp=True,
            )
            if err:
                log_once("warning", f"fallback trade | {symbol} | Failed to set TP: {err}")
        except RuntimeError as exc:
            log_once("warning", f"fallback trade | {symbol} | Failed to set TP: {exc}")
        except Exception as exc:
            log_once("warning", f"fallback trade | {symbol} | Failed to set TP: {exc}")

    qty = float(pos_abs_after or detected_qty or filled_qty)

    now = datetime.now(timezone.utc)
    ctx_copy = {k: v for k, v in ctx.items() if v is not None}
    ctx_copy.update(
        {
            "symbol": symbol,
            "side": direction.upper(),
            "entry_price": float(entry_price),
            "entry_time": now.isoformat().replace("+00:00", "Z"),
            "qty": float(qty),
            "open_bar_index": int(now.timestamp() // (5 * 60)),
            "trailing_profit_used": False,
            "order_id": order_id,
            "atr": float(atr_val),
            "tick_size": float(tick_size or 0.0),
            "sl_price": float(sl_price) if sl_price is not None else None,
            "tp_price": float(tp_price) if tp_price is not None else None,
        }
    )

    open_trade_ctx[symbol] = ctx_copy
    memory_manager.add_trade_open(ctx_copy)
    trade_id = log_entry(symbol, ctx_copy, trade_log_path)
    pair_state.setdefault(symbol, {})["trade_id"] = trade_id
    symbol_activity[symbol] = now

    logging.info(
        colorize(
            f"trade | {symbol} | fallback market entry {direction.upper()} qty={qty:.6f} price={entry_price:.6f}",
            "open",
        )
    )
    log_decision(symbol, ctx.get("reason", "model_confirmed"), decision="entry")
    try:
        ensure_exit_orders(
            ADAPTER,
            symbol,
            direction.lower(),
            qty,
            ctx_copy.get("sl_price"),
            ctx_copy.get("tp_price"),
        )
    except Exception as exc:
        log_once("warning", f"exit_guard | {symbol} | ensure_exit_orders failed: {exc}")
    return True


def open_reverse_position_with_reduced_risk(
    symbol: str,
    side: str,
    df_trend: pd.DataFrame | None = None,
    risk_multiplier: float = 1.0,
) -> None:
    """Open a reverse position using a reduced risk factor."""

    risk_factor = float(np.random.uniform(0.5, 0.75)) * risk_multiplier
    logging.info(
        f"reverse | {symbol} | Opening {side} position with risk_factor={risk_factor:.2f}"
    )
    run_trade(
        symbol,
        side.lower(),
        df_trend=df_trend,
        stats=stats,
        state=risk_state.get(symbol, PairState()),
        entry_ctx={"reduced_risk": True},
        risk_factor=risk_factor,
    )


def param_grid_search(symbols=["ETH/USDT", "SOL/USDT", "BNB/USDT", "SUI/USDT", "TON/USDT", "XRP/USDT", "TRX/USDT"]):
    # Сетка параметров для перебора
    THRESHOLDS = [0.0005, 0.001, 0.0015, 0.002]
    SL_PCTS = [0.01, 0.015, 0.02, 0.03]
    TP_PCTS = [0.03, 0.04, 0.05, 0.06]
    PROBA_FILTERS = [MIN_PROBA_FILTER, BASE_PROBA_FILTER]
    ADX_VALUES = [MIN_ADX_THRESHOLD, 15, 20]
    RSI_OB_VALUES = [65, 70, 75]
    RSI_OS_VALUES = [25, 30, 35]

    best_result = {}
    logging.info("==== PARAM GRID SEARCH ====")

    for symbol in symbols:
        results = []
        logging.info(f"[ {symbol} ]")
        for params in itertools.product(
            THRESHOLDS, SL_PCTS, TP_PCTS, PROBA_FILTERS, ADX_VALUES, RSI_OB_VALUES, RSI_OS_VALUES
        ):
            th, sl, tp, pf, adx, rsi_ob, rsi_os = params
            # --- Применяем параметры
            globals()["THRESHOLD"] = th
            globals()["SL_PCT"] = sl
            globals()["TP_PCT"] = tp
            globals()["PROBA_FILTER"] = pf
            globals()["ADX_THRESHOLD"] = adx
            globals()["RSI_OVERBOUGHT"] = rsi_ob
            globals()["RSI_OVERSOLD"] = rsi_os
            try:
                ret_dict = backtest(symbol)
                profit = ret_dict.get("return") if ret_dict else None
                results.append(
                    {
                        "THRESHOLD": th,
                        "SL_PCT": sl,
                        "TP_PCT": tp,
                        "PROBA_FILTER": pf,
                        "ADX_THRESHOLD": adx,
                        "RSI_OVERBOUGHT": rsi_ob,
                        "RSI_OVERSOLD": rsi_os,
                        "profit": profit,
                    }
                )
                if profit is not None:
                    logging.info(
                        f"TH={th:.4f} SL={sl:.3f} TP={tp:.3f} PF={pf:.2f} "
                        f"ADX={adx} RSI_OB={rsi_ob} RSI_OS={rsi_os} | profit={profit:.2%}"
                    )
                else:
                    logging.warning(
                        f"Grid search: metrics missing for {symbol} params {params}"
                    )
            except Exception as e:
                logging.error(f"Grid search skip {params}: {e}")
        # Находим лучший вариант
        if results:
            best = max(results, key=lambda x: x["profit"])
            logging.info(f"BEST for {symbol}: {best}")
            best_result[symbol] = best
    logging.info("==== GRID SEARCH DONE ====")
    logging.info("Best params by symbol:")
    for s, br in best_result.items():
        logging.info(f"{s}: {br}")
        params_only = {k: br[k] for k in br if k != "profit"}
        best_params_cache[s] = params_only
    if best_result:
        first_params = next(iter(best_result.values()))
        apply_params({k: v for k, v in first_params.items() if k != "profit"})
        DEFAULT_PARAMS.update({k: v for k, v in first_params.items() if k != "profit"})
        save_param_cache()
    return best_result


def cancel_stale_orders(symbol: str) -> int:
    """Отменяет открытые/висящие ордера по символу. Возвращает int."""
    try:
        cnt, _ids = ADAPTER.cancel_open_orders(symbol)
        cnt = int(cnt or 0)
        if cnt > 0:
            logging.info(f"order | {symbol} | cancelled open orders: {cnt}")
        else:
            logging.debug(f"order | {symbol} | no open orders to cancel")
        return cnt
    except Exception as e:  # pragma: no cover - logging only
        logging.error(f"order | {symbol} | cancel_stale_orders failed: {e}")
        return 0


def cancel_all_open_orders(symbol):
    """Cancel all open orders for a symbol regardless of open positions."""
    cnt, _ids = ADAPTER.cancel_open_orders(symbol)
    cnt = int(cnt or 0)
    if cnt > 0:
        logging.info(f"order | {symbol} | cancelled open orders: {cnt}")
    else:
        logging.debug(f"order | {symbol} | no open orders to cancel")


def cancel_related_orders(symbol: str) -> None:
    """Cancel all open orders associated with *symbol*.

    Used after a position has been closed to clean up any remaining
    orders such as protective stops or take-profit orders.
    """
    cnt, _ids = ADAPTER.cancel_open_orders(symbol)
    cnt = int(cnt or 0)
    if cnt > 0:
        logging.info(f"order | {symbol} | cancelled open orders: {cnt}")
    else:
        logging.debug(f"order | {symbol} | no open orders to cancel")


# [ANCHOR:CLOSE_UTILS]
def roi_reached(pnl_pct: float, roi_target_pct: float = 1.5) -> bool:
    return float(pnl_pct) >= float(roi_target_pct)


def cancel_all_child_orders(symbol: str) -> None:
    """Cancel all stop/limit/trailing orders for ``symbol``.

    Errors are logged but not raised.
    """
    cnt, _ids = ADAPTER.cancel_open_orders(symbol)
    cnt = int(cnt or 0)
    if cnt > 0:
        logging.info(f"order | {symbol} | cancelled open orders: {cnt}")
    else:
        logging.debug(f"order | {symbol} | no open orders to cancel")


def market_close(symbol: str) -> dict:
    """Close open position for symbol using a protected exit."""
    try:
        positions = fetch_positions_soft(symbol)
        for pos in positions:
            if float(pos.get("contracts", 0)) > 0:
                side = pos.get("side", "").upper()
                qty = float(pos.get("contracts", 0))
                close_side = "sell" if side == "LONG" else "buy"
                last_price = exchange.fetch_ticker(symbol)["last"]
                order_id = place_protected_exit(
                    symbol,
                    "TAKE_PROFIT_MARKET",
                    close_side,
                    qty,
                    last_price,
                    reduce_only=False,
                    close_all=True,
                    reference_price=float(last_price) if last_price else None,
                )
                if order_id:
                    return {"id": order_id}
    except Exception as e:
        logging.error(f"trade | {symbol} | Failed to close position: {e}")
    return {}


def save_candle_chart(df, symbol, filename="chart.png"):
    """Save a simple closing price chart for ``symbol``.

    Missing or incomplete data is treated as a soft failure: the function logs
    a warning and returns without raising, allowing the caller to continue.
    The function gracefully skips plotting when ``matplotlib`` is unavailable
    or lacks the required APIs.
    """

    if df is None or df.empty:
        logging.info(f"chart | {symbol} | No data to plot (soft skip)")
        record_no_data(symbol, "chart", "empty_dataframe")
        return

    if "close" not in df.columns:
        logging.info(
            f"chart | {symbol} | Missing close column; skipping chart generation"
        )
        record_no_data(symbol, "chart", "missing_close")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            fig, ax = plt.subplots(figsize=(6, 3))
        except Exception:
            logging.info(
                f"chart | {symbol} | matplotlib lacks figure/subplots; skipping chart"
            )
            record_no_data(symbol, "chart", "no_figure_api")
            return

        ax.plot(df["close"], label="Close")
        ax.set_title(f"{symbol} - Last Candles")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(filename)
    except Exception as e:
        logging.warning(f"chart | {symbol} | plotting failed: {e}")
        record_error(symbol, f"chart plotting failed: {e}")
    finally:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass


def get_percent_price_limits(symbol):
    """Return min and max allowed prices for a symbol from PERCENT_PRICE filter."""
    try:
        market = exchange.market(symbol)
        filters = {f["filterType"]: f for f in market["info"].get("filters", [])}
        pf = filters.get("PERCENT_PRICE")
        if pf:
            best_price = exchange.fetch_ticker(symbol)["last"]
            min_price = best_price * float(pf["multiplierDown"])
            max_price = best_price * float(pf["multiplierUp"])
            return min_price, max_price
    except Exception as e:
        logging.error(f"Failed to fetch PERCENT_PRICE filter for {symbol}: {e}")
    return None, None


def adjust_price_to_percent_filter(symbol, price):
    """Clamp price to exchange PERCENT_PRICE limits if available."""
    if price is None:
        return price
    min_price, max_price = get_percent_price_limits(symbol)
    if min_price is not None and price < min_price:
        logging.warning(f"{symbol} price {price} below limit {min_price}; adjusted")
        return min_price
    if max_price is not None and price > max_price:
        logging.warning(f"{symbol} price {price} above limit {max_price}; adjusted")
        return max_price
    return price


def get_max_position_qty(symbol: str, leverage: int, price: float) -> float | None:
    """Return approximate maximum position size for given leverage.

    The function tries ``fetch_leverage_tiers`` first and falls back to the raw
    ``fapiPrivate_get_leverageBracket`` method. If both are missing it returns
    the market's ``amount.max`` limit if available.
    """
    try:
        market = exchange.market(symbol)
        market_category = None
        if market:
            market_category = (
                market.get("type")
                or ("spot" if market.get("spot") else None)
            )
        if not market_category:
            try:
                market_category = detect_market_category(exchange, symbol)
            except Exception as detect_err:  # pragma: no cover - best effort logging
                logging.debug(
                    "Unable to detect market category for %s: %s", symbol, detect_err
                )

        # For spot markets we only rely on the amount limit and avoid derivatives API calls
        if str(market_category).lower() == "spot":
            max_amt = (market or {}).get("limits", {}).get("amount", {}).get("max")
            if max_amt:
                return float(exchange.amount_to_precision(symbol, max_amt))
            return None

        tiers = None
        if hasattr(exchange, "fetch_leverage_tiers"):
            try:
                tiers_info = exchange.fetch_leverage_tiers([symbol])
                tiers = tiers_info.get(symbol)
            except Exception as e:
                logging.info("Failed to fetch leverage tiers for %s: %s", symbol, e)
        if tiers is None:
            # CCXT exposes this endpoint under various naming conventions
            fetch_lb = (
                getattr(exchange, "fapiPrivate_get_leverageBracket", None)
                or getattr(exchange, "fapiPrivateGetLeverageBracket", None)
                or getattr(exchange, "fapiprivate_get_leveragebracket", None)
            )
            if fetch_lb:
                try:
                    info = fetch_lb({"symbol": market["id"]})
                    tiers = info[0].get("brackets", [])
                except Exception as e:
                    logging.info(
                        "Failed to fetch leverage bracket for %s: %s", symbol, e
                    )
            else:
                logging.debug(
                    "Failed to fetch leverage bracket for %s: API method missing", symbol
                )
        if tiers:
            notional_cap = None
            for br in tiers:
                level = int(br.get("initialLeverage") or br.get("maxLeverage") or 0)
                if level >= leverage:
                    notional_cap = float(br.get("notionalCap", 0))
                    break
            if notional_cap is None and tiers:
                last = tiers[-1]
                notional_cap = float(last.get("notionalCap", 0))
            if notional_cap:
                return float(exchange.amount_to_precision(symbol, notional_cap / price))
        max_amt = (market or {}).get("limits", {}).get("amount", {}).get("max")
        if max_amt:
            return float(exchange.amount_to_precision(symbol, max_amt))
    except Exception as e:
        logging.info("Failed to determine leverage bracket for %s: %s", symbol, e)
    return None


def place_protected_exit(
    symbol,
    order_type,
    side,
    qty,
    stop_price,
    reduce_only: bool = False,
    close_all: bool = True,
    *,
    reference_price: float | None = None,
    max_retries: int = 2,
):
    """Place TP/SL orders with Bybit-compliant parameters and logging."""

    upper_kind = str(order_type or "").upper()
    side_norm = str(side or "").lower()
    position_side = "buy" if side_norm == "sell" else "sell"
    is_stop = "STOP" in upper_kind and "TAKE_PROFIT" not in upper_kind
    is_take_profit = "TAKE_PROFIT" in upper_kind

    if not (is_stop or is_take_profit):
        return None

    category = detect_market_category(exchange, symbol) or "linear"
    category = str(category or "").lower()
    if category in ("", "swap"):
        category = "linear"

    ticker: dict[str, Any] | None = None
    try:
        ticker = exchange.fetch_ticker(symbol)
    except Exception:
        ticker = None

    last_price = None
    if isinstance(ticker, dict):
        for key in ("last", "close", "ask", "bid"):
            try:
                candidate = float(ticker.get(key) or 0.0)
            except (TypeError, ValueError):
                candidate = 0.0
            if math.isfinite(candidate) and candidate > 0:
                last_price = candidate
                break

    if last_price is None or last_price <= 0:
        try:
            last_price = float(reference_price) if reference_price else None
        except (TypeError, ValueError):
            last_price = None

    def _pct(target: float | None, default: float) -> float:
        if target is None:
            return default
        try:
            base = float(reference_price) if reference_price and reference_price > 0 else None
        except Exception:
            base = None
        if base is None:
            base = float(target)
        try:
            diff = abs(float(target) / base - 1)
            if diff > 0:
                return diff
        except Exception:
            pass
        return default

    if is_stop:
        if stop_price is None:
            return None
        try:
            stop_prec = float(exchange.price_to_precision(symbol, stop_price))
        except Exception:
            stop_prec = float(stop_price)
        base_value = reference_price if reference_price and reference_price > 0 else stop_prec
        try:
            base_value = float(base_value)
        except (TypeError, ValueError):
            base_value = stop_prec
        sl_pct = _pct(stop_prec, 0.02)
        entry_for_exit = (
            float(reference_price)
            if reference_price and reference_price > 0
            else float(base_value)
        )
        last_for_exit = last_price if last_price and last_price > 0 else entry_for_exit
        try:
            order_id, err = place_conditional_exit(
                exchange,
                symbol,
                position_side,
                entry_for_exit,
                last_for_exit,
                sl_pct,
                category,
                is_tp=False,
            )
        except RuntimeError as exc:
            if not str(exc).lower().startswith("exit skipped"):
                message = f"order | {symbol} | stop order rejected: {exc}"
                log_once("error", message)
                record_error(symbol, f"failed to set {order_type}")
            return None
        except Exception as exc:
            message = f"order | {symbol} | stop order rejected: {exc}"
            log_once("error", message)
            record_error(symbol, f"failed to set {order_type}")
            return None
        if err:
            if not str(err).lower().startswith("exit skipped"):
                message = f"order | {symbol} | stop order rejected: {err}"
                log_once("error", message)
                record_error(symbol, f"failed to set {order_type}")
            return None
        logging.info("order | %s | %s placed", symbol, order_type)
        return order_id

    if stop_price is None:
        return None

    try:
        tp_price = float(exchange.price_to_precision(symbol, stop_price))
    except Exception:
        tp_price = float(stop_price)

    base_tp = reference_price if reference_price and reference_price > 0 else tp_price
    try:
        base_tp = float(base_tp)
    except (TypeError, ValueError):
        base_tp = tp_price
    tp_pct = _pct(tp_price, 0.04)
    entry_for_exit = (
        float(reference_price)
        if reference_price and reference_price > 0
        else float(base_tp)
    )
    last_for_exit = last_price if last_price and last_price > 0 else entry_for_exit

    try:
        order_id, err = place_conditional_exit(
            exchange,
            symbol,
            position_side,
            entry_for_exit,
            last_for_exit,
            tp_pct,
            category,
            is_tp=True,
        )
    except RuntimeError as exc:
        if not str(exc).lower().startswith("exit skipped"):
            log_once("error", f"order | {symbol} | take-profit rejected: {exc}")
            record_error(symbol, f"failed to set {order_type}")
        return None
    except Exception as exc:
        log_once("error", f"order | {symbol} | take-profit rejected: {exc}")
        record_error(symbol, f"failed to set {order_type}")
        return None
    if err:
        if not str(err).lower().startswith("exit skipped"):
            log_once("error", f"order | {symbol} | take-profit rejected: {err}")
            record_error(symbol, f"failed to set {order_type}")
        return None

    logging.info("order | %s | %s placed qty=%s price=%s", symbol, order_type, qty, tp_price)
    return order_id


def ensure_exit_orders(
    adapter: ExchangeAdapter | Any,
    symbol: str,
    side: str,
    qty: float | int | str,
    sl_price: float | None,
    tp_price: float | None,
) -> None:
    """Ensure stop-loss and take-profit orders exist for *symbol*."""

    exchange_obj = getattr(adapter, "client", None) or getattr(adapter, "x", None) or adapter
    if not exchange_obj or not hasattr(exchange_obj, "fetch_open_orders"):
        return

    is_bybit = _is_bybit_exchange(exchange_obj)
    category = detect_market_category(exchange_obj, symbol) if is_bybit else None

    try:
        qty_value = float(qty)
    except (TypeError, ValueError):
        qty_value = 0.0
    if qty_value <= 0:
        return

    side_norm = str(side or "").lower()
    exit_side = "sell" if side_norm == "long" else "buy"
    fetch_params: dict[str, Any] = {}
    if is_bybit:
        fetch_params["category"] = category or "linear"
        if category != "spot":
            fetch_params.setdefault("positionIdx", 0)

    normalized_symbol = symbol
    if is_bybit:
        if not getattr(exchange_obj, "markets", None):
            seen_ids: set[int] = set()

            def _try_load_markets(obj: Any | None) -> None:
                if obj is None:
                    return
                ident = id(obj)
                if ident in seen_ids:
                    return
                seen_ids.add(ident)
                for attr in ("load_markets_safe", "load_markets"):
                    loader = getattr(obj, attr, None)
                    if callable(loader):
                        try:
                            loader()
                        except Exception:
                            continue
                        break

            _try_load_markets(exchange_obj)
            if not getattr(exchange_obj, "markets", None):
                _try_load_markets(getattr(adapter, "client", None))
            if not getattr(exchange_obj, "markets", None):
                _try_load_markets(getattr(adapter, "x", None))
            if not getattr(exchange_obj, "markets", None):
                _try_load_markets(adapter)

        normalized_symbol = _normalize_bybit_symbol(
            exchange_obj,
            symbol,
            fetch_params.get("category"),
        )

    try:
        qty_value = float(exchange_obj.amount_to_precision(normalized_symbol, qty_value))
    except Exception:
        qty_value = float(qty_value)
    if qty_value <= 0:
        return

    try:
        if fetch_params:
            try:
                orders = (
                    exchange_obj.fetch_open_orders(
                        normalized_symbol, None, None, fetch_params
                    )
                    or []
                )
            except TypeError:
                try:
                    orders = (
                        exchange_obj.fetch_open_orders(
                            normalized_symbol, None, fetch_params
                        )
                        or []
                    )
                except TypeError:
                    orders = (
                        exchange_obj.fetch_open_orders(normalized_symbol, fetch_params) or []
                    )
        else:
            orders = exchange_obj.fetch_open_orders(normalized_symbol) or []
    except Exception as exc:
        guard_state = exit_orders_fetch_guard.setdefault(
            symbol, {"blocked": False, "warned": False}
        )
        if not guard_state["warned"]:
            logging.warning("exit_guard | %s | fetch_open_orders failed: %s", symbol, exc)
            guard_state["warned"] = True
        else:
            logging.debug(
                "exit_guard | %s | fetch_open_orders still failing: %s", symbol, exc
            )
        guard_state["blocked"] = True
        return
    else:
        guard_state = exit_orders_fetch_guard.get(symbol)
        if guard_state and (guard_state.get("blocked") or guard_state.get("warned")):
            guard_state["blocked"] = False
            guard_state["warned"] = False

    def _order_type(order: dict) -> str:
        if not isinstance(order, dict):
            return ""
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        candidates = [
            order.get("type"),
            order.get("orderType"),
            info.get("type"),
            info.get("origType"),
            info.get("orderType"),
        ]
        for cand in candidates:
            if cand:
                return str(cand).lower()
        return ""

    has_stop = any("stop" in _order_type(o) for o in orders)
    has_tp = any(
        "take_profit" in _order_type(o) or _order_type(o) == "tp" for o in orders
    )

    need_sl = sl_price is not None and not has_stop
    need_tp = tp_price is not None and not has_tp
    if not need_sl and not need_tp:
        return

    cat = detect_market_category(exchange_obj, symbol) or "linear"
    cat = str(cat or "").lower()
    if cat in ("", "swap"):
        cat = "linear"
    _, pos_qty = logging_utils.has_open_position(exchange_obj, symbol, cat)

    if pos_qty <= 0:
        for _ in range(3):
            time.sleep(0.2)
            _, pos_qty = logging_utils.has_open_position(exchange_obj, symbol, cat)
            if pos_qty > 0:
                break

    if pos_qty <= 0:
        _last_exit_qty.pop(symbol, None)
        log_once(
            "warning",
            f"exit_guard | {symbol} | postpone exits: no position yet",
            window_sec=5.0,
        )
        return

    last_q = _last_exit_qty.get(symbol)
    if last_q and abs(last_q - pos_qty) < 1e-12:
        logging.info(
            "exit_guard | %s | exits up-to-date (qty=%s), skip re-place",
            symbol,
            pos_qty,
        )
        return

    ctx = open_trade_ctx.get(symbol, {})
    try:
        entry_price = float(ctx.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0

    placed_any = False

    ticker_info: dict[str, Any] | None = None
    try:
        ticker_info = exchange_obj.fetch_ticker(symbol)
    except Exception:
        ticker_info = None

    last_price = None
    if isinstance(ticker_info, dict):
        for key in ("last", "close", "ask", "bid"):
            try:
                candidate = float(ticker_info.get(key) or 0.0)
            except (TypeError, ValueError):
                candidate = 0.0
            if math.isfinite(candidate) and candidate > 0:
                last_price = candidate
                break

    if last_price is None or last_price <= 0:
        last_price = entry_price if entry_price > 0 else None

    def _pct(target: float | None, default: float = 0.02) -> float:
        if target is None:
            return default
        if entry_price > 0:
            try:
                ratio = abs(float(target) / entry_price - 1)
                if ratio > 0:
                    return ratio
            except Exception:
                pass
        return default

    side_open = "buy" if side_norm == "long" else "sell"
    sl_base = entry_price if entry_price > 0 else (float(sl_price) if sl_price else 0.0)
    tp_base = entry_price if entry_price > 0 else (float(tp_price) if tp_price else sl_base)

    if need_sl and sl_price is not None:
        sl_pct = _pct(sl_price)
        entry_for_exit = entry_price if entry_price > 0 else float(sl_base or 0.0)
        if entry_for_exit <= 0 and sl_price is not None:
            entry_for_exit = float(sl_price)
        last_for_exit = last_price if last_price and last_price > 0 else entry_for_exit
        try:
            order_id, err = place_conditional_exit(
                exchange_obj,
                symbol,
                side_open,
                entry_for_exit,
                last_for_exit,
                sl_pct,
                cat,
                is_tp=False,
            )
        except RuntimeError as exc:
            err_lower = str(exc).lower()
            if (
                err_lower.startswith("exit skipped")
                or err_lower.startswith("exit postponed")
                or "нет позиции" in err_lower
            ):
                pass
            else:
                log_once(
                    "warning",
                    f"exit_guard | {symbol} | stop order rejected: {exc}",
                )
        except Exception as exc:
            log_once(
                "warning",
                f"exit_guard | {symbol} | stop order rejected: {exc}",
            )
        else:
            if err and not (
                str(err).lower().startswith("exit skipped")
                or str(err).lower().startswith("exit postponed")
            ):
                log_once(
                    "warning",
                    f"exit_guard | {symbol} | stop order rejected: {err}",
                )
            elif order_id:
                placed_any = True
                ctx["sl_price"] = float(sl_price)

    if need_tp and tp_price is not None:
        tp_pct = _pct(tp_price, default=0.04)
        entry_for_exit = entry_price if entry_price > 0 else float(tp_base or 0.0)
        if entry_for_exit <= 0 and tp_price is not None:
            entry_for_exit = float(tp_price)
        last_for_exit = last_price if last_price and last_price > 0 else entry_for_exit
        try:
            order_id, err = place_conditional_exit(
                exchange_obj,
                symbol,
                side_open,
                entry_for_exit,
                last_for_exit,
                tp_pct,
                cat,
                is_tp=True,
            )
        except RuntimeError as exc:
            err_lower = str(exc).lower()
            if (
                err_lower.startswith("exit skipped")
                or err_lower.startswith("exit postponed")
                or "нет позиции" in err_lower
            ):
                pass
            else:
                log_once(
                    "warning",
                    f"exit_guard | {symbol} | take-profit rejected: {exc}",
                )
        except Exception as exc:
            log_once(
                "warning",
                f"exit_guard | {symbol} | take-profit rejected: {exc}",
            )
        else:
            if err and not (
                str(err).lower().startswith("exit skipped")
                or str(err).lower().startswith("exit postponed")
            ):
                log_once(
                    "warning",
                    f"exit_guard | {symbol} | take-profit rejected: {err}",
                )
            elif order_id:
                placed_any = True
                ctx["tp_price"] = float(tp_price)

    if placed_any:
        ctx["qty"] = float(qty_value)
        open_trade_ctx[symbol] = ctx
        _last_exit_qty[symbol] = pos_qty
        msg_parts: list[str] = []
        if ctx.get("sl_price") is not None:
            msg_parts.append(f"sl={float(ctx['sl_price']):.6f}")
        if ctx.get("tp_price") is not None:
            msg_parts.append(f"tp={float(ctx['tp_price']):.6f}")
        log(
            logging.INFO,
            "exit_guard",
            symbol,
            "ensured exits " + " ".join(msg_parts) if msg_parts else "ensured exits",
        )

def update_stop_loss(symbol: str, new_sl: float) -> None:
    """Update stop-loss order for *symbol* to *new_sl*.

    This simplified implementation logs the update request. In a full trading
    system this would edit or replace the existing stop order on the
    exchange.
    """
    logging.info(f"order | {symbol} | update_stop_loss -> {new_sl:.6f}")


def detect_pattern_from_chart(image_path):
    """Deprecated wrapper for backward compatibility."""
    info = detect_pattern_image(image_path)
    return info["pattern_name"]


def determine_trend(df) -> str:
    """Return current trend state based on EMA200 and SMA100."""
    if df.empty:
        return "neutral"
    if "ema_200" not in df.columns:
        df["ema_200"] = df["close"].ewm(span=200).mean()
    if "sma_100" not in df.columns:
        df["sma_100"] = df["close"].rolling(100).mean()
    close = df["close"].iloc[-1]
    ema200 = df["ema_200"].iloc[-1]
    sma100 = df["sma_100"].iloc[-1]
    if close > ema200 and close > sma100:
        return "bullish"
    if close < ema200 and close < sma100:
        return "bearish"
    return "neutral"


def timeframe_direction(df: pd.DataFrame, tf: str) -> str:
    """Return directional bias for a given timeframe based on EMA cross."""
    col = f"close_{tf}"
    if col not in df.columns:
        return "hold"
    closes = df[col].dropna()
    if len(closes) < 25:
        return "hold"
    ema_fast = closes.ewm(span=9).mean().iloc[-1]
    ema_slow = closes.ewm(span=21).mean().iloc[-1]
    if abs(ema_fast - ema_slow) / max(closes.iloc[-1], 1e-9) < 0.001:
        return "hold"
    return "long" if ema_fast > ema_slow else "short"


def adjust_signal_by_pattern(signal: str, pattern: str, trend: str, confidence: float = 0.0, adx: float = 0.0):
    """Modify trading signal according to detected chart pattern.

    Pattern influence applies only if ``confidence`` > 0.8 and ``adx`` > 20.
    """

    if confidence <= 0.8 or adx <= 20:
        return signal, False

    direction = get_pattern_direction(pattern)

    bullish_set = {
        "ascending_triangle",
        "double_bottom",
        "falling_wedge",
        "wedge_down",
        "triangle_asc",
        "inverse_head_and_shoulders",
    }
    bearish_set = {
        "descending_triangle",
        "double_top",
        "rising_wedge",
        "wedge_up",
        "triangle_desc",
        "head_and_shoulders",
    }

    if pattern in {"flag", "pennant"}:
        if trend == "bullish":
            direction = "long"
        elif trend == "bearish":
            direction = "short"

    if direction is None:
        if pattern in bullish_set:
            direction = "long"
        elif pattern in bearish_set:
            direction = "short"

    if direction:
        if signal == "hold" or signal != direction:
            signal = direction

    return signal, False


# [ANCHOR:PATTERN_TREND_BLEND]
def adjust_signal_by_pattern_and_trend(
    signal: str,
    pattern_name: str,
    pattern_conf: float,
    trend_state: str,
    base_proba: float,
    adx: float,
) -> tuple[str, float, bool]:
    """
    Возвращает (signal, adj_proba, cancel_trade).
    """
    adj = base_proba
    strong = pattern_conf >= 0.7
    agree = (
        pattern_name in BULLISH_PATTERNS
        and signal == "long"
        and trend_state == "bullish"
    ) or (
        pattern_name in BEARISH_PATTERNS
        and signal == "short"
        and trend_state == "bearish"
    )

    if strong and agree and adx >= ADX_THRESHOLD:
        adj = max(0.0, base_proba - 0.05)
    elif strong and not agree:
        # усиливаем фильтр, но не отменяем сделку полностью
        adj = min(1.0, base_proba + 0.05)
    return signal, adj, False


def run_bot():
    commission = 0.0006  # 0.06% комиссия вход+выход
    logging.info("[run_bot] Executing trading logic...")
    global GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES
    _maybe_retrain_global()
    ensure_model_loaded(ADAPTER, symbols)
    if GLOBAL_MODEL is None:
        for fn in [
            "global_model.joblib",
            "global_scaler.pkl",
            "global_xgb.json",
        ]:
            path = os.path.join(os.path.dirname(__file__), "models", fn)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            df_features, df_target, feature_cols = fetch_and_prepare_training_data(
                ADAPTER, symbols
            )
            GLOBAL_MODEL, GLOBAL_SCALER, GLOBAL_FEATURES, GLOBAL_CLASSES = _retrain_checked(
                df_features, df_target, feature_cols
            )
        except ValueError as exc:
            logging.warning("run_bot | retrain skipped: %s", exc)
            GLOBAL_MODEL = None
            GLOBAL_SCALER = None
            GLOBAL_FEATURES = []
            GLOBAL_CLASSES = np.array([0, 1, 2])
        except Exception as exc:
            logging.error("run_bot | retrain failed: %s", exc)
            GLOBAL_MODEL = None
            GLOBAL_SCALER = None
            GLOBAL_FEATURES = []
            GLOBAL_CLASSES = np.array([0, 1, 2])
    elif not hasattr(GLOBAL_MODEL, "predict_proba"):
        logging.warning(
            "run_bot | GLOBAL_MODEL lacks predict_proba; using deterministic fallback"
        )
    try:
        import ccxt  # type: ignore
        ccxt_ver = getattr(ccxt, "__version__", "?")
    except Exception:  # pragma: no cover - optional import
        ccxt_ver = "?"
    markets_count = len(getattr(getattr(ADAPTER, "x", None), "markets", {}) or {})
    logging.info(
        "env | backend=%s sandbox=%s futures=%s ccxt=%s markets=%s",
        getattr(ADAPTER, "backend", "?"),
        getattr(ADAPTER, "sandbox", False),
        getattr(ADAPTER, "futures", getattr(ADAPTER, "is_futures", False)),
        ccxt_ver,
        markets_count,
    )
    # Reset dynamic thresholds each run to avoid stale state across tests or
    # restarts.
    global PROBA_FILTER
    PROBA_FILTER = BASE_PROBA_FILTER
    base_dir = os.path.dirname(__file__)
    global trade_log_path, profit_report_path, equity_curve_path
    trade_log_path = os.path.join(base_dir, risk_config.get("trades_path", "trades_log.csv"))
    profit_report_path = os.path.join(base_dir, risk_config.get("profit_report_path", "profit_report.csv"))
    equity_curve_path = os.path.join(base_dir, risk_config.get("equity_curve_path", "equity_curve.csv"))
    ensure_trades_csv_header(trade_log_path)

    def rebuild_reports_on_exit():
        # [ANCHOR:REPORTS_REBUILD_ON_EXIT]
        if not ENABLE_REPORTS_BUILDER:
            return
        try:
            build_profit_report(trade_log_path, profit_report_path)
            build_equity_curve(trade_log_path, equity_curve_path)
        except Exception as e:
            logging.exception("build reports failed: %s", e)

    # === ATR trailing state (инициализация) ===
    stats_dict = update_pair_stats(
        trade_log_path, risk_config.get("lookback_trades", 10)
    )
    global risk_state
    risk_state = adjust_state_by_stats(risk_state, stats_dict, risk_config)
    save_pair_report(stats_dict)
    save_risk_state(risk_state, limiter, cool, stats)
    hard_entries = 0

    banned = [s for s in symbols if stats.is_banned(s)]
    active_symbols = [s for s in symbols if s not in banned]
    for s in banned:
        log_decision(s, "symbol_banned")
    # [ANCHOR:RESERVE_SYMBOLS]
    while len(active_symbols) < BASE_SYMBOL_COUNT and reserve_symbols:
        new_sym = reserve_symbols.pop(0)
        if new_sym not in symbols:
            symbols.append(new_sym)
            active_symbols.append(new_sym)
            logging.info(f"[run_bot] Added reserve symbol: {new_sym}")

    from logging_utils import _LOGGED_EXIT_IDS  # ensure exit log cache reset each run
    _LOGGED_EXIT_IDS.clear()

    skip_summary = {"model": [], "data": []}
    try:
        _health_check(active_symbols)
    except Exception as e:
        logging.warning("health | pre-run check failed: %s", e)
        return
    if active_symbols:
        test_sym = "ETH/USDT" if "ETH/USDT" in active_symbols else active_symbols[0]

        def _extract_price(data: dict | None) -> float | None:
            if not isinstance(data, dict):
                return None
            for key in ("last", "close", "ask", "bid"):
                val = data.get(key)
                if val is None:
                    continue
                try:
                    candidate = float(val)
                except (TypeError, ValueError):
                    continue
                if candidate > 0:
                    return candidate
            return None

        base_health_tf = timeframes[0] if timeframes else "5m"
        candle_price = None
        try:
            candle_data = ADAPTER.fetch_ohlcv(test_sym, base_health_tf, limit=1)
            if candle_data:
                candle_price = float(candle_data[-1][4])
        except AdapterOHLCVUnavailable as exc:
            logging.debug("health | %s | candle price unavailable: %s", test_sym, exc)
        except Exception as exc:
            logging.debug("health | %s | candle price fetch failed: %s", test_sym, exc)

        sizing_price = None
        ticker_info = None
        try:
            ticker_info = exchange.fetch_ticker(test_sym)
        except Exception as exc:
            logging.warning("health | %s | fetch_ticker failed: %s", test_sym, exc)
        sizing_price = _extract_price(ticker_info)

        balance = 0.0
        try:
            bal_info = safe_fetch_balance(exchange, {"type": "future"})
            balance = float((bal_info.get("total") or {}).get("USDT", 0.0))
        except Exception as exc:
            logging.warning("health | %s | fetch_balance failed: %s", test_sym, exc)

        dry_qty = risk_management.compute_order_qty(
            ADAPTER,
            test_sym,
            "buy",
            balance,
            RISK_PER_TRADE,
            price=candle_price,
        )

        effective_price = candle_price
        if (effective_price is None or effective_price <= 0) and ticker_info is None:
            try:
                ticker_info = exchange.fetch_ticker(test_sym)
            except Exception as exc:
                logging.warning("health | %s | fetch_ticker retry failed: %s", test_sym, exc)
            effective_price = _extract_price(ticker_info)
        if effective_price is None or effective_price <= 0:
            effective_price = sizing_price

        if dry_qty is None:
            logging.info("health | %s | dry-run sizing unavailable", test_sym)
        elif not effective_price or effective_price <= 0:
            logging.info(
                "health | %s | dry-run qty=%.8f (price unavailable)",
                test_sym,
                dry_qty,
            )
        else:
            notional = dry_qty * effective_price
            logging.info(
                "health | %s | dry-run qty=%.8f price=%.4f notional=%.4f",
                test_sym,
                dry_qty,
                effective_price,
                notional,
            )
            if notional < 10.0:
                logging.warning(
                    "health | %s | dry-run notional %.4f below 10 USDT; consider adjusting risk",
                    test_sym,
                    notional,
                )

    for symbol in active_symbols:
        logging.info(f"[run_bot] Processing symbol: {symbol}")

        # [ANCHOR:ENABLE_BAN_RULES]
        if ENABLE_SYMBOL_BAN and stats.is_banned(symbol):
            log_decision(symbol, "symbol_banned")
            continue
        soft_risk = 0.5 if stats.pop_soft_risk(symbol) else 1.0

        # Retrieve per-symbol parameters
        params = get_symbol_params(symbol)
        apply_params(params)

        state = risk_state.get(symbol, PairState())

        # --- Early model backtest and pattern detection ---
        has_open = symbol in open_trade_ctx
        try:
            preview = fetch_multi_ohlcv(symbol, timeframes, limit=200, warn=False)
        except Exception:
            preview = None
        if preview is None:
            logging.warning(
                "data | %s | no OHLCV for required timeframes; skipping", symbol
            )
            log_decision(symbol, "ohlcv_unavailable")
            if not has_open:
                skip_summary["data"].append(symbol)
                continue
            preview = pd.DataFrame()
        if (preview.empty or "close_15m" not in preview.columns) and not has_open:
            logging.warning(
                "data | %s | no OHLCV for required timeframes; skipping", symbol
            )
            log_decision(symbol, "ohlcv_unavailable")
            skip_summary["data"].append(symbol)
            continue

        cached_ohlcv: dict[str, pd.DataFrame] = {}
        if isinstance(preview, pd.DataFrame):
            sources = preview.attrs.get("sources") if hasattr(preview, "attrs") else None
            if isinstance(sources, dict):
                cached_ohlcv = {tf: df.copy() for tf, df in sources.items() if df is not None}

        def _cached_fetch_ohlcv(tf: str, limit: int) -> pd.DataFrame | None:
            df_cached = cached_ohlcv.get(tf)
            if df_cached is not None and not df_cached.empty:
                if limit and len(df_cached) > limit:
                    return df_cached.tail(limit).copy()
                return df_cached.copy()
            df_new = fetch_ohlcv(symbol, tf, limit=limit)
            if df_new is not None and not df_new.empty:
                cached_ohlcv[tf] = df_new.copy()
            return df_new

        metrics = {}
        try:
            metrics = backtest(symbol)
        except RuntimeError as e:
            record_error(symbol, f"backtest failed: {e}")
            logging.warning(f"Backtest failed for {symbol}: {e}")
            skip_summary["model"].append(symbol)
            log_decision(symbol, "backtest_unavailable")
            continue
        except Exception as e:
            record_error(symbol, f"backtest failed: {e}")
            logging.warning(f"Backtest failed for {symbol}: {e}")
            metrics = {"mode": "unavailable"}
        if (
            not metrics
            or metrics.get("mode") == "unavailable"
            or "return" not in metrics
        ) and not has_open:
            if (
                GLOBAL_MODEL is not None
                and not preview.empty
                and "close_15m" in preview.columns
            ):
                logging.warning(
                    "backtest | %s | data unavailable, proceeding", symbol
                )
            else:
                record_error(
                    symbol,
                    f"backtest data unavailable backend={getattr(ADAPTER, 'backend', '?')} futures={getattr(ADAPTER, 'futures', getattr(ADAPTER, 'is_futures', False))} limit=300 tfs={timeframes}",
                )
                log_decision(symbol, "backtest_unavailable")
                skip_summary["data"].append(symbol)
                continue

        df_for_chart = _cached_fetch_ohlcv("15m", limit=300)
        pattern_info = {"pattern_name": "none", "source": "none", "confidence": 0.0}
        if df_for_chart is not None and not df_for_chart.empty:
            chart_path = f"chart_{symbol.replace('/', '')}_{int(time.time()*1000)}.png"
            save_candle_chart(df_for_chart, symbol, chart_path)
            img_info = detect_pattern_image(chart_path)
            data_info = asyncio.run(detect_pattern(symbol, df_for_chart))
            pattern_info = (
                data_info
                if data_info["confidence"] >= img_info["confidence"]
                else img_info
            )
            record_pattern(symbol, pattern_info["pattern_name"])
            log(
                logging.INFO,
                "pattern",
                symbol,
                f"Detected: {pattern_info['pattern_name']} ({pattern_info['source']} {pattern_info['confidence']:.2f})",
            )
        else:
            log(logging.WARNING, "pattern", symbol, "Chart data unavailable")
            pattern_info = {"pattern_name": "none", "source": "none", "confidence": 0.0}

        # run self-test to track hit rate for this symbol
        pattern_hit_rates[symbol] = pattern_self_test(df_for_chart)

        pattern_name = str(pattern_info.get("pattern_name") or "none")
        try:
            pattern_conf = float(pattern_info.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            pattern_conf = 0.0
        pattern_dir = get_pattern_direction(pattern_name)

        open_pos = None  # <-- инициализация переменной перед использованием

        # process any closed orders even if position info is unavailable
        closed_orders = safe_fetch_closed_orders(exchange, symbol, limit=5)
        reverse_done = False
        for order in reversed(closed_orders):
            oid = str(order.get("id") or "")
            # [ANCHOR:PROCESSED_ORDERS_CACHE_USE]
            if oid in _processed_order_ids:
                continue
            commission = _extract_commission(order)
            if log_exit_from_order(symbol, order, commission, trade_log_path):
                _processed_order_ids.add(oid)
                rebuild_reports_on_exit()
                try:
                    save_risk_state(risk_state, limiter, cool, stats)
                except Exception as e:
                    logging.exception("save_risk_state failed: %s", e)
                side_prev = _normalize_prev_side(order)
                if side_prev:
                    pair_state.pop(symbol, None)
                otype = str(
                    order.get("type")
                    or order.get("info", {}).get("type")
                    or order.get("info", {}).get("origType", "")
                ).upper()
                prev_side = _normalize_prev_side(order)
                if (
                    not reverse_done
                    and otype == "STOP_MARKET"
                    and prev_side in ("LONG", "SHORT")
                ):
                    opposite_side = "SHORT" if prev_side == "LONG" else "LONG"
                    dataframes = {}
                    for tf in ("5m", "15m", "30m", "1h"):
                        df_tf = _cached_fetch_ohlcv(tf, limit=100)
                        if df_tf is not None and not df_tf.empty:
                            dataframes[tf] = df_tf
                    df_4h = _cached_fetch_ohlcv("4h", limit=100)
                    if df_4h is not None and not df_4h.empty:
                        dataframes["4h"] = df_4h
                    # [ANCHOR:TREND_SAFE_CALLER]
                    try:
                        ok = confirm_trend(dataframes, opposite_side)
                    except Exception as e:
                        logging.exception("confirm_trend failed safely: %s", e)
                        ok = False
                    bar_index = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
                    equity = float(stats.stats.get(symbol, {}).get("equity", 0.0))
                    if ok and limiter.can_trade(symbol, equity) and cool.can_trade(symbol, bar_index):
                        open_reverse_position_with_reduced_risk(
                            symbol,
                            opposite_side,
                            df_trend=dataframes.get("1h"),
                            risk_multiplier=soft_risk,
                        )
                        reverse_done = True
        cancel_stale_orders(symbol)

        positions = fetch_positions_soft(symbol)
        no_position = not any(float(p.get("contracts", 0)) > 0 for p in positions)

        closed_this_cycle = False
        for pos in positions:
            if float(pos.get("contracts", 0)) > 0:
                entry_price = float(pos.get("entryPrice", 0))
                last_price = float(pos.get("markPrice", 0))
                side = str(pos.get("side", "")).upper()
                qty = float(pos.get("contracts", 0))
                ctx = open_trade_ctx.get(symbol, {})
                tick_hint = ctx.get("tick_size") or None
                tick_size = float(tick_hint) if tick_hint else None
                atr_ctx = float(ctx.get("atr") or 0.0)
                mode_params_ctx = {
                    "sl_mult": float(ctx.get("sl_mult") or 2.0),
                    "tp_mult": float(ctx.get("tp_mult") or 4.0),
                }
                sl_price_ctx = ctx.get("sl_price")
                tp_price_ctx = ctx.get("tp_price")
                price_for_calc = entry_price if entry_price > 0 else last_price
                if price_for_calc > 0 and atr_ctx > 0 and (
                    sl_price_ctx is None or tp_price_ctx is None
                ):
                    tp_new, sl_new, _ = risk_management.calc_sl_tp(
                        price_for_calc,
                        atr_ctx,
                        mode_params_ctx,
                        "long" if side == "LONG" else "short",
                        tick_size=tick_size,
                    )
                    if sl_price_ctx is None:
                        sl_price_ctx = sl_new
                    if tp_price_ctx is None:
                        tp_price_ctx = tp_new
                if sl_price_ctx is not None:
                    ctx["sl_price"] = float(sl_price_ctx)
                if tp_price_ctx is not None:
                    ctx["tp_price"] = float(tp_price_ctx)
                open_trade_ctx[symbol] = ctx
                ensure_exit_orders(
                    ADAPTER,
                    symbol,
                    "long" if side == "LONG" else "short",
                    qty,
                    sl_price_ctx,
                    tp_price_ctx,
                )
                ctx = open_trade_ctx.get(symbol, {})

                roi = 0.0
                roi_commission = 0.0
                if entry_price > 0:
                    if side == "LONG":
                        roi = (last_price - entry_price) / entry_price
                    else:
                        roi = (entry_price - last_price) / entry_price
                    roi_commission = roi - commission
                    logging.info(
                        f"roi | {symbol} | ROI with commission: {roi_commission:.2%}"
                    )

                pnl_pct = roi_commission * 100.0

                # [ANCHOR:CLOSE_ORDERING]
                if ENABLE_CLOSE_ORDERING and roi_reached(pnl_pct, ROI_TARGET_PCT):
                    logging.info(
                        f"trade | {symbol} | ROI={pnl_pct:.2f}% >= {ROI_TARGET_PCT:.2f}% -> closing position"
                    )
                    cancel_all_child_orders(symbol)
                    order = market_close(symbol)
                    ctx = open_trade_ctx.get(symbol, {})
                    ctx["exit_type_hint"] = "TP"
                    handled = log_exit_from_order(
                        symbol, order, commission, trade_log_path
                    )
                    if handled:
                        _processed_order_ids.add(str(order.get("id") or ""))
                        rebuild_reports_on_exit()
                        try:
                            save_risk_state(risk_state, limiter, cool, stats)
                        except Exception as e:
                            logging.exception("save_risk_state failed: %s", e)
                    pair_state.pop(symbol, None)
                    closed_this_cycle = True
                    break

                atr_val = float(ctx.get("atr", 0.0))
                tick = float(ctx.get("tick_size", 0.0))
                initial_sl = ctx.get("sl_price") or entry_price
                state = pair_state.setdefault(symbol, {})
                state.setdefault("sl", float(initial_sl))
                state.setdefault("r_value", abs(entry_price - float(initial_sl)))
                breakeven_done = bool(state.get("breakeven_done", False))
                current_sl = state.get("sl")
                r_value = float(state.get("r_value", 0.0))

                # [ANCHOR:TRAIL_MANAGEMENT]
                if ENABLE_ATR_TRAIL and should_activate_trailing(
                    side, entry_price, last_price, r_value, atr_val
                ):
                    new_sl, be_flag = trail_levels(
                        side,
                        entry_price,
                        last_price,
                        atr_val,
                        tick,
                        breakeven_done,
                        current_sl,
                        symbol,
                    )

                    # Не расширяем риск: обновляем только если стоп становится "жёстче"
                    if side == "LONG":
                        if (current_sl is None) or (
                            float(new_sl) > float(current_sl) and float(new_sl) >= entry_price
                        ):
                            update_stop_loss(symbol, float(new_sl))
                            state["breakeven_done"] = be_flag or breakeven_done
                            state["sl"] = float(new_sl)
                            # [ANCHOR:TRAIL_LOG]
                            log_decision(
                                symbol,
                                f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}",
                            )
                    else:
                        if (current_sl is None) or (
                            float(new_sl) < float(current_sl) and float(new_sl) <= entry_price
                        ):
                            update_stop_loss(symbol, float(new_sl))
                            state["breakeven_done"] = be_flag or breakeven_done
                            state["sl"] = float(new_sl)
                            # [ANCHOR:TRAIL_LOG]
                            log_decision(
                                symbol,
                                f"trailing_update be={state.get('breakeven_done', False)} new_sl={new_sl:@TICK}",
                            )

                    pair_state[symbol] = state
                    continue

                ctx = open_trade_ctx.get(symbol)
                open_idx = ctx.get("open_bar_index") if ctx else None
                current_bar_index = int(
                    datetime.now(timezone.utc).timestamp() // (5 * 60)
                )
                if open_idx is not None and time_stop(
                    open_idx, current_bar_index, time_stop_bars
                ):
                    logging.info(
                        f"time_stop | {symbol} | Time stop hit → closing position"
                    )
                    cancel_all_child_orders(symbol)
                    order = market_close(symbol)
                    ctx = open_trade_ctx.get(symbol, {})
                    ctx["exit_type_hint"] = "TIME"
                    handled = log_exit_from_order(
                        symbol, order, commission, trade_log_path
                    )
                    if handled:
                        _processed_order_ids.add(str(order.get("id") or ""))
                        rebuild_reports_on_exit()
                        try:
                            save_risk_state(risk_state, limiter, cool, stats)
                        except Exception as e:
                            logging.exception("save_risk_state failed: %s", e)
                    pair_state.pop(symbol, None)
                    closed_this_cycle = True
                    break

        if open_pos:
            log_decision(symbol, "position_exists")
            continue
        # === Проверка открытых позиций и автоматическая фиксация прибыли ===
        positions = fetch_positions_soft(symbol)

        # После фиксации позиции бот сразу сможет открыть новую!

        # Получим список позиций повторно, чтобы не считать только что закрытую как открытую
        positions = fetch_positions_soft(symbol)

        open_pos = None
        for pos in positions:
            if float(pos.get("contracts", 0)) > 0:
                open_pos = pos
                break
        if not open_pos:
            cancel_stale_orders(symbol)
        if open_pos:
            log_decision(symbol, "position_exists")
            continue

        if closed_this_cycle:
            log_decision(symbol, "recently_closed")
            continue

        (
            adj_proba,
            adj_adx,
            allow_conditional,
            fb_mode,
            inactivity_hours,
            adj_rsi_overbought,
            adj_rsi_oversold,
        ) = adjust_filters_for_inactivity(symbol)
        _inc_event("inactivity_hours", inactivity_hours)
        current_bar = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
        fallback_allowed = fb_mode or FALLBACK_MODE_ENABLED
        if fallback_cooldown.get(symbol, 0) > current_bar:
            fallback_allowed = False

        # Получаем тренд и режим торговли заранее, чтобы они логировались даже при пропуске
        df_trend = fetch_ohlcv(symbol, "1h", limit=250)
        if df_trend is not None and not df_trend.empty and "1h" not in cached_ohlcv:
            cached_ohlcv["1h"] = df_trend.copy()
        if df_trend is None or df_trend.empty:
            log_decision(
                symbol,
                "no_data",
                detail=f"data | {symbol} | missing timeframe 1h",
            )
            continue

        trend_state = determine_trend(df_trend)
        mode, mode_params, data_mode = select_trade_mode(symbol, df_trend)
        mode_lev = mode_params.get("lev", LEVERAGE)
        market_category = detect_market_category(exchange, symbol)
        if market_category is None:
            adapter_category = getattr(ADAPTER, "_detect_bybit_category", None)
            if callable(adapter_category):
                try:
                    market_category = adapter_category(symbol)
                except Exception as exc:
                    logging.debug(
                        "market | %s | adapter category lookup failed: %s",
                        symbol,
                        exc,
                    )
        market_category = (market_category or "").lower() or None
        derivative_categories = {
            "linear",
            "inverse",
            "swap",
            "perpetual",
            "future",
            "futures",
        }
        is_derivative_market = market_category in derivative_categories
        if market_category == "spot":
            mode_lev = 1

        # --- Prepare dataframes for trend confirmation ---
        trend_dfs: dict[str, pd.DataFrame] = {}
        df_5m = _cached_fetch_ohlcv("5m", limit=100)
        df_15m = _cached_fetch_ohlcv("15m", limit=100)
        if df_5m is None or df_15m is None:
            log_decision(
                symbol,
                "no_data",
                detail=f"data | {symbol} | missing lower timeframe data",
            )
            continue
        trend_dfs["5m"] = df_5m
        trend_dfs["15m"] = df_15m
        trend_dfs["1h"] = df_trend
        for tf in ("30m", "4h"):
            df_tf = _cached_fetch_ohlcv(tf, limit=100)
            if df_tf is not None and not df_tf.empty:
                trend_dfs[tf] = df_tf

        multi_df = preview if isinstance(preview, pd.DataFrame) and not preview.empty else None
        if multi_df is None:
            try:
                multi_df = fetch_multi_ohlcv(symbol, timeframes, limit=300, warn=False)
            except Exception:
                multi_df = None
        if multi_df is None or multi_df.empty:
            log_decision(
                symbol,
                "no_data",
                detail=f"data | {symbol} | insufficient multi-timeframe data",
            )
            continue
        required_multi_cols = {"close_15m", "volume_15m"}
        missing_multi = [col for col in required_multi_cols if col not in multi_df.columns]
        if missing_multi:
            missing_cols = ",".join(sorted(missing_multi))
            log_decision(
                symbol,
                "no_data",
                detail=f"data | {symbol} | missing multi-timeframe columns {missing_cols}",
            )
            continue
        tf_signals = {tf: timeframe_direction(multi_df, tf) for tf in ["5m", "15m", "30m"]}
        global_dir = timeframe_direction(multi_df, "4h")

        # --- HOLD обработка с fallback ---
        pattern_side: str | None = None
        strong_pattern = False
        if pattern_info["confidence"] > 0.8:
            pname = pattern_info["pattern_name"]
            direction_hint = get_pattern_direction(pname)
            if pname in STRONG_BULL_PATTERNS:
                pattern_side = "LONG"
                strong_pattern = True
            elif pname in STRONG_BEAR_PATTERNS:
                pattern_side = "SHORT"
                strong_pattern = True
            elif direction_hint:
                pattern_side = direction_hint.upper()
        trend_confirm_pattern = False
        if pattern_side:
            # [ANCHOR:TREND_SAFE_CALLER]
            try:
                trend_confirm_pattern = confirm_trend(trend_dfs, pattern_side)
            except Exception as e:
                logging.exception("confirm_trend failed safely: %s", e)
                trend_confirm_pattern = False
            if trend_confirm_pattern and strong_pattern:
                adj_proba = max(adj_proba - 0.03, 0.0)

        # [ANCHOR:VOL_FEATURES]
        vol_5m = df_5m["volume"] if "volume" in df_5m.columns else None
        vol_series = vol_5m
        # [ANCHOR:VOLUME_FILTER]
        vol_ratio = safe_vol_ratio(vol_series, VOL_WINDOW, key=symbol)
        vol_reason = volume_reason(vol_series, VOLUME_RATIO_ENTRY, VOL_WINDOW)  # 'vol_missing' | 'vol_low' | None
        vol_ok = (vol_reason == "vol_missing") or (vol_ratio >= VOLUME_RATIO_ENTRY)
        vol_risk = 0.5 if vol_reason == "vol_missing" else 1.0
        ctx = {"vol_ratio": vol_ratio, "vol_reason": vol_reason}
        # [ANCHOR:VOL_LOG]
        logging.info(
            "params | %s | VOL_RATIO=%s",
            symbol,
            "missing" if vol_ratio is None else f"{vol_ratio:.2f}",
        )

        atr_val = safe_atr(
            df_trend["atr"] if "atr" in df_trend.columns else None, key=symbol
        ) or 0.0
        if "adx" not in df_trend.columns:
            df_trend["adx"] = 0.0
        adx_val = df_trend["adx"].iloc[-1]

        # build feature row for model inference
        for lag in range(1, 7):
            multi_df[f"close_lag{lag}"] = multi_df["close_15m"].shift(lag)
            multi_df[f"volume_lag{lag}"] = multi_df["volume_15m"].shift(lag)
        multi_df = add_custom_features(multi_df)
        multi_df["ticker_cat"] = SYMBOL_CATEGORIES.get(symbol, -1)
        multi_df["pattern_name"] = PATTERN_LABELS.get(
            pattern_info["pattern_name"], -1
        )
        multi_df["pattern_source"] = PATTERN_SOURCE_MAP.get(
            pattern_info["source"], 0
        )
        multi_df["pattern_confidence"] = pattern_info["confidence"]
        X_last = multi_df.iloc[[-1]]

        returns_1h = 0.0
        if "close_1h" in multi_df.columns:
            closes = multi_df["close_1h"].dropna()
            if len(closes) >= 2:
                returns_1h = (closes.iloc[-1] / closes.iloc[-2]) - 1.0
        rsi_cross_from_extreme = None
        rsi_series = None
        if "rsi" in multi_df.columns:
            rsi_series = multi_df["rsi"].dropna()
        elif "rsi_1h" in multi_df.columns:
            rsi_series = multi_df["rsi_1h"].dropna()
        if rsi_series is not None and len(rsi_series) >= 2:
            rsi_prev, rsi_val = rsi_series.iloc[-2], rsi_series.iloc[-1]
            if rsi_prev > 65 >= rsi_val:
                rsi_cross_from_extreme = "short"
            elif rsi_prev < 35 <= rsi_val:
                rsi_cross_from_extreme = "long"

        model_signal, confidence = predict_signal(
            symbol,
            X_last,
            float(adx_val),
            bool(rsi_cross_from_extreme),
            float(returns_1h),
        )
        record_summary(
            symbol,
            mode.upper(),
            float(atr_val),
            float(adx_val),
            float(vol_ratio or 0.0),
            model_signal,
        )
        conditional = False
        pattern_conflict = False
        if model_signal != "hold":
            # Понижаем верхний порог и расширяем диапазон условных сигналов
            high_thr = max(0.33, adj_proba)
            if confidence >= high_thr:
                pass
            elif confidence >= adj_proba:
                pass
            elif allow_conditional and confidence >= MIN_PROBA_FILTER:
                # условный вход разрешён при вероятности не ниже минимального фильтра
                conditional = True
            else:
                model_signal = "hold"
        pattern_override = False
        if pattern_dir and pattern_conf >= 0.7:
            if model_signal != pattern_dir:
                logging.info(
                    "signal | %s | Pattern override %s → %s (conf=%.2f)",
                    symbol,
                    model_signal,
                    pattern_dir,
                    pattern_conf,
                )
            model_signal = pattern_dir
            pattern_override = True
        elif model_signal == "hold" and confidence >= 0.5 and pattern_dir:
            model_signal = pattern_dir

        if (
            pattern_side
            and trend_confirm_pattern
            and model_signal in ("long", "short")
            and model_signal != pattern_side.lower()
        ):
            pattern_conflict = True
            logging.info(
                "signal | %s | Aligning with pattern trend %s → %s", 
                symbol,
                model_signal,
                pattern_side.lower(),
            )
            model_signal = pattern_side.lower()
            pattern_override = True

        logging.info(
            f"signal | {symbol} | Model signal: {model_signal} (conf={confidence:.2f}{' cond' if conditional else ''})"
        )

        strong_pattern_entry = (
            pattern_override
            and pattern_dir is not None
            and model_signal in ("long", "short")
            and model_signal == pattern_dir
            and pattern_conf >= 0.7
        )

        pattern_trade_executed = False
        if strong_pattern_entry:
            current_price: float | None = None
            if not df_trend.empty and "close" in df_trend.columns:
                try:
                    current_price = float(df_trend["close"].iloc[-1])
                except (TypeError, ValueError):
                    current_price = None
            if (current_price is None or current_price <= 0) and "close_15m" in multi_df.columns:
                series = multi_df["close_15m"].dropna()
                if not series.empty:
                    try:
                        current_price = float(series.iloc[-1])
                    except (TypeError, ValueError):
                        current_price = None
            if current_price is None or current_price <= 0:
                log_decision(symbol, "price_unavailable")
            else:
                side = "buy" if model_signal == "long" else "sell"
                cat = detect_market_category(exchange, symbol) or "linear"
                cat = str(cat or "").lower()
                if cat in ("", "swap"):
                    cat = "linear"
                if cat not in {"linear", "inverse"}:
                    log_decision(symbol, "no_futures_contract")
                else:
                    qty_signed, _qty_abs = has_open_position(exchange, symbol, cat)
                    if (side == "buy" and qty_signed > 0) or (side == "sell" and qty_signed < 0):
                        log_decision(symbol, "position_already_open")
                    elif has_pending_entry(exchange, symbol, side, cat):
                        log_decision(symbol, "pending_entry_exists")
                    else:
                        now_bar5 = int(time.time() // (5 * 60))
                        guard_state = _entry_guard.get(symbol) or {}
                        if guard_state.get("bar") == now_bar5:
                            log_decision(symbol, "entry_guard_active")
                        else:
                            balance = 0.0
                            available_margin = 0.0
                            try:
                                balance_info = safe_fetch_balance(exchange, {"type": "future"})
                                totals = balance_info.get("total") if isinstance(balance_info, dict) else None
                                frees = balance_info.get("free") if isinstance(balance_info, dict) else None
                                balance = float((totals or {}).get("USDT", 0.0))
                                available_margin = float((frees or totals or {}).get("USDT", 0.0))
                            except Exception as exc:
                                logging.warning(
                                    "pattern trade | %s | fetch_balance failed: %s",
                                    symbol,
                                    exc,
                                )
                            try:
                                market = exchange.market(symbol) or {}
                            except Exception as exc:
                                logging.warning(
                                    "pattern trade | %s | market lookup failed: %s",
                                    symbol,
                                    exc,
                                )
                                market = {}
                            min_qty = float(
                                ((market.get("limits") or {}).get("amount") or {}).get("min", 0.0) or 0.0
                            )
                            precision = market.get("precision") or {}
                            price_precision = 0
                            if isinstance(precision, dict):
                                try:
                                    price_precision = int((precision or {}).get("price") or 0)
                                except (TypeError, ValueError):
                                    price_precision = 0
                            tick_size = 1 / (10 ** price_precision) if price_precision else None

                            symbol_norm = _normalize_bybit_symbol(ADAPTER.x, symbol, cat)
                            qty_target = _compute_entry_qty(
                                symbol,
                                side,
                                current_price,
                                mode_params.get("lev", LEVERAGE),
                                balance,
                                available_margin,
                                risk_factor=1.0,
                            )
                            if qty_target <= 0:
                                log_decision(symbol, "qty_insufficient")
                            else:
                                try:
                                    leverage_val = int(float(mode_params.get("lev", LEVERAGE)))
                                except Exception:
                                    leverage_val = int(LEVERAGE)
                                leverage_val = max(leverage_val, 1)
                                affordable_qty = _max_affordable_amount(
                                    exchange,
                                    symbol,
                                    side,
                                    leverage_val,
                                    current_price,
                                    MIN_NOTIONAL,
                                )
                                if affordable_qty <= 0:
                                    log_decision(symbol, "insufficient_balance")
                                else:
                                    qty_target = min(qty_target, affordable_qty)
                                    qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
                                    if qty_target <= 0:
                                        log_decision(symbol, "qty_insufficient")
                                    else:
                                        adjusted_qty, margin_reason = _adjust_qty_for_margin(
                                            exchange,
                                            symbol,
                                            qty_target,
                                            current_price,
                                            leverage_val,
                                            available_margin,
                                            min_qty,
                                        )
                                        if adjusted_qty is None or adjusted_qty <= 0:
                                            log_decision(symbol, margin_reason or "insufficient_balance")
                                        else:
                                            qty_target = adjusted_qty
                                            max_pos_qty = get_max_position_qty(symbol, leverage_val, current_price)
                                            if max_pos_qty:
                                                qty_target = min(qty_target, max_pos_qty)
                                            qty_target = _round_qty(ADAPTER.x, symbol_norm, qty_target)
                                            if qty_target <= 0:
                                                log_decision(symbol, "qty_insufficient")
                                            else:
                                                sl_mult = mode_params.get("sl_mult", 2.0)
                                                tp_mult = mode_params.get("tp_mult", 4.0)
                                                atr_pct = (atr_val / current_price) if current_price else 0.0
                                                if atr_pct > 0.01:
                                                    sl_mult *= 1.2
                                                    tp_mult *= 1.2
                                                pattern_mode_params = {
                                                    "sl_mult": float(sl_mult),
                                                    "tp_mult": float(tp_mult),
                                                }
                                                tp_price_raw, sl_price_raw, sl_pct_raw = risk_management.calc_sl_tp(
                                                    current_price,
                                                    atr_val,
                                                    pattern_mode_params,
                                                    "long" if model_signal == "long" else "short",
                                                    tick_size=tick_size,
                                                )
                                                try:
                                                    sl_price = float(exchange.price_to_precision(symbol, sl_price_raw))
                                                except Exception:
                                                    sl_price = float(sl_price_raw)
                                                try:
                                                    tp_price = float(exchange.price_to_precision(symbol, tp_price_raw))
                                                except Exception:
                                                    tp_price = float(tp_price_raw)

                                                try:
                                                    filled_qty = enter_ensure_filled(
                                                        ADAPTER.x,
                                                        symbol,
                                                        side,
                                                        qty_target,
                                                        category=cat,
                                                    )
                                                except Exception as exc:
                                                    logging.warning(
                                                        "pattern trade | %s | ensure_filled failed: %s",
                                                        symbol,
                                                        exc,
                                                    )
                                                    log_decision(symbol, "order_failed")
                                                    filled_qty = 0.0

                                                if (filled_qty or 0.0) <= 0:
                                                    log_decision(symbol, "order_failed")
                                                else:
                                                    detected_qty = wait_position_after_entry(
                                                        ADAPTER.x,
                                                        symbol,
                                                        category=cat,
                                                        timeout_sec=3.0,
                                                    )
                                                    if detected_qty <= 0:
                                                        log_decision(symbol, "position_unavailable")
                                                    else:
                                                        entry_price = get_position_entry_price(
                                                            exchange,
                                                            symbol,
                                                            cat,
                                                        )
                                                        if entry_price is None or entry_price <= 0:
                                                            entry_price = current_price
                                                        last_price = get_last_price(exchange, symbol, cat)
                                                        if last_price is None or last_price <= 0:
                                                            last_price = entry_price

                                                        try:
                                                            sl_pct_eff = abs((sl_price / entry_price) - 1)
                                                        except Exception:
                                                            sl_pct_eff = sl_pct_raw
                                                        if not sl_pct_eff or not math.isfinite(sl_pct_eff):
                                                            sl_pct_eff = max(sl_pct_raw, SL_PCT)
                                                        try:
                                                            tp_pct_eff = abs((tp_price / entry_price) - 1)
                                                        except Exception:
                                                            tp_pct_eff = TP_PCT
                                                        if not tp_pct_eff or not math.isfinite(tp_pct_eff):
                                                            tp_pct_eff = TP_PCT

                                                        _, tp_err = place_conditional_exit(
                                                            exchange,
                                                            symbol,
                                                            side,
                                                            entry_price,
                                                            last_price,
                                                            tp_pct_eff,
                                                            cat,
                                                            is_tp=True,
                                                        )
                                                        if tp_err:
                                                            log_once(
                                                                "warning",
                                                                f"pattern trade | {symbol} | Failed to set TP: {tp_err}",
                                                                window_sec=60.0,
                                                            )
                                                        _, sl_err = place_conditional_exit(
                                                            exchange,
                                                            symbol,
                                                            side,
                                                            entry_price,
                                                            last_price,
                                                            sl_pct_eff,
                                                            cat,
                                                            is_tp=False,
                                                        )
                                                        if sl_err:
                                                            log_once(
                                                                "warning",
                                                                f"pattern trade | {symbol} | Failed to set SL: {sl_err}",
                                                                window_sec=60.0,
                                                            )

                                                        _entry_guard[symbol] = {
                                                            "bar": now_bar5,
                                                            "side": side,
                                                        }
                                                        _, pos_qty = has_open_position(exchange, symbol, cat)
                                                        qty_val = float(pos_qty)
                                                        now = datetime.now(timezone.utc)
                                                        entry_ctx = {
                                                            "symbol": symbol,
                                                            "side": model_signal.upper(),
                                                            "entry_price": float(entry_price),
                                                            "entry_time": now.isoformat().replace("+00:00", "Z"),
                                                            "qty": float(qty_val),
                                                            "open_bar_index": int(now.timestamp() // (5 * 60)),
                                                            "trailing_profit_used": False,
                                                            "order_id": None,
                                                            "source": "pattern",
                                                            "reason": "pattern_override",
                                                            "atr": float(atr_val),
                                                            "tick_size": float(tick_size or 0.0),
                                                            "sl_price": float(sl_price) if sl_price is not None else None,
                                                            "tp_price": float(tp_price) if tp_price is not None else None,
                                                        }
                                                        open_trade_ctx[symbol] = entry_ctx
                                                        memory_manager.add_trade_open(entry_ctx)
                                                        trade_id = log_entry(symbol, entry_ctx, trade_log_path)
                                                        pair_state.setdefault(symbol, {})["trade_id"] = trade_id
                                                        symbol_activity[symbol] = now
                                                        log(
                                                            logging.INFO,
                                                            "order",
                                                            symbol,
                                                            f"Pattern entry {model_signal} qty={qty_val:.6f} price≈{entry_price:.6f}",
                                                        )
                                                        log_decision(symbol, "pattern_entry", decision="entry")
                                                        try:
                                                            ensure_exit_orders(
                                                                ADAPTER,
                                                                symbol,
                                                                model_signal,
                                                                qty_val,
                                                                entry_ctx.get("sl_price"),
                                                                entry_ctx.get("tp_price"),
                                                            )
                                                        except Exception as exc:
                                                            logging.warning(
                                                                "exit_guard | %s | ensure_exit_orders failed: %s",
                                                                symbol,
                                                                exc,
                                                            )
                                                        recent_hits.append(True)
                                                        recent_trade_times.append(now)
                                                        fallback_cooldown[symbol] = entry_ctx["open_bar_index"] + 2
                                                        hard_entries += 1
                                                        emit_summary(symbol, "entry")
                                                        pattern_trade_executed = True

        if pattern_trade_executed:
            continue

        if "ema_fast" not in df_trend.columns:
            df_trend["ema_fast"] = df_trend["close"].ewm(span=9).mean()
        if "ema_slow" not in df_trend.columns:
            df_trend["ema_slow"] = df_trend["close"].ewm(span=21).mean()
        ema_fast = df_trend["ema_fast"].iloc[-1]
        ema_slow = df_trend["ema_slow"].iloc[-1]
        ema_long = ema_fast > ema_slow
        ema_short = ema_fast < ema_slow

        use_fallback = False
        signal_to_use = model_signal
        fallback_attempted = False
        fb_confirm = False
        if (
            fallback_allowed
            and not pattern_conflict
            and (model_signal == "hold" or (confidence < 0.5 and strong_pattern))
        ):
            fb_sig = fallback_signal(symbol)
            if fb_sig in ["long", "short"]:
                fallback_attempted = True
                fb_confirm = (
                    adx_val >= adj_adx
                    or (ema_long if fb_sig == "long" else ema_short)
                    or tf_skip_counters.get(symbol, 0) >= TF_SKIP_THRESHOLD
                )
                if fb_confirm:
                    signal_to_use = fb_sig
                    use_fallback = True
                    _inc_event("fallback_used")
                    logging.info(
                        f"signal | {symbol} | Hold -> fallback {signal_to_use} (conf={confidence:.2f}, inactivity={inactivity_hours:.1f}h)"
                    )
                else:
                    logging.info(
                        f"signal | {symbol} | Hold no confirm (inactivity={inactivity_hours:.1f}h)"
                    )
            else:
                logging.info(
                    f"signal | {symbol} | Hold (conf={confidence:.2f}, inactivity={inactivity_hours:.1f}h)"
                )
        elif model_signal == "hold":
            logging.info(
                f"signal | {symbol} | Hold (conf={confidence:.2f}, inactivity={inactivity_hours:.1f}h)"
            )
        if signal_to_use != "hold":
            lower_matches = sum(s == signal_to_use for s in tf_signals.values())
            lower_ok = lower_matches >= MIN_LOWER_TF_MATCHES
            global_ok = global_dir in (signal_to_use, "hold")
            pname = pattern_info["pattern_name"]
            pconf = pattern_info["confidence"]
            pattern_support = (
                pconf > 0.7
                and (
                    (pname in BULLISH_PATTERNS and signal_to_use == "long")
                    or (pname in BEARISH_PATTERNS and signal_to_use == "short")
                )
            )
            # [ANCHOR:TREND_SAFE_CALLER]
            try:
                trend_ok = confirm_trend(trend_dfs, signal_to_use.upper())
            except Exception as e:
                logging.exception("confirm_trend failed safely: %s", e)
                trend_ok = False
            if not (lower_ok and global_ok):
                # Allow confident patterns with confirmed trend to override timeframe disagreements
                if pconf > 0.65 and trend_ok:
                    lower_ok = True
                    global_ok = True
                else:
                    inc_tf_skip(symbol)
                    log_decision(symbol, "tf_mismatch")
                    continue
            elif (
                lower_matches == MIN_LOWER_TF_MATCHES
                and pattern_support
                and trend_ok
            ):
                logging.info(
                    f"signal | {symbol} | Entry allowed by pattern {pname} (conf={pconf:.2f})"
                )
                adj_proba = max(adj_proba - 0.03, 0.0)
            base_thr = 0.60
            pattern_conflict = (
                (pname in BULLISH_PATTERNS and signal_to_use == "short")
                or (pname in BEARISH_PATTERNS and signal_to_use == "long")
            )
            if pattern_conflict and pconf >= base_thr and pconf < base_thr + 0.05:
                logging.info(
                    "signal | %s | Pattern/trend disagreement kept for monitoring", symbol
                )
                pattern_conflict = False
        # [ANCHOR:NORMALIZE_DECLINE_REASONS]
        reason = None
        if signal_to_use == "hold":
            reason = "hold_no_position" if no_position else "hold_position_exists"
        elif fallback_attempted and not fb_confirm:
            reason = "adx_low" if adx_val < adj_adx else "fallback_blocked"
        else:
            required_proba = apply_pattern_proba_bonus(
                adj_proba, pattern_info["confidence"], trend_ok
            )
            if confidence < required_proba:
                reason = "proba_low"
        if reason:
            reset_tf_skip(symbol)
            _inc_event(reason)
            if signal_to_use == "hold":
                _inc_event("holds")
                log_decision(symbol, reason)
            continue

        # [ANCHOR:REASONS_VOLUME]
        if vol_reason == "vol_low":
            _inc_event("vol_low")
            logging.warning(
                "low volume for %s | VOL_RATIO=%.2f", symbol, vol_ratio
            )

        # === Подтверждения сигнала ===
        # [ANCHOR:VOL_CONFIRM]
        ema_ok = ema_long if signal_to_use == "long" else ema_short
        confirmations = sum([adx_val >= adj_adx, vol_ok, ema_ok])
        # достаточно одного подтверждения или сильного паттерна
        if confirmations < 1 and not (
            use_fallback and tf_skip_counters.get(symbol, 0) >= TF_SKIP_THRESHOLD
        ):
            strong_pattern = (
                pattern_info["confidence"] >= 0.65
                and (
                    (pattern_info["pattern_name"] in BULLISH_PATTERNS and signal_to_use == "long")
                    or (
                        pattern_info["pattern_name"] in BEARISH_PATTERNS and signal_to_use == "short"
                    )
                )
            )
            if not strong_pattern:
                log_decision(symbol, "weak_confirmation")
                continue
        reset_tf_skip(symbol)

        # === Дополнительная фильтрация по тренду и паттернам ===
        rsi_val = df_trend["rsi"].iloc[-1]
        if is_derivative_market:
            leverage_category = detect_market_category(exchange, symbol)
            if str(leverage_category).lower() == "linear":
                safe_set_leverage(exchange, symbol, mode_lev)
            else:
                logging.info(
                    "leverage | %s | skip: market %s not linear",
                    symbol,
                    leverage_category or "unknown",
                )
        else:
            logging.debug(
                "leverage | %s | Skipping leverage setup for %s market",
                symbol,
                market_category or "unknown",
            )
        if not use_fallback:
            if signal_to_use == "long" and rsi_val > adj_rsi_overbought:
                log_decision(symbol, "rsi_overbought")
                continue
            if signal_to_use == "short" and rsi_val < adj_rsi_oversold:
                log_decision(symbol, "rsi_oversold")
                continue

        # [ANCHOR:PATTERN_TREND_APPLY]
        orig = signal_to_use
        signal_to_use, adj_proba, cancel = adjust_signal_by_pattern_and_trend(
            orig,
            pattern_info["pattern_name"],
            pattern_info["confidence"],
            trend_state,
            adj_proba,
            adx_val,
        )

        orig_signal = signal_to_use
        signal_to_use, _ = adjust_signal_by_pattern(
            signal_to_use,
            pattern_info["pattern_name"],
            trend_state,
            confidence=pattern_info["confidence"],
            adx=adx_val,
        )
        if signal_to_use != orig_signal:
            log(logging.INFO, "pattern", symbol, f"Signal adjusted by pattern: {orig_signal} \u2192 {signal_to_use}")

        if (
            (signal_to_use == "long" and trend_state == "bearish")
            or (signal_to_use == "short" and trend_state == "bullish")
        ):
            _inc_event("trend_conflict")
            inc_tf_skip(symbol)
            log_decision(symbol, "trend_conflict")
            continue

        # [ANCHOR:NORMALIZE_DECLINE_REASONS]
        if signal_to_use == "hold":
            positions = fetch_positions_soft(symbol)
            pos_exists = any(float(pos.get("contracts", 0)) > 0 for pos in positions)
            reason = "hold_position_exists" if pos_exists else "hold_no_position"
            reset_tf_skip(symbol)
            _inc_event("holds")
            _inc_event(reason)
            log_decision(symbol, reason)
            continue

        # [ANCHOR:TREND_SAFE_CALLER]
        try:
            trend_ok = confirm_trend(trend_dfs, signal_to_use.upper())
        except Exception as e:
            logging.exception("confirm_trend failed safely: %s", e)
            trend_ok = False
        if not trend_ok:
            _inc_event("trend_no_confirm")
            log_decision(symbol, "trend_no_confirm")
            continue

        # === TRAILING STOP + ДИНАМИЧЕСКИЙ SL/TP (NEW) ===
        entry_reason = "fallback_confirmed" if use_fallback else "model_confirmed"
        if data_mode == "reduced":
            entry_reason = "reduced_data_ok"
        ctx |= {
            "rsi": float(rsi_val),
            "adx": float(adx_val),
            "pattern": pattern_info["pattern_name"],
            "pattern_confidence": pattern_info["confidence"],
            "trend": trend_state,
            "used_fallback": use_fallback,
            "reason": entry_reason,
            "market_category": market_category,
        }
        balance_info = safe_fetch_balance(exchange, {"type": "future"})
        equity = float((balance_info.get("total") or {}).get("USDT", 0.0))
        bar_index = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
        if not limiter.can_trade(symbol, equity):
            _inc_event("daily_loss_limit")
            log_decision(symbol, "daily_loss_limit")
            continue
        if not cool.can_trade(symbol, bar_index):
            _inc_event("cool_down")
            log_decision(symbol, "cool_down")
            continue
        if not best_entry_moment(
            symbol,
            signal_to_use,
            source="fallback" if use_fallback else "model",
            mode=mode.upper(),
            confidence=confidence,
            adx=adx_val,
            trend_ok=trend_ok,
        ):
            if use_fallback and ALLOW_FALLBACK_ENTRY:
                logging.info(
                    f"entry | {symbol} | fallback override - entering despite timing"
                )
            else:
                log_decision(symbol, "entry_timing")
                continue
        if len(open_trade_ctx) >= MAX_OPEN_TRADES:
            _inc_event("open_trades_limit")
            log_decision(symbol, "open_trades_limit")
            continue
        logging.info(
            f"entry | {symbol} | {signal_to_use} reason={entry_reason}"
        )
        entry_risk_factor = (0.5 if use_fallback else 1.0) * soft_risk * vol_risk
        price_hint = None
        if not multi_df.empty and "close_15m" in multi_df.columns:
            try:
                price_hint = float(multi_df["close_15m"].iloc[-1])
            except (TypeError, ValueError):
                price_hint = None
        if (price_hint is None or price_hint <= 0) and "close" in df_trend.columns:
            try:
                price_hint = float(df_trend["close"].iloc[-1])
            except (TypeError, ValueError):
                price_hint = None

        success = run_trade(
            symbol,
            signal_to_use,
            df_trend,
            stats,
            state,
            ctx,
            sl_mult=mode_params.get("sl_mult", 2.0),
            tp_mult=mode_params.get("tp_mult", 4.0),
            trailing_start=mode_params.get("trailing_start"),
            partial_tp=mode_params.get("partial_tp"),
            partial_tp_mult=mode_params.get("partial_tp_mult"),
            leverage=mode_lev,
            risk_factor=entry_risk_factor,
        )
        if not success:
            logging.info(
                "entry | %s | run_trade failed, attempting simplified market entry",
                symbol,
            )
            success = attempt_direct_market_entry(
                symbol,
                signal_to_use,
                ctx=ctx,
                df_trend=df_trend,
                multi_df=multi_df,
                mode_params=mode_params,
                risk_factor=entry_risk_factor,
                atr_value=atr_val,
                leverage=mode_lev,
                price_hint=price_hint,
            )
        current_bar = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
        if success:
            recent_hits.append(True)
            recent_trade_times.append(datetime.now(timezone.utc))
            fallback_cooldown[symbol] = current_bar + 2
            if not use_fallback:
                hard_entries += 1
            emit_summary(symbol, "entry")
        else:
            recent_hits.append(False)
            emit_summary(symbol, "entry_failed")
        closed_orders = safe_fetch_closed_orders(exchange, symbol, limit=5)
        for order in reversed(closed_orders):
            oid = str(order.get("id") or "")
            # [ANCHOR:PROCESSED_ORDERS_CACHE_USE]
            if oid in _processed_order_ids:
                continue
            commission = _extract_commission(order)
            if log_exit_from_order(symbol, order, commission, trade_log_path):
                _processed_order_ids.add(oid)
                rebuild_reports_on_exit()
                try:
                    save_risk_state(risk_state, limiter, cool, stats)
                except Exception as e:
                    logging.exception("save_risk_state failed: %s", e)

        # Trade log entries are written on every position close via
        # the ``log_trade`` helper above.
    if hard_entries == 0:
        current_bar = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
        for symbol in symbols:
            if fallback_cooldown.get(symbol, 0) > current_bar:
                continue
            positions = fetch_positions_soft(symbol)
            if any(float(p.get("contracts", 0)) > 0 for p in positions):
                continue
            if len(open_trade_ctx) >= MAX_OPEN_TRADES:
                _inc_event("open_trades_limit")
                log_decision(symbol, "open_trades_limit")
                continue
            multi_df = fetch_multi_ohlcv(symbol, timeframes, limit=300, warn=False)
            if multi_df is None or multi_df.empty:
                continue
            df_trend = fetch_ohlcv(symbol, "1h", limit=250)
            if df_trend is None or df_trend.empty:
                continue
            if "adx" not in df_trend.columns:
                df_trend["adx"] = 0.0
            adx_val = float(df_trend["adx"].iloc[-1])

            for lag in range(1, 7):
                multi_df[f"close_lag{lag}"] = multi_df["close_15m"].shift(lag)
                multi_df[f"volume_lag{lag}"] = multi_df["volume_15m"].shift(lag)
            multi_df = add_custom_features(multi_df)
            multi_df["ticker_cat"] = SYMBOL_CATEGORIES.get(symbol, -1)
            multi_df["pattern_name"] = PATTERN_LABELS.get("none", -1)
            multi_df["pattern_source"] = PATTERN_SOURCE_MAP.get("none", 0)
            multi_df["pattern_confidence"] = 0.0
            X_last = multi_df.iloc[[-1]]
            returns_1h = 0.0
            if "close_1h" in multi_df.columns:
                closes = multi_df["close_1h"].dropna()
                if len(closes) >= 2:
                    returns_1h = (closes.iloc[-1] / closes.iloc[-2]) - 1.0
            rsi_cross_from_extreme = None
            rsi_val = 0.0
            if "rsi" in df_trend.columns:
                rsi_series = df_trend["rsi"].dropna()
                if len(rsi_series) >= 2:
                    rsi_prev, rsi_val = rsi_series.iloc[-2], rsi_series.iloc[-1]
                    if rsi_prev > 70 >= rsi_val:
                        rsi_cross_from_extreme = "short"
                    elif rsi_prev < 30 <= rsi_val:
                        rsi_cross_from_extreme = "long"
            model_signal, confidence = predict_signal(
                symbol, X_last, adx_val, bool(rsi_cross_from_extreme), returns_1h
            )
            if model_signal == "hold":
                continue
            tf_signals = {
                tf: timeframe_direction(multi_df, tf) for tf in ["5m", "15m", "30m"]
            }
            tf_matches = sum(s == model_signal for s in tf_signals.values())
            if tf_matches < 2:
                continue
            global_dir = timeframe_direction(multi_df, "4h")
            if global_dir not in (model_signal, "hold"):
                continue
            trend_supported = tf_matches >= 2 and global_dir == model_signal
            pattern_supports_signal = (
                pattern_dir in (model_signal, "both")
                if pattern_dir is not None
                else False
            )
            required_confidence = apply_pattern_proba_bonus(
                0.5,
                pattern_info["confidence"],
                trend_supported and pattern_supports_signal,
            )
            # PATCH NOTES:
            # - применили бонус паттерна в fallback-фильтре и перенесли проверку после трендов.
            # - безопасно: удерживаем cap MIN_PROBA_FILTER и прежние фильтры направлений.
            # - критерии: сильный паттерн (>=0.7) + тренд снижает порог, слабый/hold остаётся без изменений.
            if confidence < required_confidence:
                continue
            vol_ratio_fb = safe_vol_ratio(
                df_trend.get("volume"), VOL_WINDOW, key=f"{symbol}_fb"
            )
            vol_reason_fb = volume_reason(
                df_trend.get("volume"), VOLUME_RATIO_MIN, VOL_WINDOW
            )
            vol_risk_fb = 0.5 if vol_reason_fb == "vol_missing" else 1.0
            trend_state = determine_trend(df_trend)
            fallback_leverage = LEVERAGE if is_derivative_market else 1
            if is_derivative_market:
                leverage_category = detect_market_category(exchange, symbol)
                if str(leverage_category).lower() == "linear":
                    safe_set_leverage(exchange, symbol, fallback_leverage)
                else:
                    logging.info(
                        "leverage | %s | skip: market %s not linear",
                        symbol,
                        leverage_category or "unknown",
                    )
            else:
                logging.debug(
                    "leverage | %s | Skipping leverage setup for %s market",
                    symbol,
                    market_category or "unknown",
                )
            ctx = {
                "rsi": float(rsi_val),
                "adx": float(adx_val),
                "pattern": "none",
                "pattern_confidence": 0.0,
                "trend": trend_state,
                "used_fallback": True,
                "reason": "idle_fallback",
                "vol_ratio": vol_ratio_fb,
                "vol_reason": vol_reason_fb,
                "market_category": market_category,
            }
            success_fb = run_trade(
                symbol,
                model_signal,
                df_trend,
                stats,
                risk_state.get(symbol, PairState()),
                ctx,
                sl_mult=1.5,
                tp_mult=3.0,
                risk_factor=0.5 * soft_risk * vol_risk_fb,
                leverage=fallback_leverage,
            )
            if success_fb:
                recent_hits.append(True)
                recent_trade_times.append(datetime.now(timezone.utc))
                fallback_cooldown[symbol] = current_bar + 2

    if skip_summary["model"] or skip_summary["data"]:
        parts: list[str] = []
        if skip_summary["model"]:
            parts.append("model:" + ",".join(skip_summary["model"]))
        if skip_summary["data"]:
            parts.append("data:" + ",".join(skip_summary["data"]))
        summary_text = "skip summary | " + "; ".join(parts)
        logging.warning(summary_text)
        token = os.environ.get("TG_TOKEN")
        chat_id = os.environ.get("TG_CHAT")
        if token and chat_id:
            try:  # pragma: no cover - network alert
                from monitoring import send_telegram_alert

                send_telegram_alert(summary_text, token, chat_id)
            except Exception as e:
                logging.error("alert failed: %s", e)

    # flush any remaining statuses for this cycle
    update_dynamic_thresholds()
    flush_cycle_logs()

    # Safety check: capture any closes that happened after per-symbol processing
    # [ANCHOR:SAFETY_SWEEP_END_OF_CYCLE]
    sweep_closed_any = False
    for symbol in symbols:
        closed_orders = safe_fetch_closed_orders(exchange, symbol, limit=10)

        reverse_done = False  # не открывать больше одного реверса на символ за проход

        # от новых к старым
        for order in reversed(closed_orders):
            oid = str(order.get("id") or "")
            # [ANCHOR:PROCESSED_ORDERS_CACHE_USE]
            if oid in _processed_order_ids:
                continue
            commission = _extract_commission(order)
            ctx_info = open_trade_ctx.get(symbol, {})
            trail_active = bool(ctx_info.get("trailing_profit_used"))
            exit_hint = str(ctx_info.get("exit_type_hint") or "").upper()

            # 1) всегда сперва логируем выход (идемпотентно внутри функции)
            handled = log_exit_from_order(symbol, order, commission, trade_log_path)
            if not handled:
                continue
            sweep_closed_any = True
            _processed_order_ids.add(oid)
            rebuild_reports_on_exit()

            exit_type = "MANUAL"
            otype = (order.get("type") or order.get("info", {}).get("type") or "").upper()
            if otype == "STOP_MARKET":
                exit_type = "TRAIL_STOP" if trail_active else "SL"
            elif otype in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
                exit_type = "TP"
            elif exit_hint == "TIME":
                exit_type = "TIME"
            elif exit_hint == "TP":
                exit_type = "TP"

            # 2) опциональный реверс ТОЛЬКО после SL и только один раз
            if not reverse_done and exit_type == "SL":
                prev_side = _normalize_prev_side(order)
                if prev_side in ("LONG", "SHORT"):
                    opposite_side = "SHORT" if prev_side == "LONG" else "LONG"

                    # confirm_trend на 5m/15m/30m/1h (+4h если доступен)
                    dataframes = {tf: fetch_ohlcv(symbol, tf, limit=100) for tf in ("5m", "15m", "30m", "1h")}
                    df_4h = fetch_ohlcv(symbol, "4h", limit=100)
                    if df_4h is not None and not getattr(df_4h, "empty", False):
                        dataframes["4h"] = df_4h

                    # [ANCHOR:TREND_SAFE_CALLER]
                    try:
                        ok = confirm_trend(dataframes, opposite_side)
                    except Exception as e:
                        logging.exception("confirm_trend failed safely: %s", e)
                        ok = False
                    bar_index = int(datetime.now(timezone.utc).timestamp() // (5 * 60))
                    equity = float(stats.stats.get(symbol, {}).get("equity", 0.0))

                    if ok and limiter.can_trade(symbol, equity) and cool.can_trade(symbol, bar_index):
                        open_reverse_position_with_reduced_risk(
                            symbol,
                            opposite_side,
                            df_trend=dataframes.get("1h"),
                            risk_multiplier=soft_risk,
                        )
                        reverse_done = True

            # 3) после любого успешно залогированного закрытия — сохраняем состояние риска
            try:
                save_risk_state(risk_state, limiter, cool, stats)
            except Exception as e:
                logging.exception("save_risk_state failed: %s", e)

    # [ANCHOR:SAFETY_SWEEP_SAVE]
    if sweep_closed_any:
        try:
            save_risk_state(risk_state, limiter, cool, stats)
        except Exception as e:
            logging.exception("save_risk_state failed: %s", e)

    rebuild_reports_on_exit()

def run_bot_loop():
    logging.info("=== LIVE BOT START (TESTNET) ===")
    logging.info("[run_bot_loop] Loop starting...")
    last_analysis = time.time()
    analysis_interval = 60 * 60 * 3  # 3 hours
    try:
        while True:
            ensure_model_loaded(ADAPTER, symbols)
            try:
                run_bot()
            except Exception as e:
                logging.error(f"Unhandled error in bot loop: {e}", exc_info=True)
            global _cycle_counter
            _cycle_counter += 1
            if _cycle_counter % SUMMARY_CYCLES == 0:
                _summarize_events()
            if _cycle_counter % PATTERN_HIT_LOG_CYCLES == 0 and pattern_hit_rates:
                for sym, rate in pattern_hit_rates.items():
                    logging.info("pattern_hit_rate | %s | %.2f", sym, rate)
            if time.time() - last_analysis >= analysis_interval:
                logging.info("[run_bot_loop] Running scheduled trade analysis...")
                try:
                    run_trade_analysis(n_trials=100)
                    for sym, params in best_params_cache.items():
                        if sym == "GLOBAL":
                            continue
                        apply_params(params)
                except Exception as e:
                    logging.error(f"Trade analysis error: {e}", exc_info=True)
                last_analysis = time.time()
            logging.info("[LOOP] Sleeping 5 minutes...")
            time.sleep(60 * 5)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_trade_analysis(n_trials=trials)
        return

    logging.info("=== INITIALIZING BOT ===")
    logging.info("Timestamp: %s", datetime.now(timezone.utc))
    logging.info("Symbols: %s", symbols)
    sandbox_active = bool(SANDBOX_MODE and getattr(ADAPTER, "sandbox", False))
    if sandbox_active:
        logging.info("Exchange sandbox mode: ENABLED")
    else:
        logging.info("Exchange sandbox mode: DISABLED")

    if not ADAPTER_READY:
        logging.error(
            "startup | exchange adapter unavailable; aborting sandbox run"
        )
        raise SystemExit(
            "Exchange adapter unavailable; check sandbox connectivity or API keys."
        )

    if not BUNDLE_PATH.exists():
        logging.warning(
            "model | global model missing at startup: %s; retraining", BUNDLE_PATH
        )
        try:
            df_features, df_target, feature_cols = fetch_and_prepare_training_data(
                ADAPTER, symbols
            )
            _retrain_checked(df_features, df_target, feature_cols)
        except Exception as e:
            logging.error(f"retrain | global | initial retrain failed: {e}")

    _maybe_retrain_global()

    # Clean up any leftover open orders at startup but keep those linked to active positions
    for sym in symbols:
        cancel_stale_orders(sym)

    run_bot_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception:
        logging.exception("fatal: unhandled")
        raise
