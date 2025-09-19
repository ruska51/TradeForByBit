# Refactoring Guide

This document describes suggested improvements for the trading bot located in `main.py`. Each section corresponds to the numbered requirement.

## 1. Split training and live trading

- **Idea**: Models should be trained offline on a schedule. Live trading only loads saved models.
- **Example**

```python
# training_cli.py
# run once a day or manually
from bot import train_model
for symbol in SYMBOLS:
    model = train_model(symbol)
```

```python
# live_trading.py
from bot import load_model, predict_signal
MODEL_CACHE = {s: load_model(s) for s in SYMBOLS}
async def trading_cycle():
    for symbol, model in MODEL_CACHE.items():
        signal = predict_signal(symbol, model)
        # execute trade
```

Hold-out testing should use future data only:

```python
train, test = df[df.date < split_date], df[df.date >= split_date]
```

## 2. Explicit parameter containers

Store per-pair parameters in a dataclass and pass them explicitly.

```python
from dataclasses import dataclass
@dataclass
class PairParams:
    sl_pct: float
    tp_pct: float
    threshold: float
    proba_filter: float

PAIR_CONFIG: dict[str, PairParams] = {
    "BTC/USDT": PairParams(0.02, 0.04, 0.001, 0.6),
}
```

Functions then accept `params: PairParams` instead of using globals.

## 3. Fallback trading rules

Add simple rule-based logic if the ML model produces only `hold` signals for too long.

```python
async def rule_based_entry(df) -> str:
    if df.ema_fast.iloc[-1] > df.ema_slow.iloc[-1]:
        return "long"
    elif df.ema_fast.iloc[-1] < df.ema_slow.iloc[-1]:
        return "short"
    return "hold"

async def get_signal(symbol, model, params, last_trade):
    signal = predict_signal(symbol, model, params.proba_filter)
    if signal == "hold" and (datetime.utcnow() - last_trade).total_seconds() > 3600:
        signal = await rule_based_entry(fetch_ohlcv(symbol, "15m", 50))
    return signal
```

## 4. Restricted retraining

Keep track of the last training timestamp and performance metrics.

```python
TRAIN_LOG: dict[str, datetime] = {}

async def maybe_retrain(symbol, equity_curve):
    if datetime.utcnow() - TRAIN_LOG.get(symbol, datetime.min) < timedelta(days=1):
        return
    old_metrics = evaluate_model(load_model(symbol))
    new_model = train_model(symbol)
    new_metrics = evaluate_model(new_model)
    if new_metrics.sharpe > old_metrics.sharpe and new_metrics.drawdown < old_metrics.drawdown:
        save_model(symbol, new_model)
        TRAIN_LOG[symbol] = datetime.utcnow()
```

## 5. Additional backtest metrics

Use libraries such as `pyfolio` or custom calculations for Sharpe and drawdown.

```python
def backtest_metrics(equity: pd.Series) -> dict:
    returns = equity.pct_change(fill_method=None).dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).min()
    return {"sharpe": sharpe, "max_drawdown": drawdown}
```

## 6. Async exchange and data fetch

Switch to `ccxt.pro` to use websockets and asyncio.

```python
import ccxt.pro as ccxt
import asyncio

exchange = ccxt.bybit({'apiKey': API_KEY, 'secret': API_SECRET, 'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}})

async def fetch_ohlcv_async(symbol, tf):
    return await exchange.watch_ohlcv(symbol, tf)

async def place_order_async(symbol, side, amount, price=None):
    params = {"type": "future"}
    return await exchange.create_order(symbol, 'market', side, amount, price, params)
```

## 7. Advanced trade logging

Log decision reasons and store everything in CSV or a lightweight database.

```python
import csv

def log_decision(symbol: str, reason: str, path='decision_log.csv'):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'symbol', 'reason'])
        writer.writerow([datetime.utcnow(), symbol, reason])
```

## 8. Monitoring and alerts

Send messages to Telegram on errors or drawdown and serve a small dashboard with FastAPI.

```python
from fastapi import FastAPI
import requests

TELEGRAM_URL = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'

async def send_alert(text: str):
    requests.post(TELEGRAM_URL, json={'chat_id': TG_CHAT, 'text': text})

app = FastAPI()

@app.get('/equity')
async def equity_curve():
    df = pd.read_csv('equity_log.csv')
    return df.to_dict('records')
```

## 9. Parameter profiles

Support predefined conservative/aggressive profiles.

```python
PROFILES = {
    "conservative": PairParams(0.015, 0.03, 0.002, 0.65),
    "aggressive": PairParams(0.03, 0.06, 0.0005, 0.5),
}

def load_profile(name: str) -> PairParams:
    return PROFILES[name]
```

Optimization results should be stored in `params_history.json` for later inspection.

## 10. Style improvements

Add docstrings and type hints to all public functions. Keep modules small and focused.

```python
def predict_signal(symbol: str, model) -> str:
    """Return 'long', 'short' or 'hold' signal for a symbol."""
    ...
```

Document architecture in `README.md` and keep functions under 50 lines when possible.
