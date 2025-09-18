## Trading Bot Overview

This repository contains a single script `main.py` implementing an automated trading bot. The bot fetches data from multiple timeframes, trains a machine-learning model for each symbol and places orders via the exchange API.

Run the bot from the command line using:

```bash
python main.py
```

Starting it by double-clicking the file will not execute the `__main__` section and the logger will not be initialised.

All runtime messages are written both to the console and to log files under
`logs/` (`console.log` with an accompanying `console_errors.log`).

## Exchange backend

The bot works with both [ccxt](https://github.com/ccxt/ccxt) and
[python-binance](https://github.com/binance/binance-connector-python) via a
unified ``ExchangeAdapter``. The backend is selected automatically or forced
through the ``EXCHANGE_BACKEND`` environment variable
(``auto``/``ccxt``/``binance_sdk``). In ``auto`` mode python-binance is used if
installed, otherwise the adapter falls back to ccxt and logs one warning.

Symbols are normalised to ccxt style (``ETH/USDT``) and testnet support is
available for both backends. The ``python-binance`` dependency is optional – the
project functions with ccxt alone.

## Improving trade coverage

In practice some instruments may remain inactive because the per-symbol model does not produce enough long/short signals. To diversify trading across all assets listed in `symbols`, consider:

1. **Fallback to a global model** – train a generic model on combined data or on a group of similar symbols. If an individual model yields negative returns or too few signals, the bot can use this global model instead.
2. **Dynamic threshold tuning** – gradually lower `PROBA_FILTER` or relax ADX/RSI restrictions when a symbol sees no trades for a prolonged period.
3. **Activity window** – track the number of trades per symbol over the last `N` days. If an asset is "dead", temporarily accept signals with weaker confirmation so it can re-enter the rotation.
4. **Automatic re‑training** – the current `train_model` already adjusts `threshold` until at least 30 long/short samples appear. Keeping this logic ensures each model adapts to recent volatility.

### Runtime threshold overrides

`PROBA_FILTER`, `ADX_THRESHOLD` and the RSI bounds can be changed without
modifying the source code. Define environment variables before launching
`main.py`:

```bash
export PROBA_FILTER=0.55
export ADX_THRESHOLD=12
export RSI_OVERBOUGHT=80
export RSI_OVERSOLD=20
python main.py
```

When using predefined profiles, the same parameters may be overridden in code:

```python
from profiles import load_profile
params = load_profile("conservative", {"PROBA_FILTER": 0.6})
```

These hooks make it easy to temporarily relax filters for symbols that have
seen little activity.

## Checking log messages

The bot outputs detailed messages explaining why trades are skipped. Typical examples:

```text
⏸ BTC/USDT: ADX too low (9.5) → skipping trade
⏸ ETH/USDT: RSI 72.0 overbought → skip long
⏸ XRP/USDT: model signal is HOLD → not entering trade
```

Search the log file or console output for phrases like "ADX too low", "RSI", or "model signal is HOLD" to determine why a symbol has no active positions. These messages help fine‑tune thresholds and verify that the bot attempts to trade every asset.
\nSee `docs/refactor_guidelines.md` for a checklist of design improvements.

Additional helper scripts:

- `train_models.py` – run scheduled model training outside the live bot
- Hyperparameters for an `ExtraTreesClassifier` are tuned with [Optuna](https://optuna.org).
- `async_utils.py` – examples of asynchronous data fetching with ccxt.pro
- `logging_utils.py` – CSV logger for trade decisions
- `metrics_utils.py` – helpers to compute Sharpe and drawdowns
- `fallback.py` – simple EMA crossover when ML holds
- `monitoring.py` – Telegram alerts and FastAPI dashboard
- `profiles.py` – predefined conservative/aggressive parameter sets
- `retrain_utils.py` – limits on model retraining frequency

## Risk management

The bot tracks performance of each trading pair in `trades_log.csv`. Based on the
recent win rate and consecutive losses it automatically scales the trade size.
Parameters such as the allowed loss per trade, lookback window and streak limits
are stored in `risk_config.json`. The file also defines `max_open_trades` to
limit how many positions can be active at once (default 3).
Metrics per pair are written to `pair_report.csv` every cycle so results can be
monitored outside the bot.
If the log file is missing it will be recreated automatically when the first
trade closes. You can also call `load_trades(create=True)` to generate an empty
file with the correct headers.

Each order must meet Binance's 10 USDT minimum notional requirement. If the
calculated position size is smaller but the account balance has at least 10 USDT
available, the bot will open a fixed 10 USDT trade instead of skipping. When a
larger order fails due to insufficient funds, the bot also retries with a
minimal 10 USDT position if the balance allows.

## Binance Futures order examples

When placing protective orders on Binance Futures, `reduceOnly` should only be
used for orders that explicitly close an existing position. A stop loss or take
profit placed right after entering a trade usually requires only the
`closePosition` flag:

```python
order_params = {"stopPrice": stop_price, "closePosition": True}
if is_manual_exit:
    order_params["reduceOnly"] = True

exchange.create_order(
    symbol, "TAKE_PROFIT_MARKET", side, amount, None, order_params
)
```

Equivalent logic in the bot code looks like:

```python
params = {"stopPrice": stop}
if closing_position:
    params["reduceOnly"] = True

exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)
```

If such orders are rejected, the bot falls back to a market order to close the
position and logs the error message.

## Leverage setting

The bot sets the same leverage for all symbols on startup. By default it uses
`20x`. To override this value, define the environment variable `LEVERAGE` before
launching `main.py`:

```bash
export LEVERAGE=10
python main.py
```

The maximum position size for each order is adjusted based on the current
leverage using Binance Futures leverage brackets.

## Trade mode detection

The bot categorizes each opportunity as **scalp**, **intraday**, or **swing**.
Mode selection uses a mix of ATR, recent volume and short‑term volatility to
approximate market activity. These heuristics help size positions more
consistently but they are *not* foolproof—rapid changes in volatility can still
lead to suboptimal mode choices. Always monitor results and adjust thresholds if
necessary.

## Lightweight Torch stubs

This repository includes minimal placeholder packages under `torch_stub/` and
`torchvision_stub/`. They provide just enough functionality for the unit tests
to run without installing PyTorch. When running the bot in a real environment,
make sure the actual `torch` and `torchvision` packages take precedence in
Python's import path. Keeping the stub directories renamed as shipped prevents
conflicts with the real libraries.
