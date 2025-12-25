# TradeForByBit: Project Specification & Instructions

## 🛠 Build & Run Commands
- **Install Dependencies:** `pip install -r requirements.txt` (Main: ccxt, pandas, sklearn, xgboost, optuna)
- **Run Bot:** `python main.py` (Default: trading loop)
- **Run Analysis:** `python main.py analyze <repeats>` (Backtest/optimization mode)
- **Environment Variables:**
  - Filters: `PROBA_FILTER` (default 0.55), `ADX_THRESHOLD` (12), `RSI_OVERBOUGHT` (80), `RSI_OVERSOLD` (20)
  - Exchange: `LEVERAGE` (default 20x), `EXCHANGE_BACKEND` (Bybit/Binance)

## 🏗 Project Architecture
- `main.py`: Entry point. Entry/Exit loop, model loading, order management.
- `exchange_adapter.py`: Interface for Bybit (CCXT V5) & Binance. Handles Sandbox/Mainnet.
- `data_prep.py`: OHLCV fetching, technical indicators (RSI, ADX, CCI, EMA), feature engineering.
- `model_utils.py`: ML model lifecycle (ExtraTreesClassifier/XGBoost), loading/saving, scaling.
- `risk_management.py`: Dynamic position sizing, winrate tracking, loss limits.
- `logging_utils.py`: CSV logging (`trades_log.csv`, `decision_log.csv`, `profit_report.csv`).
- `fallback.py`: Secondary EMA-based strategy when ML signal is weak or absent.
- `profiles.py`: Presets for `conservative` or `aggressive` trading modes.

## 📝 Code Style & Standards
- **Typing:** Strict use of Python Type Hints for all functions. 
  *Example:* `def predict_signal(symbol: str, data: pd.DataFrame) -> tuple[str, float]:`
- **Async:** Use `asyncio` / `ccxt.pro` for non-blocking API calls (refer to `async_utils.py`).
- **Logging:** Use `logging` module ONLY. NO `print()` statements after initialization.
- **Error Handling:** Wrap ALL exchange API calls in `try/except` blocks with detailed logging.
- **Modularity:** Keep logic separated. No business logic in `main.py` that belongs in `data_prep` or `risk_management`.

## 📈 Strategy Logic Flow
1. **Data:** Fetch multi-timeframe candles -> Calculate indicators in `data_prep.py`.
2. **Signal:** ML Model `predict_signal` returns (Direction, Confidence).
3. **Filter:** Apply `PROBA_FILTER`, `ADX_THRESHOLD`, and `RSI` checks.
4. **Execution:** If filters pass, execute Market Order. If filters fail + timeout, use `fallback.py`.
5. **Protection:** Immediately set TP/SL with `reduceOnly=True` and `closeOnTrigger=True`.

## ⚠️ Critical Constraints
- **Bybit V5:** Use specific flags for protective orders.
- **Risk:** Default `max_trade_loss_pct` is 0.1. `max_open_trades` is 3.
- **Normalization:** Symbols must be `XXX/USDT` format.