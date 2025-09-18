from datetime import datetime, timezone

import pandas as pd

import main


def _setup_env(tmp_path, monkeypatch, *, confirm_trend=True, limiter_ok=True, cool_ok=True):
    symbol = "BTC/USDT"
    log_path = tmp_path / "trades.csv"

    # basic risk config
    main.risk_config["trades_path"] = str(log_path)
    main.ENABLE_REPORTS_BUILDER = False

    # global state
    main.symbols = [symbol]
    main.reserve_symbols = []
    main.risk_state = {}
    main.open_trade_ctx.clear()
    main._processed_order_ids.clear()
    main.open_trade_ctx[symbol] = {
        "symbol": symbol,
        "side": "LONG",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }

    order = {
        "id": "1",
        "type": "STOP_MARKET",
        "avgPrice": 95.0,
        "info": {"prev_side": "LONG"},
    }

    class DummyExchange:
        def fetch_closed_orders(self, sym, since=None, limit=10):
            return [order]

    main.exchange = DummyExchange()

    class DummyAdapter:
        def fetch_open_orders(self, symbol=None):
            return (0, [])

        def cancel_open_orders(self, symbol):
            return (0, [])

        def fetch_ohlcv(self, *a, **k):
            return [[0, 1, 1, 1, 1, 1]]

        def load_markets(self):
            return None

    main.ADAPTER = DummyAdapter()

    class DummyModel:
        classes_ = [0, 1, 2]

        def predict(self, X):
            return [0] * len(X)

    class DummyScaler:
        def transform(self, X):
            return X

    monkeypatch.setattr(main, "GLOBAL_MODEL", DummyModel())
    monkeypatch.setattr(main, "GLOBAL_SCALER", DummyScaler())
    monkeypatch.setattr(main, "GLOBAL_FEATURES", ["close_15m"])
    monkeypatch.setattr(main, "_maybe_retrain_global", lambda: None)

    class DummyStats:
        def __init__(self):
            self.stats = {symbol: {"equity": 0.0}}

        def is_banned(self, sym):
            return False

        def pop_soft_risk(self, sym):
            return 0

    main.stats = DummyStats()

    class DummyLimiter:
        def can_trade(self, sym, equity):
            return limiter_ok

    class DummyCool:
        def can_trade(self, sym, bar):
            return cool_ok

    main.limiter = DummyLimiter()
    main.cool = DummyCool()

    # lightweight helpers
    monkeypatch.setattr(main, "update_pair_stats", lambda *a, **k: {})
    monkeypatch.setattr(main, "adjust_state_by_stats", lambda state, stats_dict, cfg: state)
    monkeypatch.setattr(main, "save_pair_report", lambda *a, **k: None)
    monkeypatch.setattr(main, "save_risk_state", lambda *a, **k: None)
    monkeypatch.setattr(main, "update_dynamic_thresholds", lambda: None)
    monkeypatch.setattr(main, "flush_cycle_logs", lambda: None)
    monkeypatch.setattr(main, "record_error", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(
        main,
        "get_symbol_params",
        lambda s: main.StrategyParams.from_dict(main.DEFAULT_PARAMS),
    )
    monkeypatch.setattr(main, "apply_params", lambda params: None)
    monkeypatch.setattr(main, "backtest", lambda s: {"mode": "unavailable"})
    monkeypatch.setattr(main, "fetch_ohlcv", lambda *a, **k: pd.DataFrame({"close": [1, 2, 3]}))
    monkeypatch.setattr(main, "confirm_trend", lambda *a, **k: confirm_trend)

    reverse_calls: list = []
    monkeypatch.setattr(
        main,
        "open_reverse_position_with_reduced_risk",
        lambda *a, **k: reverse_calls.append((a, k)),
    )

    return symbol, log_path, reverse_calls


def test_safety_sweep_logs_once(tmp_path, monkeypatch):
    symbol, log_path, reverse_calls = _setup_env(
        tmp_path, monkeypatch, confirm_trend=False
    )

    main.run_bot()
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df["exit_type"].iloc[0] == "SL"
    assert reverse_calls == []

    # second run: no duplicates
    main.run_bot()
    df2 = pd.read_csv(log_path)
    assert len(df2) == 1


def test_safety_sweep_triggers_reverse(tmp_path, monkeypatch):
    symbol, log_path, reverse_calls = _setup_env(
        tmp_path, monkeypatch, confirm_trend=True
    )

    main.run_bot()
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert len(reverse_calls) == 1
    # side of reverse is opposite
    assert reverse_calls[0][0][1] == "SHORT"

    # subsequent run should not trigger another reverse
    main.run_bot()
    assert len(reverse_calls) == 1

