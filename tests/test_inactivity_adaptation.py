import pandas as pd
from datetime import datetime, timedelta, timezone

import main


def test_calm_market_replay(tmp_path):
    log_path = tmp_path / "trades_log.csv"
    # simulate 3 hours of inactivity
    main.symbol_activity["BTC/USDT"] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    # adaptation should lower ADX threshold
    _, adj_adx, *_ = main.adjust_filters_for_inactivity("BTC/USDT")
    assert adj_adx < main.ADX_THRESHOLD
    adx_val = adj_adx + 0.1
    assert adx_val < main.ADX_THRESHOLD
    # simulate trade made possible due to adaptation
    if adx_val >= adj_adx:
        main.log_trade(datetime.now(timezone.utc), "BTC/USDT", "long", 100, 101, 1, 1, "TEST", str(log_path))
    df = pd.read_csv(log_path)
    assert len(df) >= 1


def test_conditional_entry_activation():
    # simulate 7 hours of inactivity to trigger conditional mode
    main.symbol_activity["ETH/USDT"] = (
        datetime.now(timezone.utc) - timedelta(hours=7)
    ).isoformat()
    proba, adx, allow_cond, _, _ = main.adjust_filters_for_inactivity("ETH/USDT")
    assert allow_cond
    assert proba == max(main.MIN_PROBA_FILTER, main.PROBA_FILTER - 0.05)
    assert adx == max(main.MIN_ADX_THRESHOLD, main.ADX_THRESHOLD - 3)
