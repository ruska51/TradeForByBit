import pandas as pd
from datetime import datetime, timedelta, timezone

import pytest

import main


def test_calm_market_replay(tmp_path):
    log_path = tmp_path / "trades_log.csv"
    # simulate 3 hours of inactivity
    main.symbol_activity["BTC/USDT"] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    # adaptation should lower ADX threshold
    result = main.adjust_filters_for_inactivity("BTC/USDT")
    adj_adx = result[1]
    assert adj_adx <= main.ADX_THRESHOLD
    cycles = int((3 * 60) / main.BOT_CYCLE_MINUTES)
    steps = max(cycles // max(main.INACTIVITY_ADAPT_CYCLES, 1), 0)
    expected_adx = max(
        float(main.MIN_ADX_THRESHOLD),
        float(main.ADX_THRESHOLD) - steps * float(main.INACTIVITY_ADX_STEP),
    )
    assert adj_adx == pytest.approx(expected_adx)
    # simulate trade made possible due to adaptation when the threshold relaxed
    if adj_adx < main.ADX_THRESHOLD:
        adx_val = adj_adx + 0.1
        assert adx_val < main.ADX_THRESHOLD
        main.log_trade(
            datetime.now(timezone.utc),
            "BTC/USDT",
            "long",
            100,
            101,
            1,
            1,
            "TEST",
            str(log_path),
        )
        assert log_path.exists()
        df = pd.read_csv(log_path)
        assert len(df) >= 1
    else:
        assert not log_path.exists()


def test_conditional_entry_activation():
    # simulate 7 hours of inactivity to trigger conditional mode
    main.symbol_activity["ETH/USDT"] = (
        datetime.now(timezone.utc) - timedelta(hours=7)
    ).isoformat()
    (
        proba,
        adx,
        allow_cond,
        _,
        _,
        adj_rsi_ob,
        adj_rsi_os,
    ) = main.adjust_filters_for_inactivity("ETH/USDT")
    assert allow_cond
    cycles = int((7 * 60) / main.BOT_CYCLE_MINUTES)
    steps = max(cycles // max(main.INACTIVITY_ADAPT_CYCLES, 1), 0)
    expected_proba = max(
        float(main.MIN_PROBA_FILTER),
        float(main.PROBA_FILTER) - steps * float(main.INACTIVITY_PROBA_STEP),
    )
    expected_adx = max(
        float(main.MIN_ADX_THRESHOLD),
        float(main.ADX_THRESHOLD) - steps * float(main.INACTIVITY_ADX_STEP),
    )
    expected_rsi_ob = min(
        float(main.RSI_OVERBOUGHT_MAX),
        float(main.RSI_OVERBOUGHT) + steps * float(main.INACTIVITY_RSI_STEP),
    )
    expected_rsi_os = max(
        float(main.RSI_OVERSOLD_MIN),
        float(main.RSI_OVERSOLD) - steps * float(main.INACTIVITY_RSI_STEP),
    )

    assert proba == pytest.approx(expected_proba)
    assert adx == pytest.approx(expected_adx)
    assert adj_rsi_ob == pytest.approx(expected_rsi_ob)
    assert adj_rsi_os == pytest.approx(expected_rsi_os)
