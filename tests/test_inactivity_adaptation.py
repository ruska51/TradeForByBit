from datetime import datetime, timedelta, timezone

import main
from utils.csv_utils import read_csv_safe


def test_calm_market_replay(tmp_path):
    log_path = tmp_path / "trades_log.csv"
    symbol = "BTC/USDT"
    main.symbol_activity[symbol] = datetime.now(timezone.utc) - timedelta(hours=3)
    main.reset_inactivity(symbol)
    adjustments = None
    for _ in range(main.INACTIVITY_RELAX_CYCLES):
        adjustments = main.adjust_filters_for_inactivity(symbol)
    assert adjustments is not None
    (
        _proba,
        adj_adx,
        _allow_cond,
        _fb_mode,
        inactivity_hours,
        rsi_over,
        rsi_under,
    ) = adjustments
    assert inactivity_hours >= 0
    assert adj_adx <= main.ADX_THRESHOLD + 1e-9
    assert rsi_over >= main.RSI_OVERBOUGHT
    assert rsi_under <= main.RSI_OVERSOLD
    assert main.symbol_relax_steps[symbol] >= 1
    main.log_trade(
        datetime.now(timezone.utc), symbol, "long", 100, 101, 1, 1, "TEST", str(log_path)
    )
    df = read_csv_safe(log_path)
    assert len(df) >= 1
    assert main.symbol_inactivity_cycles[symbol] == 0


def test_conditional_entry_activation():
    symbol = "ETH/USDT"
    main.symbol_activity[symbol] = datetime.now(timezone.utc) - timedelta(hours=7)
    main.reset_inactivity(symbol)
    adjustments = None
    for _ in range(main.INACTIVITY_RELAX_CYCLES):
        adjustments = main.adjust_filters_for_inactivity(symbol)
    assert adjustments is not None
    proba, _adx, allow_cond, fb_mode, inactivity, rsi_over, rsi_under = adjustments
    assert allow_cond
    assert fb_mode == (main.FALLBACK_MODE_ENABLED or inactivity >= main.INACTIVITY_FALLBACK_HOURS)
    assert proba <= main.PROBA_FILTER + 1e-9
    assert rsi_over >= main.RSI_OVERBOUGHT
    assert rsi_under <= main.RSI_OVERSOLD
