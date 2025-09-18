import pandas as pd
import numpy as np

from risk_management import confirm_trend, maybe_invert_position


def _make_df(direction: str, n: int = 250) -> pd.DataFrame:
    if direction == 'up':
        close = pd.Series(np.arange(n, dtype=float))
    else:
        close = pd.Series(np.arange(n, 0, -1, dtype=float))
    high = close + 1
    low = close - 1
    return pd.DataFrame({'close': close, 'high': high, 'low': low})


def _make_low_adx_df(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = pd.Series(rng.normal(0, 0.05, n))
    high = close + 0.05
    low = close - 0.05
    return pd.DataFrame({'close': close, 'high': high, 'low': low})


def test_confirm_trend_long_short():
    up = _make_df('up')
    down = _make_df('down')
    data_long = {'5m': up, '15m': up, '30m': up, '4h': up}
    data_short = {'5m': down, '15m': down, '30m': down, '4h': down}
    assert confirm_trend(data_long, 'LONG')
    assert confirm_trend(data_short, 'SHORT')


def test_confirm_trend_missing_tf():
    up = _make_df('up')
    data = {'5m': up, '15m': up}
    assert confirm_trend(data, 'LONG')


def test_confirm_trend_empty_df_returns_reason():
    data = {'5m': pd.DataFrame()}
    ok, reason = confirm_trend(data, 'LONG', return_reason=True)
    assert ok is False
    assert reason == 'insufficient_data'


def test_confirm_trend_min_length_ok():
    df = _make_df('up', n=21)
    data = {'5m': df}
    ok, reason = confirm_trend(data, 'LONG', return_reason=True)
    assert ok
    assert reason is None


def test_confirm_trend_weak_trend_logged(tmp_path, monkeypatch):
    df = _make_low_adx_df()
    data = {'5m': df, '15m': df}
    monkeypatch.chdir(tmp_path)
    ok, reason = confirm_trend(data, 'LONG', symbol='BTC/USDT', return_reason=True)
    assert ok is True
    assert reason == 'weak_trend'
    log = pd.read_csv(tmp_path / 'decision_log.csv')
    assert 'weak_trend' in log['reason'].tolist()


def test_maybe_invert_position():
    assert maybe_invert_position('LONG', True, True) == 'SHORT'
    assert maybe_invert_position('SHORT', True, True) == 'LONG'
    assert maybe_invert_position('LONG', False, True) is None
    assert maybe_invert_position('LONG', True, False) is None
    assert maybe_invert_position('OTHER', True, True) is None
