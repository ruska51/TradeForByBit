import pandas as pd
from volume_utils import safe_vol_ratio, volume_reason, VOL_WINDOW, VOL_RATIO_MAX

# [ANCHOR:TESTS_VOL_LOGIC]

def test_safe_vol_ratio_basic():
    s = pd.Series([1, 2, 3, 4, 5] * 5)
    assert safe_vol_ratio(s, VOL_WINDOW) is not None
    assert safe_vol_ratio(pd.Series([], dtype=float), VOL_WINDOW) == 1.0
    assert safe_vol_ratio(None, VOL_WINDOW) == 1.0

def test_safe_vol_ratio_clamped():
    s = pd.Series([0.0] * (VOL_WINDOW - 1) + [100.0])
    assert safe_vol_ratio(s, VOL_WINDOW) == VOL_RATIO_MAX


def test_safe_vol_ratio_zero_mean():
    s = pd.Series([0.0] * VOL_WINDOW)
    ratio = safe_vol_ratio(s, VOL_WINDOW)
    assert ratio is not None
    assert 0.08 <= ratio <= VOL_RATIO_MAX


def test_volume_reason_helper():
    assert volume_reason(pd.Series([1] * VOL_WINDOW), 1.0, VOL_WINDOW) is None
    short = pd.Series([1, 2])
    assert volume_reason(short, 1.0, VOL_WINDOW) == "vol_missing"
    s = pd.Series([1.0] * (VOL_WINDOW - 1) + [0.5])
    assert volume_reason(s, 1.0, VOL_WINDOW) == "vol_low"


def test_vol_ratio_with_nans_and_infs():
    data = [1.0] * (VOL_WINDOW - 2) + [float("nan"), 2.0, float("inf")]
    s = pd.Series(data)
    ratio = safe_vol_ratio(s, VOL_WINDOW)
    assert ratio is not None
    assert 0.08 <= ratio <= VOL_RATIO_MAX


def test_vol_missing_short_series():
    s = pd.Series([1, 2])
    assert safe_vol_ratio(s, VOL_WINDOW) == 1.0
    assert volume_reason(s, 1.3, VOL_WINDOW) == "vol_missing"
