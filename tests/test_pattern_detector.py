import asyncio
import pandas as pd
import numpy as np

from pattern_detector import detect_pattern


def make_channel(up=True, n=30):
    base = 100
    slope = 0.5 if up else -0.5
    idx = np.arange(n)
    mid = base + slope * idx
    data = {
        "open": mid,
        "high": mid + 1,
        "low": mid - 1,
        "close": mid + 0.2,
        "volume": np.full(n, 1000),
    }
    return pd.DataFrame(data)


def test_detect_ascending_channel():
    df = make_channel(True)
    info = asyncio.run(detect_pattern("TEST", df))
    assert info["pattern_name"] == "ascending_channel"
    assert info["source"] == "real"
    assert info["confidence"] >= 0.5
