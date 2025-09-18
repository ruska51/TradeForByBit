from __future__ import annotations

import os
from typing import Dict, Tuple
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore
try:  # pragma: no cover - optional dependency
    import torch
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency
    import torch_stub as torch
    from torchvision_stub import transforms
import asyncio
import pandas as pd
import numpy as np

try:  # pragma: no cover - optional dependency
    import talib  # type: ignore
except Exception:  # pragma: no cover - talib is optional
    talib = None

# Pattern classes used by the CNN model
pattern_classes = [
    "none",
    # --- classical chart patterns ---
    "ascending_triangle",
    "descending_triangle",
    "double_top",
    "double_bottom",
    "rising_wedge",
    "falling_wedge",
    "symmetrical_triangle",
    "cup_and_handle",
    "rectangle",
    "bull_flag",
    "bear_flag",
    "ascending_channel",
    "descending_channel",
    "megaphone",
    "triple_top",
    "triple_bottom",
    # --- synthetic helpers / legacy classes ---
    "flag",
    "pennant",
    "wedge_up",
    "wedge_down",
    "triangle_sym",
    "triangle_asc",
    "triangle_desc",
    "head_and_shoulders",
    "inverse_head_and_shoulders",
]

REAL_PATTERNS = {
    "ascending_triangle",
    "descending_triangle",
    "double_top",
    "double_bottom",
    "rising_wedge",
    "falling_wedge",
    "symmetrical_triangle",
    "cup_and_handle",
    "rectangle",
    "bull_flag",
    "bear_flag",
    "ascending_channel",
    "descending_channel",
    "megaphone",
    "triple_top",
    "triple_bottom",
}
SYNTHETIC_PATTERNS = set(pattern_classes) - REAL_PATTERNS - {"none"}

# Basic categorisation to determine directional bias of detected patterns.
# These sets are shared with the trading logic to decide whether a pattern
# supports a long or short entry.
BULLISH_PATTERNS = {
    "bull_flag",
    "ascending_triangle",
    "ascending_channel",
    "cup_and_handle",
    "inverse_head_and_shoulders",
    "double_bottom",
    "triple_bottom",
    "hammer",
    "dragonfly_doji",
}

BEARISH_PATTERNS = {
    "bear_flag",
    "descending_triangle",
    "descending_channel",
    "head_and_shoulders",
    "double_top",
    "triple_top",
    "hanging_man",
    "dark_cloud_cover",
}

__all__ = [
    "detect_pattern",
    "detect_pattern_image",
    "pattern_classes",
    "BULLISH_PATTERNS",
    "BEARISH_PATTERNS",
]


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_model: torch.nn.Module | None = None


def _load_model() -> None:
    """Load CNN model from disk if available."""
    global _model
    if _model is None:
        path = os.path.join(os.path.dirname(__file__), 'pattern_model.pt')
        if os.path.exists(path):
            _model = torch.load(path, map_location="cpu", weights_only=False)
            _model.eval()
            # sanity check: output dimension must match number of classes
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    out = _model(dummy)
                if out.shape[1] != len(pattern_classes):
                    _model = None
            except Exception:  # pragma: no cover - incompatible model
                _model = None


def _classify(image_path: str) -> tuple[str, float]:
    """Return predicted pattern name and confidence from the CNN model."""
    _load_model()
    if _model is None or Image is None:
        return 'none', 0.0
    img = Image.open(image_path).convert('RGB')
    img_tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    name = pattern_classes[idx.item()]
    return name, float(conf)


def detect_pattern_image(image_path: str) -> Dict[str, object]:
    """Detect chart pattern from an image and return structured info."""
    name, conf = _classify(image_path)
    if name == 'none':
        source = 'none'
    elif name in SYNTHETIC_PATTERNS:
        source = 'synthetic'
    else:
        source = 'real'
    return {"pattern_name": name, "source": source, "confidence": conf}


async def detect_pattern(symbol: str, data: pd.DataFrame) -> Dict[str, object]:
    """Analyse OHLCV ``data`` and return detected pattern information."""

    await asyncio.sleep(0)

    if data is None or data.empty or len(data) < 10:
        return {"pattern_name": "none", "source": "none", "confidence": 0.0}

    open_ = data["open"].values
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values

    patterns: list[Tuple[str, float]] = []

    # candlestick patterns via TA-Lib (if available)
    if talib is not None:
        cdl_map = {
            "hammer": talib.CDLHAMMER,
            "hanging_man": talib.CDLHANGINGMAN,
            "dragonfly_doji": talib.CDLDRAGONFLYDOJI,
            "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,
        }
        for name, func in cdl_map.items():
            try:
                res = func(open_, high, low, close)
                if res[-1] != 0:
                    patterns.append((name, 0.6))
            except Exception:  # pragma: no cover - talib may not support
                continue

    def _linear_regression(y: np.ndarray) -> Tuple[float, float, float]:
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        return slope, intercept, r2

    # --- channel detection ---
    def _detect_channel(ascending: bool) -> float:
        slope_h, _, r2_h = _linear_regression(high)
        slope_l, _, r2_l = _linear_regression(low)
        if ascending and (slope_h > 0 and slope_l > 0):
            diff = abs(slope_h - slope_l) / (abs(slope_h) + abs(slope_l) + 1e-8)
            return max(0.0, (1 - diff) * (r2_h + r2_l) / 2)
        if not ascending and (slope_h < 0 and slope_l < 0):
            diff = abs(slope_h - slope_l) / (abs(slope_h) + abs(slope_l) + 1e-8)
            return max(0.0, (1 - diff) * (r2_h + r2_l) / 2)
        return 0.0

    conf = _detect_channel(True)
    if conf > 0.6:
        patterns.append(("ascending_channel", float(min(1.0, conf))))

    conf = _detect_channel(False)
    if conf > 0.6:
        patterns.append(("descending_channel", float(min(1.0, conf))))

    # --- rectangle (sideways channel) ---
    slope_h, _, r2_h = _linear_regression(high)
    slope_l, _, r2_l = _linear_regression(low)
    if abs(slope_h) < 1e-3 and abs(slope_l) < 1e-3:
        conf = (r2_h + r2_l) / 2
        if conf > 0.5:
            patterns.append(("rectangle", float(min(1.0, conf))))

    # --- megaphone (broadening formation) ---
    if slope_h > 0 and slope_l < 0:
        conf = (r2_h + r2_l) / 2
        if conf > 0.5:
            patterns.append(("megaphone", float(min(1.0, conf))))

    # --- bull/bear flags ---
    def _detect_flag(bullish: bool) -> float:
        n = len(close)
        first = close[: n // 2]
        second = close[n // 2 :]
        s1, _, _ = _linear_regression(first)
        s2, _, r2 = _linear_regression(second)
        if bullish and s1 > 0 and s2 < s1 * 0.5:
            return max(0.0, r2 * (s1 / (abs(s2) + 1e-8)))
        if not bullish and s1 < 0 and s2 > s1 * 0.5:
            return max(0.0, r2 * (-s1 / (abs(s2) + 1e-8)))
        return 0.0

    conf = _detect_flag(True)
    if conf > 0.6:
        patterns.append(("bull_flag", float(min(1.0, conf))))

    conf = _detect_flag(False)
    if conf > 0.6:
        patterns.append(("bear_flag", float(min(1.0, conf))))

    # --- triple top/bottom ---
    def _detect_triple(peaks: np.ndarray, tolerance: float = 0.003) -> float:
        if len(peaks) < 3:
            return 0.0
        last3 = peaks[-3:]
        if np.max(last3) - np.min(last3) <= np.mean(last3) * tolerance:
            return 0.7
        return 0.0

    peaks = high[(np.r_[True, high[1:] > high[:-1]] & np.r_[high[:-1] > high[1:], True])]
    conf = _detect_triple(peaks)
    if conf:
        patterns.append(("triple_top", conf))

    troughs = low[(np.r_[True, low[1:] < low[:-1]] & np.r_[low[:-1] < low[1:], True])]
    conf = _detect_triple(troughs)
    if conf:
        patterns.append(("triple_bottom", conf))

    # --- cup and handle ---
    def _detect_cup_handle() -> float:
        n = len(close)
        if n < 20:
            return 0.0
        mid = np.argmin(close)
        if not (n * 0.25 < mid < n * 0.75):
            return 0.0
        start = close[0]
        end = close[mid]
        last = close[-1]
        if abs(close[-1] - start) / start > 0.05:
            return 0.0
        if (start - end) / start < 0.05:
            return 0.0
        handle = close[int(n * 0.75) :]
        if len(handle) < 3:
            return 0.0
        if handle.min() > start * 0.9:
            return 0.7
        return 0.0

    conf = _detect_cup_handle()
    if conf:
        patterns.append(("cup_and_handle", conf))

    # choose best pattern
    if not patterns:
        return {"pattern_name": "none", "source": "none", "confidence": 0.0}

    name, conf = max(patterns, key=lambda x: x[1])
    source = "synthetic" if name in SYNTHETIC_PATTERNS else "real"
    conf = float(min(1.0, conf))
    return {"pattern_name": name, "source": source, "confidence": conf}


def self_test(df: pd.DataFrame, window: int = 50) -> float:
    """Return share of valid pattern detections over a sliding ``window``.

    The function repeatedly applies :func:`detect_pattern` to subwindows of
    ``df`` and calculates the fraction of windows where a non-``"none"``
    pattern is detected with at least medium confidence. If the dataframe is
    empty or smaller than the window size ``0.0`` is returned.
    """

    if df is None or df.empty or len(df) < window:
        return 0.0

    hits = 0
    total = 0
    for i in range(len(df) - window + 1):
        sub = df.iloc[i : i + window]
        # ``detect_pattern`` is asynchronous; run it in a local loop.
        info = asyncio.run(detect_pattern("self", sub))
        if info["pattern_name"] != "none" and info["confidence"] >= 0.6:
            hits += 1
        total += 1

    return hits / total if total else 0.0
