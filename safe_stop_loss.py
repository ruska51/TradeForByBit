"""
Adaptive Stop Loss System with Leverage Awareness

This module provides safe stop-loss calculation functions that:
1. Account for leverage to prevent liquidation
2. Adapt to each trading pair's volatility
3. Ensure margin loss stays within acceptable limits

Key Features:
- Leverage-aware SL calculation (prevents >25% margin loss)
- Per-pair volatility parameters (MIN_SL)
- Safety validation before order placement
- Detailed logging for monitoring

Mathematical Formula:
    SL% = max(MAX_MARGIN_LOSS / leverage, MIN_SL_FOR_PAIR)

Where:
    MAX_MARGIN_LOSS = 0.15 (15% of margin - conservative)
    MIN_SL_FOR_PAIR = volatility-based minimum (0.8% - 2.0%)

Example:
    BTC/USDT with leverage 10x:
    SL% = max(0.15/10, 0.008) = max(0.015, 0.008) = 1.5%
    Margin loss if triggered: 1.5% * 10 = 15%

Author: Claude Code
Date: 2025-12-28
Version: 1.0.0
"""

import logging
import math
from typing import Tuple, Optional


# === CONFIGURATION ===

# Maximum acceptable margin loss when SL is triggered
# ИСПРАВЛЕНО 2025-12-30: Снижено до 12% для защиты от превращения прибыли в убыток
# При leverage 10x: SL = 12%/10 = 1.2% (оптимально для scalping)
# При leverage 12x: SL = 12%/12 = 1.0% (защита от широких стопов)
MAX_MARGIN_LOSS = 0.12  # 12% of margin (СНИЖЕНО с 25%)

# Maximum SL distance - оставлен для swing mode
MAX_SL_PCT = 0.040  # 4.0% (для swing режима с широкими стопами)

# Per-pair minimum SL based on volatility characteristics
# ИСПРАВЛЕНО 2025-12-30: Снижено для защиты прибыли
# При MAX_MARGIN_LOSS=12% и leverage 10x: SL = max(1.2%, MIN_SL) = 1.2%
# При MAX_MARGIN_LOSS=12% и leverage 12x: SL = max(1.0%, MIN_SL) = 1.0%
#
# Цель: предотвратить превращение +$30 прибыли в -$3 убыток из-за широких стопов
MIN_SL_CONFIG = {
    "BTC/USDT": 0.010,   # 1.0% - узкий стоп для защиты прибыли
    "ETH/USDT": 0.010,   # 1.0% - узкий стоп для защиты прибыли
    "LINK/USDT": 0.010,  # 1.0% - узкий стоп
    "MNT/USDT": 0.012,   # 1.2% - немного шире из-за волатильности
    "SOL/USDT": 0.010,   # 1.0% - узкий стоп
    "SUI/USDT": 0.010,   # 1.0% - узкий стоп
    "TRX/USDT": 0.010,   # 1.0% - не используется (TRX игнорируется)
}

# Default MIN_SL for pairs not in config
DEFAULT_MIN_SL = 0.010  # 1.0%

# Safety threshold: warn if margin loss exceeds this
MARGIN_LOSS_WARNING_THRESHOLD = 0.15  # 15% (СНИЖЕНО с 25%)


# === CORE FUNCTIONS ===

def get_min_sl_for_pair(symbol: str) -> float:
    """
    Get minimum stop-loss percentage for a trading pair.

    Returns pair-specific MIN_SL based on volatility characteristics.
    Falls back to DEFAULT_MIN_SL if pair is not configured.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT', 'ETHUSDT')

    Returns:
        Minimum SL percentage as decimal (e.g., 0.008 for 0.8%)

    Examples:
        >>> get_min_sl_for_pair('BTC/USDT')
        0.008
        >>> get_min_sl_for_pair('UNKNOWN/USDT')
        0.015
    """
    # Normalize symbol (handle both 'BTC/USDT' and 'BTCUSDT' formats)
    symbol_normalized = symbol.replace('/', '').upper()

    # Try exact match first
    if symbol in MIN_SL_CONFIG:
        return MIN_SL_CONFIG[symbol]

    # Try to find by normalized format (e.g., BTCUSDT -> BTC/USDT)
    for key in MIN_SL_CONFIG:
        if key.replace('/', '').upper() == symbol_normalized:
            return MIN_SL_CONFIG[key]

    # Fallback to default
    logging.debug(f"safe_sl | {symbol} | Using default MIN_SL={DEFAULT_MIN_SL:.3%} (pair not in config)")
    return DEFAULT_MIN_SL


def calculate_safe_sl(
    symbol: str,
    entry_price: float,
    leverage: float,
    side: str
) -> Tuple[float, float, float]:
    """
    Calculate safe stop-loss price accounting for leverage and volatility.

    This function ensures:
    1. SL accounts for leverage (margin loss ≤ 15%)
    2. SL respects pair's minimum distance (noise protection)
    3. SL never exceeds maximum distance (2.5%)
    4. Calculations are numerically safe (handle edge cases)

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        entry_price: Entry price of the position
        leverage: Position leverage (e.g., 5.0, 10.0, 20.0)
        side: Position side ('long' or 'short')

    Returns:
        Tuple of (sl_price, sl_pct, margin_loss_pct):
            - sl_price: Calculated stop-loss price
            - sl_pct: Stop-loss distance as percentage (e.g., 0.015 = 1.5%)
            - margin_loss_pct: Expected margin loss if SL triggers (e.g., 0.15 = 15%)

    Raises:
        ValueError: If inputs are invalid (negative, zero, NaN, etc.)

    Examples:
        >>> # BTC Long with 10x leverage
        >>> calculate_safe_sl('BTC/USDT', 90000.0, 10.0, 'long')
        (88650.0, 0.015, 0.15)  # SL at -1.5%, margin loss 15%

        >>> # SOL Short with 20x leverage
        >>> calculate_safe_sl('SOL/USDT', 200.0, 20.0, 'short')
        (204.0, 0.02, 0.40)  # SL at +2.0%, margin loss 40%
    """
    # === INPUT VALIDATION ===

    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: {symbol}")

    if not entry_price or entry_price <= 0 or not math.isfinite(entry_price):
        raise ValueError(f"Invalid entry_price: {entry_price}")

    if not leverage or leverage <= 0 or not math.isfinite(leverage):
        raise ValueError(f"Invalid leverage: {leverage}")

    if leverage > 100:
        logging.warning(f"safe_sl | {symbol} | Very high leverage: {leverage}x (possible error?)")

    side_norm = str(side).lower()
    if side_norm not in ['long', 'short', 'buy', 'sell']:
        raise ValueError(f"Invalid side: {side} (must be 'long' or 'short')")

    # Normalize side to long/short
    if side_norm in ['buy', 'long']:
        side_norm = 'long'
    else:
        side_norm = 'short'

    # === CALCULATE SL PERCENTAGE ===

    # Get minimum SL for this pair (volatility-based)
    min_sl = get_min_sl_for_pair(symbol)

    # Calculate base SL from leverage (margin loss / leverage)
    # Example: 15% margin loss / 10x leverage = 1.5% SL
    base_sl = MAX_MARGIN_LOSS / leverage

    # Take maximum of base_sl and min_sl
    # This ensures we don't set SL too tight (noise protection)
    sl_pct = max(base_sl, min_sl)

    # Cap at maximum to prevent too wide stops
    sl_pct = min(sl_pct, MAX_SL_PCT)

    # === CALCULATE SL PRICE ===

    if side_norm == 'long':
        # Long: SL below entry price
        sl_price = entry_price * (1 - sl_pct)
    else:
        # Short: SL above entry price
        sl_price = entry_price * (1 + sl_pct)

    # === CALCULATE ACTUAL MARGIN LOSS ===

    # Margin loss = price movement % * leverage
    margin_loss_pct = sl_pct * leverage

    # === SAFETY CHECK ===

    if not math.isfinite(sl_price) or sl_price <= 0:
        raise ValueError(f"Calculated SL price is invalid: {sl_price}")

    # === RETURN RESULTS ===

    return (sl_price, sl_pct, margin_loss_pct)


def validate_sl_safety(
    symbol: str,
    sl_price: float,
    entry_price: float,
    leverage: float,
    side: str
) -> Tuple[bool, str]:
    """
    Validate that stop-loss won't cause excessive margin loss or liquidation.

    Safety criteria:
    1. Margin loss if SL triggers must be < 25% (conservative threshold)
    2. SL price must be valid (positive, finite)
    3. SL must be on correct side of entry (long: below, short: above)

    Args:
        symbol: Trading pair
        sl_price: Proposed stop-loss price
        entry_price: Entry price of position
        leverage: Position leverage
        side: Position side ('long' or 'short')

    Returns:
        Tuple of (is_safe, message):
            - is_safe: True if SL is safe, False otherwise
            - message: Explanation (warning message if unsafe, empty if safe)

    Examples:
        >>> # Safe SL for BTC long
        >>> validate_sl_safety('BTC/USDT', 88650.0, 90000.0, 10.0, 'long')
        (True, '')

        >>> # Unsafe SL - too close, would cause 50% margin loss
        >>> validate_sl_safety('BTC/USDT', 85500.0, 90000.0, 10.0, 'long')
        (False, 'SL would cause 50.0% margin loss (threshold: 25%)')
    """
    # === INPUT VALIDATION ===

    if not sl_price or sl_price <= 0 or not math.isfinite(sl_price):
        return (False, f"Invalid SL price: {sl_price}")

    if not entry_price or entry_price <= 0 or not math.isfinite(entry_price):
        return (False, f"Invalid entry price: {entry_price}")

    if not leverage or leverage <= 0 or not math.isfinite(leverage):
        return (False, f"Invalid leverage: {leverage}")

    side_norm = str(side).lower()
    if side_norm in ['buy', 'long']:
        side_norm = 'long'
    elif side_norm in ['sell', 'short']:
        side_norm = 'short'
    else:
        return (False, f"Invalid side: {side}")

    # === CHECK SL DIRECTION ===

    if side_norm == 'long' and sl_price >= entry_price:
        return (False, f"Long SL must be below entry (SL={sl_price:.2f}, Entry={entry_price:.2f})")

    if side_norm == 'short' and sl_price <= entry_price:
        return (False, f"Short SL must be above entry (SL={sl_price:.2f}, Entry={entry_price:.2f})")

    # === CALCULATE MARGIN LOSS ===

    # Calculate SL distance as percentage
    sl_pct = abs((sl_price / entry_price) - 1)

    # Calculate margin loss if SL triggers
    margin_loss_pct = sl_pct * leverage

    # === CHECK MARGIN LOSS THRESHOLD ===

    if margin_loss_pct > MARGIN_LOSS_WARNING_THRESHOLD:
        return (
            False,
            f"SL would cause {margin_loss_pct:.1%} margin loss (threshold: {MARGIN_LOSS_WARNING_THRESHOLD:.0%})"
        )

    # === ALL CHECKS PASSED ===

    return (True, "")


def log_safe_sl_info(
    symbol: str,
    entry_price: float,
    sl_price: float,
    tp_price: Optional[float],
    sl_pct: float,
    margin_loss_pct: float,
    leverage: float,
    side: str
) -> None:
    """
    Log detailed information about safe SL configuration.

    Provides clear, formatted output for monitoring and debugging.

    Args:
        symbol: Trading pair
        entry_price: Entry price
        sl_price: Stop-loss price
        tp_price: Take-profit price (optional)
        sl_pct: SL distance as percentage
        margin_loss_pct: Expected margin loss
        leverage: Position leverage
        side: Position side

    Example output:
        🛡️ Safe SL: BTC/USDT leverage=10x
           SL: 1.50% (price: 88,650.00)
           Margin loss if SL hit: 15.0%
           Entry: 90,000.00, TP: 94,500.00
    """
    # Format prices with appropriate precision
    if entry_price >= 1000:
        # Large prices (BTC): show 2 decimals with thousand separators
        price_fmt = ",.2f"
    elif entry_price >= 10:
        # Medium prices (ETH, SOL): show 2 decimals
        price_fmt = ".2f"
    else:
        # Small prices (altcoins): show 4 decimals
        price_fmt = ".4f"

    # Build log message
    tp_str = f"{tp_price:{price_fmt}}" if tp_price else "N/A"

    logging.info(
        f"🛡️ Safe SL: {symbol} leverage={leverage:.0f}x {side.upper()}\n"
        f"   SL: {sl_pct:.2%} (price: {sl_price:{price_fmt}})\n"
        f"   Margin loss if SL hit: {margin_loss_pct:.1%}\n"
        f"   Entry: {entry_price:{price_fmt}}, TP: {tp_str}"
    )


# === UTILITY FUNCTIONS ===

def get_expected_results_table() -> str:
    """
    Generate expected results table for all configured pairs.

    Returns:
        Formatted markdown table showing expected SL% and margin loss
        for each pair at different leverage levels.

    Useful for:
    - Documentation
    - Verification during testing
    - User education
    """
    table = [
        "| Pair | Leverage 5x | Leverage 10x | Leverage 20x | Margin Loss Range |",
        "|------|-------------|--------------|--------------|-------------------|"
    ]

    leverages = [5, 10, 20]

    for symbol, min_sl in MIN_SL_CONFIG.items():
        row_data = [symbol.replace('/USDT', '')]

        for lev in leverages:
            _, sl_pct, _ = calculate_safe_sl(symbol, 100.0, lev, 'long')
            row_data.append(f"{sl_pct:.1%}")

        # Calculate margin loss range
        _, _, margin_5x = calculate_safe_sl(symbol, 100.0, 5, 'long')
        _, _, margin_20x = calculate_safe_sl(symbol, 100.0, 20, 'long')
        margin_range = f"{margin_5x:.0%}-{margin_20x:.0%}"
        row_data.append(margin_range)

        table.append("| " + " | ".join(row_data) + " |")

    return "\n".join(table)


if __name__ == "__main__":
    """
    Self-test and demonstration.

    Run this module directly to verify calculations and see examples.
    """
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    print("=" * 70)
    print("SAFE STOP LOSS SYSTEM - SELF TEST")
    print("=" * 70)
    print()

    # Test cases
    test_cases = [
        ("BTC/USDT", 90000.0, 10.0, "long"),
        ("ETH/USDT", 3000.0, 10.0, "long"),
        ("SOL/USDT", 200.0, 20.0, "long"),
        ("LINK/USDT", 15.0, 5.0, "short"),
    ]

    print("Test Cases:")
    print("-" * 70)

    for symbol, entry, lev, side in test_cases:
        try:
            sl_price, sl_pct, margin_loss = calculate_safe_sl(symbol, entry, lev, side)
            is_safe, msg = validate_sl_safety(symbol, sl_price, entry, lev, side)

            log_safe_sl_info(symbol, entry, sl_price, None, sl_pct, margin_loss, lev, side)

            if is_safe:
                print(f"   [OK] Validation: PASSED")
            else:
                print(f"   [FAIL] Validation: FAILED - {msg}")
            print()

        except Exception as e:
            print(f"   [ERROR]: {e}")
            print()

    print("=" * 70)
    print("EXPECTED RESULTS TABLE")
    print("=" * 70)
    print()
    print(get_expected_results_table())
    print()

    print("=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
