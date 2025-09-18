from datetime import datetime, timezone, timedelta

from risk_management import StatsTracker


def test_symbol_ban_and_soft_risk():
    stats = StatsTracker()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    stats.on_trade_close("BTC/USDT", -1, timestamp=base)
    stats.on_trade_close("BTC/USDT", -1, timestamp=base + timedelta(minutes=10))

    assert stats.is_banned("BTC/USDT", now=base + timedelta(minutes=10))

    assert not stats.is_banned("BTC/USDT", now=base + timedelta(minutes=70))
    assert stats.pop_soft_risk("BTC/USDT")

    stats.on_trade_close("BTC/USDT", -1, timestamp=base + timedelta(minutes=80))
    assert stats.pop_soft_risk("BTC/USDT")

    stats.on_trade_close("BTC/USDT", 1, timestamp=base + timedelta(minutes=90))
    assert not stats.pop_soft_risk("BTC/USDT")
    assert stats.stats["BTC/USDT"]["loss_streak"] == 0
