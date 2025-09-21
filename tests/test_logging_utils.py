import builtins
import csv
from logging_utils import (
    safe_create_order,
    flush_symbol_logs,
    safe_set_leverage,
    setup_logger,
    log_entry,
    _ENTRY_CACHE,
    ensure_trades_csv_header,
    ensure_report_schema,
    TRADES_CSV_HEADER,
    LOG_ENTRY_FIELDS,
)

class DummyExchange:
    def create_order(self, *args, **kwargs):
        return {"id": "1", "status": "filled"}
    def set_leverage(self, leverage, symbol):
        return True

class FailExchange:
    def create_order(self, *args, **kwargs):
        raise RuntimeError("fail")
    def set_leverage(self, leverage, symbol):
        raise RuntimeError("bad")


def test_safe_create_order_success(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    order_id, err = safe_create_order(DummyExchange(), "BTC/USDT", "market", "buy", 1)
    assert order_id == "1"
    assert err is None
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "Order BUY" in logged


def test_safe_create_order_error(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    order_id, err = safe_create_order(FailExchange(), "ETH/USDT", "market", "buy", 1)
    assert order_id is None
    assert err is not None
    flush_symbol_logs("ETH/USDT")
    logged = caplog.text
    assert "Order failed" in logged


def test_safe_set_leverage(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    assert safe_set_leverage(DummyExchange(), "LTC/USDT", 20)
    flush_symbol_logs("LTC/USDT")
    logged = caplog.text
    assert "Leverage set to 20x" in logged


def test_safe_set_leverage_error(caplog):
    setup_logger()
    import logging
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    assert not safe_set_leverage(FailExchange(), "DOGE/USDT", 20)
    flush_symbol_logs("DOGE/USDT")
    logged = caplog.text
    assert "Failed to set leverage" in logged


def test_log_entry_idempotent_and_defaults(tmp_path):
    _ENTRY_CACHE.clear()
    log_file = tmp_path / "entries.csv"
    ctx = {
        "entry_price": 100.0,
        "qty": 1.0,
        "reason": "test",
        "side": "LONG",
        "sl_price": 95.0,
        "tp_price": 110.0,
    }
    t1 = log_entry("BTC/USDT", ctx, str(log_file))
    t2 = log_entry("BTC/USDT", ctx, str(log_file))
    assert t1 == t2
    with open(log_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "live"
    assert row["reduced_risk"] == "False"
    assert row["trade_id"] == t1
    assert row["sl"] == "95.0"
    assert row["tp"] == "110.0"


class PercentFilterBase:
    markets = True

    def market(self, symbol):
        return {
            "info": {
                "filters": [
                    {
                        "filterType": "PERCENT_PRICE",
                        "multiplierUp": "1.05",
                        "multiplierDown": "0.95",
                    }
                ]
            }
        }

    def fetch_ticker(self, symbol):
        return {"ask": 100.0, "bid": 99.0}

    def price_to_precision(self, symbol, price):
        return str(price)


class RetryExchange(PercentFilterBase):
    def __init__(self):
        self.calls = []

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append((order_type, price))
        if len(self.calls) == 1:
            raise RuntimeError("{\"code\":-4131}")
        return {"id": "1", "status": "FILLED", "price": price, "filled": qty}


class FallbackExchange(PercentFilterBase):
    def __init__(self):
        self.calls = []

    def create_order(self, symbol, order_type, side, qty, price=None, params=None):
        self.calls.append((order_type, price))
        if len(self.calls) <= 2:
            raise RuntimeError("{\"code\":-4131}")
        return {"id": "2", "status": "FILLED", "price": price, "filled": qty}


def test_safe_create_order_percent_filter_retry(caplog):
    setup_logger()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    ex = RetryExchange()
    order_id, err = safe_create_order(ex, "BTC/USDT", "limit", "buy", 1, 100)
    assert order_id == "1"
    assert err is None
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "order_retry_limit_price_adjusted" in logged


def test_safe_create_order_percent_filter_market_fallback(caplog):
    setup_logger()
    import logging

    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level("INFO")
    ex = FallbackExchange()
    order_id, err = safe_create_order(ex, "BTC/USDT", "limit", "buy", 1, 100)
    assert order_id == "2"
    assert err is None
    assert ex.calls[-1][0] == "market"
    flush_symbol_logs("BTC/USDT")
    logged = caplog.text
    assert "order_market_fallback" in logged


def test_ensure_trades_csv_header_migrates(tmp_path):
    path = tmp_path / "trades.csv"
    # simulate old header missing ``trade_id`` as the first column
    old_header = LOG_ENTRY_FIELDS[1:]
    with open(path, "w", newline="") as f:
        f.write(",".join(old_header) + "\n")
        f.write(",".join(str(i) for i in range(len(old_header))) + "\n")

    ensure_trades_csv_header(str(path))
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == TRADES_CSV_HEADER

    backup = path.with_name("trades_misaligned.csv")
    assert backup.exists()
    with open(backup) as f:
        legacy_header = f.readline().strip().split(",")
    assert legacy_header == old_header


def test_ensure_trades_csv_header_creates(tmp_path):
    path = tmp_path / "trades.csv"
    ensure_trades_csv_header(str(path))
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == TRADES_CSV_HEADER


def test_ensure_report_schema_resets(tmp_path):
    path = tmp_path / "pair_report.csv"
    with open(path, "w", newline="") as f:
        f.write("foo,bar\n1,2\n")

    ensure_report_schema(str(path), ["symbol", "winrate"])

    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == ["symbol", "winrate"]

    backup_new = tmp_path / "pair_report_misaligned.csv"
    backup_legacy = tmp_path / "pair_report_legacy.csv"
    assert backup_new.exists() or backup_legacy.exists()


def test_setup_logger_writes_file(tmp_path):
    log_path = tmp_path / "run.log"
    setup_logger(str(log_path))
    import logging

    logging.info("hello world")
    with open(log_path) as f:
        data = f.read()
    assert "hello world" in data


def test_setup_logger_creates_directory(tmp_path):
    log_path = tmp_path / "nested" / "run.log"
    setup_logger(str(log_path))
    assert log_path.exists()
