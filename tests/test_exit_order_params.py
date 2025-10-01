import types


def _make_dummy_exchange():
    class DummyExchange:
        id = "bybit"

        def price_to_precision(self, symbol, price):
            return price

        def amount_to_precision(self, symbol, amount):
            return amount

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_open_orders(self, *args, **kwargs):
            return []

    return DummyExchange()


def test_place_protected_exit_spot_skips_trigger_direction(monkeypatch):
    import main

    exchange = _make_dummy_exchange()
    monkeypatch.setattr(main, "exchange", exchange, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "spot")
    monkeypatch.setattr(main, "adjust_price_to_percent_filter", lambda _symbol, price: price)

    captured_params = {}

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_params.update(params or {})
        return "order-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    result = main.place_protected_exit(
        "BTC/USDT",
        "STOP_MARKET",
        "sell",
        1.0,
        100.0,
    )

    assert result == "order-id"
    assert "triggerDirection" not in captured_params
    assert "triggerBy" not in captured_params


def test_ensure_exit_orders_spot_skips_trigger_direction(monkeypatch):
    import main

    exchange = _make_dummy_exchange()

    adapter = types.SimpleNamespace(client=exchange)

    monkeypatch.setattr(main, "open_trade_ctx", {}, raising=False)
    monkeypatch.setattr(main, "detect_market_category", lambda *_: "spot")

    captured_calls = []

    def fake_safe_create_order(_exchange, symbol, order_kind, side, qty, price, params):
        captured_calls.append({"order_kind": order_kind, "params": dict(params or {})})
        return f"{order_kind}-id", None

    monkeypatch.setattr(main, "safe_create_order", fake_safe_create_order)

    main.ensure_exit_orders(
        adapter,
        "BTC/USDT",
        "long",
        1.0,
        sl_price=95.0,
        tp_price=105.0,
    )

    assert captured_calls, "Expected at least one safe_create_order call"
    for call in captured_calls:
        assert "triggerDirection" not in call["params"]
        assert "triggerBy" not in call["params"]
