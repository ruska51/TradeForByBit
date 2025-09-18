class NetworkError(Exception):
    """Minimal stub for ccxt NetworkError."""


class binance:
    def __init__(self, *args, **kwargs):
        self.symbols = []

    def set_sandbox_mode(self, mode):
        pass

    def load_markets(self, reload=False, /):
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        return {}

    def fetch_positions(self, symbols):
        return []

    def cancel_all_orders(self, symbol):
        return []

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        return {"id": "stub", "type": type, "side": side, "avgPrice": price or 0.0}
