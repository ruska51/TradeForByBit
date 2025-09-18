import argparse
import logging
from main import train_model, symbols, backtest
from asset_scanner import scan_markets


def run(selected):
    for sym in selected:
        logging.info(f"model | {sym} | Training new model...")
        try:
            train_model(sym)
            logging.info(f"model | {sym} | ✅ Model trained and saved")
        except Exception as e:
            logging.error(f"model | {sym} | ❌ Training failed: {e}")


def backtest_new_assets():
    """Scan markets and backtest symbols not in the current list."""
    new_symbols = scan_markets()
    for sym in new_symbols:
        if sym in symbols:
            continue
        logging.info(f"backtest | {sym} | Running backtest")
        try:
            metrics = backtest(sym)
            ret_val = metrics.get("return") if metrics else None
            if ret_val is not None:
                logging.info(f"backtest | {sym} | Return {ret_val:.2%}")
            else:
                logging.warning(f"backtest | {sym} | metrics missing")
        except Exception as e:
            logging.error(f"backtest | {sym} | Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train models for the trading bot")
    parser.add_argument("--symbol", help="train only the given symbol")
    parser.add_argument(
        "--backtest-new-assets",
        action="store_true",
        help="scan markets and backtest unfamiliar symbols",
    )
    args = parser.parse_args()
    if args.backtest_new_assets:
        backtest_new_assets()
    elif args.symbol:
        run([args.symbol])
    else:
        run(symbols)


if __name__ == "__main__":
    main()
