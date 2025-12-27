"""
Скрипт для отмены всех conditional ордеров на Bybit testnet.
Используйте перед тестированием нового механизма установки SL/TP.
"""

import logging
from credentials import API_KEY, API_SECRET
from exchange_adapter import ExchangeAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def cancel_all_conditional_orders():
    """Отменяем все conditional ордера."""

    config = {
        "sandbox": True,
        "exchange_id": "bybit",
        "apiKey": API_KEY,
        "secret": API_SECRET,
    }

    adapter = ExchangeAdapter(config=config)

    if not adapter.x:
        logging.error("❌ Не удалось инициализировать exchange")
        return False

    exchange = adapter.x

    logging.info("✅ Подключение к Bybit testnet успешно")

    # Получаем все символы с открытыми позициями
    symbols = []
    try:
        positions = exchange.fetch_positions(params={"category": "linear"})
        for pos in positions:
            if pos.get('contracts', 0) > 0:
                symbols.append(pos.get('symbol'))
    except Exception as e:
        logging.warning(f"Ошибка при получении позиций: {e}")

    if not symbols:
        logging.info("📋 Нет открытых позиций")
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "TRX/USDT:USDT"]  # Проверим основные пары

    logging.info(f"🔍 Проверяем ордера для символов: {symbols}")

    total_cancelled = 0

    for symbol in symbols:
        try:
            # Получаем все условные ордера
            orders = exchange.fetch_open_orders(
                symbol,
                params={"category": "linear", "orderFilter": "StopOrder"}
            )

            logging.info(f"📋 {symbol}: найдено {len(orders)} условных ордеров")

            for order in orders:
                order_id = order.get('id')
                order_type = order.get('type')
                side = order.get('side')

                try:
                    exchange.cancel_order(order_id, symbol, params={"category": "linear"})
                    logging.info(f"   ✅ Отменен: {order_id} ({order_type} {side})")
                    total_cancelled += 1
                except Exception as e:
                    logging.warning(f"   ❌ Не удалось отменить {order_id}: {e}")

        except Exception as e:
            logging.warning(f"Ошибка при обработке {symbol}: {e}")

    logging.info("="*60)
    logging.info(f"✅ Всего отменено ордеров: {total_cancelled}")
    logging.info("="*60)

    return True


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("ОТМЕНА ВСЕХ CONDITIONAL ОРДЕРОВ")
    logging.info("="*60)

    cancel_all_conditional_orders()
