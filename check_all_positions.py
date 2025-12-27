"""
Скрипт для проверки всех позиций на Bybit testnet.
"""

import logging
from credentials import API_KEY, API_SECRET
from exchange_adapter import ExchangeAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def check_all_positions():
    """Проверяем все позиции."""

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

    # Получаем все позиции
    try:
        positions = exchange.fetch_positions(params={"category": "linear"})
        logging.info(f"📋 Всего позиций: {len(positions)}")

        for pos in positions:
            symbol = pos.get('symbol')
            contracts = pos.get('contracts', 0)
            side = pos.get('side')
            entry_price = pos.get('entryPrice')
            unrealized_pnl = pos.get('unrealizedPnl', 0)

            info = pos.get('info', {})
            position_idx = info.get('positionIdx', 'N/A')
            leverage = info.get('leverage', 'N/A')

            if contracts and contracts > 0:
                logging.info("="*60)
                logging.info(f"📊 Символ: {symbol}")
                logging.info(f"   Размер: {contracts}")
                logging.info(f"   Сторона: {side}")
                logging.info(f"   Цена входа: {entry_price}")
                logging.info(f"   Нереализованный P&L: {unrealized_pnl}")
                logging.info(f"   Position Idx: {position_idx}")
                logging.info(f"   Кредитное плечо: {leverage}")

                # Дополнительная информация
                logging.info(f"   Info: {info}")

    except Exception as e:
        logging.error(f"❌ Ошибка при получении позиций: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("ПРОВЕРКА ВСЕХ ПОЗИЦИЙ")
    logging.info("="*60)

    check_all_positions()
