"""
Прямое закрытие позиции TRX SHORT (positionIdx=2).
"""

import logging
import time
from credentials import API_KEY, API_SECRET
from exchange_adapter import ExchangeAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def close_trx_short():
    """Закрываем SHORT позицию TRX."""

    config = {
        "sandbox": True,
        "exchange_id": "bybit",
        "apiKey": API_KEY,
        "secret": API_SECRET,
    }

    adapter = ExchangeAdapter(config=config)
    exchange = adapter.x

    logging.info("✅ Подключение к Bybit testnet успешно")

    symbol = "TRX/USDT:USDT"
    category = "linear"
    qty = 2232.0  # Точное количество из позиции
    side = "buy"  # Для закрытия SHORT нужен BUY
    position_idx = 2  # SHORT в hedge mode

    logging.info(f"🎯 Закрываем SHORT позицию TRX")
    logging.info(f"   Символ: {symbol}")
    logging.info(f"   Количество: {qty}")
    logging.info(f"   Сторона: {side}")
    logging.info(f"   Position Idx: {position_idx}")

    # Попытка 1: Market order с правильным positionIdx
    logging.info("="*60)
    logging.info("Попытка 1: Market order")
    logging.info("="*60)

    try:
        params = {
            "category": category,
            "reduceOnly": True,
            "positionIdx": position_idx,
        }

        logging.info(f"📤 Параметры: {params}")

        order = exchange.create_order(
            symbol,
            "market",
            side,
            qty,
            None,
            params
        )

        logging.info(f"✅ Ордер создан: {order}")
        time.sleep(2)

        # Проверяем позицию
        positions = exchange.fetch_positions([symbol], params={"category": category})
        for pos in positions:
            if pos.get('info', {}).get('positionIdx') == str(position_idx):
                contracts = pos.get('contracts', 0)
                if contracts > 0:
                    logging.warning(f"⚠️ Позиция еще открыта: {contracts}")
                else:
                    logging.info("🎉 ПОЗИЦИЯ ЗАКРЫТА!")
                    return True

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")

    # Попытка 2: Без reduceOnly
    logging.info("="*60)
    logging.info("Попытка 2: Market order БЕЗ reduceOnly")
    logging.info("="*60)

    try:
        params = {
            "category": category,
            "positionIdx": position_idx,
        }

        logging.info(f"📤 Параметры: {params}")

        order = exchange.create_order(
            symbol,
            "market",
            side,
            qty,
            None,
            params
        )

        logging.info(f"✅ Ордер создан: {order}")
        time.sleep(2)

        # Проверяем позицию
        positions = exchange.fetch_positions([symbol], params={"category": category})
        for pos in positions:
            if pos.get('info', {}).get('positionIdx') == str(position_idx):
                contracts = pos.get('contracts', 0)
                if contracts > 0:
                    logging.warning(f"⚠️ Позиция еще открыта: {contracts}")
                else:
                    logging.info("🎉 ПОЗИЦИЯ ЗАКРЫТА!")
                    return True

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")

    return False


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("ЗАКРЫТИЕ TRX SHORT ПОЗИЦИИ")
    logging.info("="*60)

    success = close_trx_short()

    if success:
        logging.info("="*60)
        logging.info("✅ УСПЕХ!")
        logging.info("="*60)
    else:
        logging.info("="*60)
        logging.info("❌ НЕ УДАЛОСЬ ЗАКРЫТЬ")
        logging.info("="*60)
