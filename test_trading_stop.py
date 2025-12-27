"""
Тестовый скрипт для проверки установки SL/TP через set_trading_stop API.
Этот скрипт открывает тестовую позицию и устанавливает на неё SL/TP.
"""

import os
import sys
import logging
import ccxt
from credentials import API_KEY, API_SECRET

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def test_trading_stop():
    """Тестируем установку trading stop на позицию."""

    # Инициализация Bybit testnet через ExchangeAdapter
    from exchange_adapter import ExchangeAdapter

    config = {
        "sandbox": True,
        "exchange_id": "bybit",
        "apiKey": API_KEY,  # Правильное имя для ExchangeAdapter
        "secret": API_SECRET,  # Правильное имя для ExchangeAdapter
    }

    adapter = ExchangeAdapter(config=config)

    if not adapter.x:
        logging.error("❌ Не удалось инициализировать exchange")
        return False

    exchange = adapter.x

    logging.info("✅ Подключение к Bybit testnet успешно через ExchangeAdapter")

    # Тестовый символ
    symbol = "BTC/USDT:USDT"
    category = "linear"

    # Проверяем наличие открытых позиций
    from logging_utils import has_open_position

    try:
        # Сначала попробуем получить все позиции напрямую для диагностики
        logging.info(f"🔍 Проверяем позиции для {symbol} (category={category})...")

        try:
            positions = exchange.fetch_positions([symbol], params={"category": category})
            logging.info(f"📋 Получено позиций: {len(positions)}")

            for pos in positions:
                logging.info(f"   Position: {pos.get('symbol')} | size={pos.get('contracts')} | side={pos.get('side')}")

        except Exception as e:
            logging.warning(f"Ошибка при fetch_positions: {e}")

        qty_signed, qty_abs = has_open_position(exchange, symbol, category)
        logging.info(f"📊 has_open_position вернула: qty_signed={qty_signed}, qty_abs={qty_abs}")

        if qty_abs <= 0:
            logging.warning("❌ Нет открытой позиции. Пожалуйста, откройте позицию вручную на testnet.")
            logging.info("Перейдите на https://testnet.bybit.com/trade/usdt/BTCUSDT и откройте небольшую позицию")

            # Попробуем альтернативный символ
            logging.info("🔄 Пробуем альтернативный формат символа...")
            alt_symbol = "BTCUSDT"
            qty_signed, qty_abs = has_open_position(exchange, alt_symbol, category)
            logging.info(f"📊 Альтернативный символ {alt_symbol}: qty_signed={qty_signed}, qty_abs={qty_abs}")

            if qty_abs <= 0:
                return False
            else:
                # Используем альтернативный символ
                symbol = alt_symbol
                logging.info(f"✅ Используем символ: {symbol}")

        logging.info(f"✅ Найдена открытая позиция: qty={qty_abs}, signed={qty_signed}")

        # Получаем текущую цену
        ticker = exchange.fetch_ticker(symbol)
        current_price = float(ticker['last'])
        logging.info(f"📊 Текущая цена {symbol}: {current_price}")

        # Определяем направление позиции
        is_long = qty_signed > 0
        side_open = "buy" if is_long else "sell"

        logging.info(f"📈 Направление позиции: {'LONG' if is_long else 'SHORT'}")

        # Рассчитываем SL и TP (1% от текущей цены)
        if is_long:
            sl_price = current_price * 0.99  # SL на 1% ниже для long
            tp_price = current_price * 1.02  # TP на 2% выше для long
        else:
            sl_price = current_price * 1.01  # SL на 1% выше для short
            tp_price = current_price * 0.98  # TP на 2% ниже для short

        logging.info(f"🎯 Устанавливаем: SL={sl_price:.2f}, TP={tp_price:.2f}")

        # Используем нашу новую функцию
        from logging_utils import set_position_tp_sl

        success, error = set_position_tp_sl(
            exchange=exchange,
            symbol=symbol,
            tp_price=tp_price,
            sl_price=sl_price,
            category=category,
            side_open=side_open,
        )

        if success:
            logging.info("✅ Trading stop успешно установлен!")
            logging.info("Проверьте на веб-интерфейсе Bybit testnet:")
            logging.info("https://testnet.bybit.com/trade/usdt/BTCUSDT")
            logging.info("В секции 'Positions' должны быть видны TP/SL привязанные к позиции (не как отдельные ордера)")
            return True
        else:
            logging.error(f"❌ Ошибка установки trading stop: {error}")
            return False

    except Exception as exc:
        logging.error(f"❌ Исключение: {exc}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("ТЕСТ: Установка Trading Stop (SL/TP) на позицию")
    logging.info("="*60)

    success = test_trading_stop()

    logging.info("="*60)
    if success:
        logging.info("✅ ТЕСТ ПРОЙДЕН")
    else:
        logging.info("❌ ТЕСТ НЕ ПРОЙДЕН")
    logging.info("="*60)
