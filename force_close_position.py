"""
Скрипт для принудительного закрытия позиции на Bybit testnet.
Использует несколько методов для гарантированного закрытия.
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

def force_close_position(symbol="TRX/USDT:USDT"):
    """Принудительно закрываем позицию."""

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
    logging.info(f"🎯 Попытка закрыть позицию: {symbol}")

    category = "linear"

    # Импортируем нужные функции
    from logging_utils import has_open_position, _normalize_bybit_symbol

    # Нормализуем символ
    norm_symbol = _normalize_bybit_symbol(exchange, symbol, category)
    logging.info(f"📋 Нормализованный символ: {norm_symbol}")

    # Проверяем позицию
    qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

    if qty_abs <= 0:
        logging.warning(f"❌ Нет открытой позиции для {symbol}")

        # Попробуем альтернативные форматы
        for alt_symbol in ["TRXUSDT", "TRX/USDT", symbol]:
            logging.info(f"🔄 Пробуем {alt_symbol}...")
            qty_signed, qty_abs = has_open_position(exchange, alt_symbol, category)
            if qty_abs > 0:
                norm_symbol = alt_symbol
                logging.info(f"✅ Найдена позиция для {alt_symbol}")
                break

        if qty_abs <= 0:
            return False

    is_long = qty_signed > 0
    side = "long" if is_long else "short"
    close_side = "sell" if is_long else "buy"

    logging.info(f"📊 Найдена позиция: {side.upper()} | qty={qty_abs} (signed={qty_signed})")

    # Получаем текущую цену
    try:
        ticker = exchange.fetch_ticker(norm_symbol)
        current_price = float(ticker['last'])
        logging.info(f"💰 Текущая цена: {current_price}")
    except Exception as e:
        logging.warning(f"Не удалось получить цену: {e}")
        current_price = None

    # Сначала отменим все открытые ордера для этого символа
    logging.info("🧹 Отменяем все открытые ордера...")
    try:
        orders = exchange.fetch_open_orders(norm_symbol, params={"category": category})
        logging.info(f"📋 Найдено открытых ордеров: {len(orders)}")

        for order in orders:
            try:
                order_id = order.get('id')
                exchange.cancel_order(order_id, norm_symbol, params={"category": category})
                logging.info(f"   ✅ Отменен ордер: {order_id}")
            except Exception as e:
                logging.warning(f"   ⚠️ Не удалось отменить ордер: {e}")
    except Exception as e:
        logging.warning(f"Ошибка при отмене ордеров: {e}")

    # Удаляем SL/TP если есть
    logging.info("🧹 Удаляем SL/TP...")
    try:
        from logging_utils import set_position_tp_sl
        success, err = set_position_tp_sl(
            exchange,
            norm_symbol,
            tp_price=0,  # 0 = удалить
            sl_price=0,  # 0 = удалить
            category=category,
            side_open=close_side,
        )
        if success:
            logging.info("   ✅ SL/TP удалены")
        else:
            logging.warning(f"   ⚠️ Не удалось удалить SL/TP: {err}")
    except Exception as e:
        logging.warning(f"   ⚠️ Ошибка при удалении SL/TP: {e}")

    # Определяем hedge mode
    from logging_utils import _force_hedge_mode_check
    is_hedge = _force_hedge_mode_check(exchange, norm_symbol, category)

    position_idx = 0
    if is_hedge:
        # В hedge mode: 1=Long, 2=Short
        position_idx = 1 if is_long else 2
        logging.info(f"🔍 HEDGE MODE detected: positionIdx={position_idx}")
    else:
        logging.info(f"🔍 ONE-WAY MODE detected: positionIdx=0")

    # МЕТОД 1: Попытка закрыть через market order с reduceOnly
    logging.info("="*60)
    logging.info("МЕТОД 1: Market order с reduceOnly=True и правильным positionIdx")
    logging.info("="*60)

    try:
        params = {
            "category": category,
            "reduceOnly": True,
            "closeOnTrigger": False,
            "positionIdx": position_idx,  # КРИТИЧНО для hedge mode!
        }

        # Округляем количество
        from logging_utils import _round_qty
        qty_rounded = _round_qty(exchange, norm_symbol, qty_abs)

        logging.info(f"📤 Создаем market order: {close_side} {qty_rounded} {norm_symbol} (positionIdx={position_idx})")

        order = exchange.create_order(
            norm_symbol,
            "market",
            close_side,
            qty_rounded,
            None,
            params
        )

        logging.info(f"✅ Ордер создан: {order.get('id')}")

        # Ждем 2 секунды
        time.sleep(2)

        # Проверяем позицию
        qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

        if qty_abs <= 0:
            logging.info("="*60)
            logging.info("🎉 ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА!")
            logging.info("="*60)
            return True
        else:
            logging.warning(f"⚠️ Позиция еще открыта: qty={qty_abs}")

    except Exception as e:
        logging.error(f"❌ Метод 1 не сработал: {e}")

    # МЕТОД 2: Попытка закрыть через create_reduce_only_order
    logging.info("="*60)
    logging.info("МЕТОД 2: Reduce-only order")
    logging.info("="*60)

    try:
        # Проверяем позицию снова
        qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

        if qty_abs <= 0:
            logging.info("✅ Позиция уже закрыта")
            return True

        qty_rounded = _round_qty(exchange, norm_symbol, qty_abs)

        params = {
            "category": category,
            "reduceOnly": True,
            "positionIdx": position_idx,  # КРИТИЧНО для hedge mode!
        }

        logging.info(f"📤 Создаем limit order по текущей цене (positionIdx={position_idx})")

        # Используем лимит ордер по текущей цене
        order = exchange.create_order(
            norm_symbol,
            "limit",
            close_side,
            qty_rounded,
            current_price,
            params
        )

        logging.info(f"✅ Ордер создан: {order.get('id')}")

        # Ждем 3 секунды
        time.sleep(3)

        # Проверяем позицию
        qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

        if qty_abs <= 0:
            logging.info("="*60)
            logging.info("🎉 ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА!")
            logging.info("="*60)
            return True

    except Exception as e:
        logging.error(f"❌ Метод 2 не сработал: {e}")

    # МЕТОД 3: Попытка через приватный API endpoint
    logging.info("="*60)
    logging.info("МЕТОД 3: Прямой вызов close position API")
    logging.info("="*60)

    try:
        # Проверяем позицию снова
        qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

        if qty_abs <= 0:
            logging.info("✅ Позиция уже закрыта")
            return True

        # Проверяем, есть ли метод для закрытия позиции
        close_position_fn = getattr(exchange, "private_post_v5_position_close", None)

        if close_position_fn:
            # Получаем символ в формате Bybit
            bybit_symbol = norm_symbol.replace("/", "").replace(":USDT", "")

            params = {
                "category": category,
                "symbol": bybit_symbol,
            }

            logging.info(f"📤 Вызываем close position API: {params}")

            response = close_position_fn(params)
            logging.info(f"📥 Ответ: {response}")

            # Ждем 2 секунды
            time.sleep(2)

            # Проверяем позицию
            qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

            if qty_abs <= 0:
                logging.info("="*60)
                logging.info("🎉 ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА!")
                logging.info("="*60)
                return True

        else:
            logging.warning("⚠️ Метод close position API не найден")

    except Exception as e:
        logging.error(f"❌ Метод 3 не сработал: {e}")

    # Финальная проверка
    qty_signed, qty_abs = has_open_position(exchange, norm_symbol, category)

    if qty_abs <= 0:
        logging.info("="*60)
        logging.info("🎉 ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА!")
        logging.info("="*60)
        return True
    else:
        logging.error("="*60)
        logging.error(f"❌ НЕ УДАЛОСЬ ЗАКРЫТЬ ПОЗИЦИЮ: qty={qty_abs}")
        logging.error("="*60)
        logging.error("Попробуйте закрыть позицию вручную на веб-интерфейсе:")
        logging.error("https://testnet.bybit.com/trade/usdt/TRXUSDT")
        return False


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ ПОЗИЦИИ TRX")
    logging.info("="*60)

    success = force_close_position("TRX/USDT:USDT")

    if not success:
        logging.info("")
        logging.info("Если скрипт не помог, попробуйте:")
        logging.info("1. Зайти на https://testnet.bybit.com/trade/usdt/TRXUSDT")
        logging.info("2. Отключить изолированную маржу (переключить на кросс)")
        logging.info("3. Попробовать закрыть позицию снова")
        logging.info("4. Или обнулить баланс testnet через faucet")
