"""
Native Bybit API V5 Trailing Stop Manager

Использует нативную функциональность Bybit API V5 для создания
скользящих стоп-ордеров, которые отображаются в интерфейсе биржи
во вкладке "Скользящий стоп-ордер".

Основные отличия от старого trailing_stop.py:
- Использует параметр trailingStop в API /v5/position/trading-stop
- Работает на стороне биржи (не требует постоянного обновления бота)
- Отображается в интерфейсе Bybit как "Скользящий стоп-ордер"
- Коррекция задается в долларах (по сумме), а не в процентах

API Documentation:
https://bybit-exchange.github.io/docs/v5/position/trading-stop

Author: Claude Code
Date: 2025-12-29
Version: 2.0.0
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple


# === OPTIMAL TRAILING DISTANCE CONFIGURATION ===
# На основе исследования волатильности крипторынка (2025)
# https://goodcrypto.app/bybit-trailing-stop-orders/
# https://medium.com/bybit/cryptocurrency-trading-exit-strategy-c377a97561b7

DEFAULT_TRAILING_CONFIG = {
    "BTC/USDT": {
        "trailing_distance_pct": 0.06,    # 6% - оптимально для BTC
        "activation_price_pct": 0.03,      # Активация при +3% прибыли
        "enabled": True,
        "description": "Bitcoin - средняя волатильность, 6% коррекция"
    },
    "ETH/USDT": {
        "trailing_distance_pct": 0.06,
        "activation_price_pct": 0.03,
        "enabled": True,
        "description": "Ethereum - средняя волатильность, 6% коррекция"
    },
    "SOL/USDT": {
        "trailing_distance_pct": 0.10,     # 10% - высокая волатильность
        "activation_price_pct": 0.05,      # Активация при +5% прибыли
        "enabled": True,
        "description": "Solana - высокая волатильность, 10% коррекция"
    },
    "LINK/USDT": {
        "trailing_distance_pct": 0.08,
        "activation_price_pct": 0.04,
        "enabled": True,
        "description": "Chainlink - повышенная волатильность, 8% коррекция"
    },
    "MNT/USDT": {
        "trailing_distance_pct": 0.08,
        "activation_price_pct": 0.04,
        "enabled": True,
        "description": "Mantle - повышенная волатильность, 8% коррекция"
    },
    "SUI/USDT": {
        "trailing_distance_pct": 0.08,
        "activation_price_pct": 0.04,
        "enabled": True,
        "description": "Sui - повышенная волатильность, 8% коррекция"
    },
    "TRX/USDT": {
        "trailing_distance_pct": 0.07,
        "activation_price_pct": 0.03,
        "enabled": True,
        "description": "Tron - умеренная волатильность, 7% коррекция"
    },
}


class NativeTrailingStopManager:
    """
    Менеджер нативных trailing stop orders через Bybit API V5.

    Ключевые методы:
    - set_trailing_stop(): Устанавливает trailing stop на существующую позицию
    - calculate_trailing_distance(): Рассчитывает дистанцию в долларах для текущей цены
    - update_all_positions(): Обновляет trailing stops для всех позиций

    Технические детали:
    - trailingStop задается в ДОЛЛАРАХ (price distance), не в процентах
    - activePrice - цена активации (опционально, можно активировать сразу)
    - Работает только с существующими позициями (не с новыми ордерами)
    """

    def __init__(
        self,
        exchange,
        config_path: Optional[str] = None,
        dry_run: bool = False
    ):
        """
        Инициализация Native Trailing Stop Manager.

        Args:
            exchange: CCXT exchange объект (или ExchangeAdapter)
            config_path: Путь к файлу конфигурации
            dry_run: Если True, только логирование без реальных API вызовов
        """
        self.exchange = exchange
        self.dry_run = dry_run

        # Путь к конфигурации
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                'native_trailing_config.json'
            )

        self.config_path = config_path
        self.config = self._load_config()

        logging.info(
            f"NativeTrailingStopManager initialized (dry_run={dry_run})"
        )

    def _load_config(self) -> Dict:
        """Загрузка конфигурации из файла или создание дефолтной."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logging.info(
                    f"native_trailing | Config loaded from {self.config_path}"
                )
                return config
            except Exception as e:
                logging.warning(
                    f"native_trailing | Failed to load config: {e}, "
                    "using defaults"
                )

        # Создаём дефолтную конфигурацию
        self._save_config(DEFAULT_TRAILING_CONFIG)
        logging.info(
            f"native_trailing | Created default config at {self.config_path}"
        )
        return DEFAULT_TRAILING_CONFIG.copy()

    def _save_config(self, config: Dict) -> None:
        """Сохранение конфигурации в файл."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"native_trailing | Failed to save config: {e}")

    def calculate_trailing_distance(
        self,
        symbol: str,
        current_price: float,
        trailing_pct: Optional[float] = None
    ) -> float:
        """
        Рассчитывает trailing distance в ДОЛЛАРАХ для текущей цены.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            current_price: Текущая рыночная цена
            trailing_pct: Процент коррекции (опционально, берется из config)

        Returns:
            Trailing distance в долларах

        Example:
            BTC по цене $90000, trailing_pct=6%
            → trailing_distance = $5400
        """
        if trailing_pct is None:
            config = self.config.get(symbol, {})
            trailing_pct = config.get('trailing_distance_pct', 0.06)

        # Рассчитываем дистанцию в долларах
        trailing_distance = current_price * trailing_pct

        logging.debug(
            f"native_trailing | {symbol} | Trailing distance: "
            f"${trailing_distance:.2f} ({trailing_pct:.1%} of ${current_price:.2f})"
        )

        return trailing_distance

    def calculate_activation_price(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        activation_pct: Optional[float] = None
    ) -> Optional[float]:
        """
        Рассчитывает цену активации trailing stop.

        Args:
            symbol: Trading pair
            entry_price: Цена входа в позицию
            side: 'LONG' или 'SHORT'
            activation_pct: Процент прибыли для активации (None = активация сразу)

        Returns:
            Activation price или None (если активация сразу)

        Example:
            Entry=$90000, side=LONG, activation_pct=3%
            → activation_price = $92700

            activation_pct=None → None (активация сразу)
        """
        if activation_pct is None:
            config = self.config.get(symbol, {})
            activation_pct = config.get('activation_price_pct')

            # Если в конфиге тоже None - возвращаем None (активация сразу)
            if activation_pct is None:
                logging.debug(
                    f"native_trailing | {symbol} | Activation: IMMEDIATE (no delay)"
                )
                return None

        if side.upper() in ('BUY', 'LONG'):
            activation_price = entry_price * (1 + activation_pct)
        else:  # SHORT
            activation_price = entry_price * (1 - activation_pct)

        logging.debug(
            f"native_trailing | {symbol} | Activation price: "
            f"${activation_price:.2f} ({activation_pct:.1%} from entry ${entry_price:.2f})"
        )

        return activation_price

    def set_trailing_stop(
        self,
        symbol: str,
        trailing_distance: Optional[float] = None,
        activation_price: Optional[float] = None,
        category: str = "linear",
        position_idx: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Устанавливает trailing stop на существующую позицию.

        Args:
            symbol: Символ торговой пары (e.g., 'BTCUSDT')
            trailing_distance: Дистанция в долларах (если None, рассчитывается автоматически)
            activation_price: Цена активации (если None, активируется сразу)
            category: Категория рынка ("linear", "inverse")
            position_idx: Position index (0=oneway, 1=long hedge, 2=short hedge)

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)

        Example API Call:
            POST /v5/position/trading-stop
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "trailingStop": "5400",      # $5400 trailing distance
                "activePrice": "92700",       # Активация при $92700
                "positionIdx": 0,
                "tpslMode": "Full"
            }
        """
        # Нормализуем символ для Bybit (убираем / и оставляем только базовый символ)
        # Поддержка формата: BTC/USDT, BTC/USDT:USDT, BTCUSDT
        clean_symbol = symbol.split(':')[0]  # Убираем :USDT если есть
        norm_symbol = clean_symbol.replace('/', '')  # BTC/USDT -> BTCUSDT

        # Получаем информацию о позиции
        try:
            # ИСПРАВЛЕНО: Получаем ВСЕ позиции (фильтрация по symbol не работает в Bybit)
            all_positions = self.exchange.fetch_positions(params={"category": category})

            if not all_positions:
                return False, "No positions found"

            # Фильтруем по символу (учитываем разные форматы)
            position = None
            for pos in all_positions:
                size = float(pos.get('contracts', 0) or pos.get('size', 0))
                pos_symbol = pos.get('symbol', '')
                # Проверяем совпадение
                symbol_match = (
                    pos_symbol == symbol or
                    pos_symbol == f"{symbol}:USDT" or
                    pos_symbol.split(':')[0] == symbol or
                    pos_symbol.split(':')[0] == symbol.split(':')[0]
                )
                if size > 0 and symbol_match:
                    position = pos
                    logging.debug(f"native_trailing | {symbol} | Matched position: {pos_symbol}")
                    break

            if not position:
                return False, "No open position"

            # Извлекаем данные позиции
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
            side = str(position.get('side', '')).upper()

            if not entry_price or not current_price:
                return False, f"Invalid prices: entry={entry_price}, current={current_price}"

        except Exception as e:
            error_msg = f"Failed to fetch position: {e}"
            logging.error(f"native_trailing | {symbol} | {error_msg}")
            return False, error_msg

        # Определяем positionIdx если не задан
        if position_idx is None:
            # Получаем из позиции или ставим 0 (oneway mode)
            info = position.get('info', {})
            position_idx = int(info.get('positionIdx', 0))

        # Рассчитываем trailing distance если не задан
        if trailing_distance is None:
            trailing_distance = self.calculate_trailing_distance(
                symbol,
                current_price
            )

        # Рассчитываем activation price если не задан
        if activation_price is None:
            activation_price = self.calculate_activation_price(
                symbol,
                entry_price,
                side
            )

        # Округляем до precision
        try:
            trailing_distance_str = str(self.exchange.price_to_precision(
                symbol,
                trailing_distance
            ))
        except Exception:
            trailing_distance_str = f"{trailing_distance:.2f}"

        # ИЗМЕНЕНО: activation_price может быть None (активация сразу)
        activation_price_str = None
        if activation_price is not None:
            # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ: проверяем что activation price логичен
            if side in ('BUY', 'LONG'):
                if activation_price <= entry_price:
                    logging.warning(
                        f"native_trailing | {symbol} | Invalid activation price for LONG: "
                        f"activation={activation_price:.2f} <= entry={entry_price:.2f}. "
                        f"Setting to None (immediate activation)"
                    )
                    activation_price = None
            else:  # SHORT
                if activation_price >= entry_price:
                    logging.warning(
                        f"native_trailing | {symbol} | Invalid activation price for SHORT: "
                        f"activation={activation_price:.2f} >= entry={entry_price:.2f}. "
                        f"Setting to None (immediate activation)"
                    )
                    activation_price = None

            # Округляем если activation_price валиден
            if activation_price is not None:
                try:
                    activation_price_str = str(self.exchange.price_to_precision(
                        symbol,
                        activation_price
                    ))
                except Exception:
                    activation_price_str = f"{activation_price:.2f}"

        if self.dry_run:
            activation_str = f"${activation_price:.2f}" if activation_price is not None else "IMMEDIATE"

            logging.info(
                f"[DRY-RUN] native_trailing | {symbol} | Trailing stop configured:\n"
                f"   Entry: ${entry_price:.2f}, Current: ${current_price:.2f}\n"
                f"   Trailing Distance: ${trailing_distance:.2f} "
                f"({(trailing_distance/current_price):.1%})\n"
                f"   Activation: {activation_str}\n"
                f"   Position Idx: {position_idx}, Side: {side}"
            )
            return True, None

        # Реальное API обращение к Bybit
        try:
            # Получаем метод set_trading_stop
            set_trading_stop_fn = getattr(
                self.exchange,
                "set_trading_stop",
                None
            ) or getattr(self.exchange, "setTradingStop", None)

            if not callable(set_trading_stop_fn):
                return False, "set_trading_stop method not available"

            # ВАЖНО: Параметры для нативного trailing stop
            params = {
                "category": category,
                "positionIdx": position_idx,
                "tpslMode": "Full",  # Full = весь размер позиции
                "trailingStop": trailing_distance_str,  # Коррекция в долларах
            }

            # Добавляем activePrice только если он задан (иначе активация сразу)
            if activation_price_str is not None:
                params["activePrice"] = activation_price_str

            # Вызываем API
            response = set_trading_stop_fn(
                norm_symbol,
                params=params
            )

            activation_info = f"${activation_price:.2f}" if activation_price is not None else "IMMEDIATE"
            logging.info(
                f"native_trailing | {symbol} | Trailing stop SET successfully:\n"
                f"   Entry: ${entry_price:.2f}, Current: ${current_price:.2f}\n"
                f"   Trailing Distance: ${trailing_distance:.2f} "
                f"({(trailing_distance/current_price):.1%})\n"
                f"   Activation Price: {activation_info}\n"
                f"   Position Idx: {position_idx}, Side: {side}\n"
                f"   Response: {response}"
            )

            return True, None

        except Exception as exc:
            error_msg = str(exc)

            # Обработка специфичных ошибок Bybit
            if "34040" in error_msg or "not modified" in error_msg.lower():
                logging.info(
                    f"native_trailing | {symbol} | Already set (retCode=34040)"
                )
                return True, None  # Не ошибка

            elif "10001" in error_msg and "zero position" in error_msg.lower():
                logging.debug(
                    f"native_trailing | {symbol} | Position closed (retCode=10001)"
                )
                return False, "Position closed"

            else:
                logging.error(
                    f"native_trailing | {symbol} | Failed to set trailing stop: "
                    f"{error_msg}"
                )
                return False, error_msg

    def update_all_positions(self) -> None:
        """
        Обновляет trailing stops для всех открытых позиций.

        Вызывается периодически из главного цикла бота.

        Процесс:
        1. Получает все открытые позиции
        2. Для каждой позиции проверяет наличие trailing stop
        3. Устанавливает trailing stop если его нет
        """
        try:
            # Получаем все позиции
            positions = self.exchange.fetch_positions()

            if not positions:
                logging.debug("native_trailing | No open positions")
                return

            open_positions = [
                p for p in positions
                if float(p.get('contracts', 0) or p.get('size', 0)) > 0
            ]

            if not open_positions:
                logging.debug("native_trailing | No open positions")
                return

            logging.info(
                f"native_trailing | Updating {len(open_positions)} positions"
            )

            for pos in open_positions:
                symbol = pos.get('symbol')
                if not symbol:
                    continue

                # Нормализуем символ (убираем :USDT суффикс)
                symbol = symbol.split(':')[0]

                # Проверяем конфигурацию
                config = self.config.get(symbol)
                if config and not config.get('enabled', True):
                    logging.debug(
                        f"native_trailing | {symbol} | Disabled in config"
                    )
                    continue

                # Проверяем есть ли уже trailing stop
                info = pos.get('info', {})
                existing_trailing = info.get('trailingStop', '0')

                if existing_trailing and str(existing_trailing) not in ('0', '0.0', ''):
                    logging.debug(
                        f"native_trailing | {symbol} | Already has trailing stop: "
                        f"{existing_trailing}"
                    )
                    continue

                # Устанавливаем trailing stop
                try:
                    success, err = self.set_trailing_stop(
                        symbol,
                        category="linear"
                    )

                    if not success and err:
                        logging.warning(
                            f"native_trailing | {symbol} | Failed: {err}"
                        )

                except Exception as e:
                    logging.error(
                        f"native_trailing | {symbol} | Update failed: {e}"
                    )

        except Exception as e:
            logging.error(
                f"native_trailing | update_all_positions failed: {e}"
            )


# === HELPER FUNCTIONS ===

def set_native_trailing_stop(
    exchange,
    symbol: str,
    trailing_distance_pct: Optional[float] = None,
    activation_price_pct: Optional[float] = None,
    category: str = "linear"
) -> Tuple[bool, Optional[str]]:
    """
    Удобная функция для установки trailing stop на позицию.

    Args:
        exchange: CCXT exchange объект
        symbol: Торговая пара (e.g., 'BTC/USDT')
        trailing_distance_pct: Процент коррекции (если None, берется из конфига)
        activation_price_pct: Процент прибыли для активации
        category: Категория рынка

    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)

    Example:
        >>> set_native_trailing_stop(
        ...     exchange,
        ...     'BTC/USDT',
        ...     trailing_distance_pct=0.06,  # 6%
        ...     activation_price_pct=0.03     # активация при +3%
        ... )
        (True, None)
    """
    manager = NativeTrailingStopManager(exchange, dry_run=False)

    # Получаем текущую цену
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = float(ticker['last'])
    except Exception as e:
        return False, f"Failed to fetch ticker: {e}"

    # Получаем позицию для определения entry price и side
    try:
        logging.debug(f"native_trailing | {symbol} | Fetching all positions with category={category}")
        # ИСПРАВЛЕНО: Получаем ВСЕ позиции, т.к. фильтрация по символу не работает
        all_positions = exchange.fetch_positions(params={"category": category})
        logging.debug(f"native_trailing | {symbol} | Got {len(all_positions)} total positions")

        # Фильтруем по символу (учитываем разные форматы: ETH/USDT, ETH/USDT:USDT)
        position = None
        for pos in all_positions:
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            pos_symbol = pos.get('symbol', '')
            # Проверяем совпадение с учётом разных форматов
            symbol_match = (
                pos_symbol == symbol or
                pos_symbol == f"{symbol}:USDT" or
                pos_symbol.split(':')[0] == symbol or
                pos_symbol.split(':')[0] == symbol.split(':')[0]
            )
            logging.debug(f"native_trailing | {symbol} | Checking position: {pos_symbol}, size={size}, match={symbol_match}")
            if size > 0 and symbol_match:
                position = pos
                logging.info(f"native_trailing | {symbol} | Found matching position: {pos_symbol}")
                break

        if not position:
            logging.warning(f"native_trailing | {symbol} | No open position found among {len(all_positions)} positions")
            return False, "No open position"

        entry_price = float(position.get('entryPrice', 0))
        side = str(position.get('side', '')).upper()

    except Exception as e:
        return False, f"Failed to fetch position: {e}"

    # Рассчитываем параметры
    if trailing_distance_pct is not None:
        trailing_distance = current_price * trailing_distance_pct
    else:
        trailing_distance = manager.calculate_trailing_distance(symbol, current_price)

    if activation_price_pct is not None:
        if side in ('BUY', 'LONG'):
            activation_price = entry_price * (1 + activation_price_pct)
        else:
            activation_price = entry_price * (1 - activation_price_pct)
    else:
        activation_price = manager.calculate_activation_price(symbol, entry_price, side)

    # Устанавливаем trailing stop
    return manager.set_trailing_stop(
        symbol,
        trailing_distance=trailing_distance,
        activation_price=activation_price,
        category=category
    )


if __name__ == "__main__":
    """Self-test with mock exchange."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    print("=" * 70)
    print("NATIVE TRAILING STOP MANAGER - SELF TEST (DRY RUN)")
    print("=" * 70)
    print()

    # Mock exchange для тестирования
    class MockExchange:
        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    'symbol': 'BTC/USDT',
                    'entryPrice': 90000.0,
                    'markPrice': 94000.0,
                    'side': 'LONG',
                    'contracts': 0.1,
                    'info': {
                        'positionIdx': 0,
                        'trailingStop': '0'
                    }
                }
            ]

        def price_to_precision(self, symbol, price):
            return round(price, 2)

    mock_exchange = MockExchange()
    manager = NativeTrailingStopManager(mock_exchange, dry_run=True)

    print("Test 1: Calculate trailing distance for BTC at $94000")
    distance = manager.calculate_trailing_distance('BTC/USDT', 94000)
    print(f"  -> Trailing Distance: ${distance:.2f}\n")

    print("Test 2: Calculate activation price for LONG entry at $90000")
    activation = manager.calculate_activation_price('BTC/USDT', 90000, 'LONG')
    print(f"  -> Activation Price: ${activation:.2f}\n")

    print("Test 3: Set trailing stop on BTC position")
    success, err = manager.set_trailing_stop('BTC/USDT')
    print(f"  -> Success: {success}, Error: {err}\n")

    print("=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
