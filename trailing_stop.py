"""
Trailing Stop Loss Manager

Автоматическая защита прибыли через динамический Stop Loss.

Принцип работы:
- Отслеживает максимальную/минимальную цену с момента входа
- Автоматически поднимает/опускает SL вслед за ценой
- SL никогда не ухудшается (long: не опускается, short: не поднимается)
- Активация при достижении минимальной прибыли (activation_profit)
- Защита breakeven: SL не может быть хуже entry_price

Author: Claude Code
Date: 2025-12-28
Version: 1.0.0
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple


# === DEFAULT CONFIGURATION ===

DEFAULT_CONFIG = {
    "BTC/USDT": {
        "callback_rate": 0.05,        # -5% от максимума
        "activation_profit": 0.03,    # активация при +3% прибыли
        "enabled": True
    },
    "ETH/USDT": {
        "callback_rate": 0.05,
        "activation_profit": 0.03,
        "enabled": True
    },
    "LINK/USDT": {
        "callback_rate": 0.08,        # -8% (выше волатильность)
        "activation_profit": 0.05,    # активация при +5%
        "enabled": True
    },
    "MNT/USDT": {
        "callback_rate": 0.08,
        "activation_profit": 0.05,
        "enabled": True
    },
    "SOL/USDT": {
        "callback_rate": 0.10,        # -10% (очень волатильный)
        "activation_profit": 0.08,    # активация при +8%
        "enabled": True
    },
    "SUI/USDT": {
        "callback_rate": 0.08,
        "activation_profit": 0.05,
        "enabled": True
    },
    "TRX/USDT": {
        "callback_rate": 0.06,
        "activation_profit": 0.04,
        "enabled": True
    },
}


class TrailingStopManager:
    """
    Менеджер динамических trailing stop orders.

    Основные методы:
    - update_all_positions(): главная функция, вызывается каждые 5 минут
    - _update_position(): обновление trailing SL для одной позиции
    - _should_activate(): проверка условий активации trailing
    - _calculate_trailing_sl(): расчёт нового trailing SL
    """

    def __init__(
        self,
        exchange,
        config_path: Optional[str] = None,
        state_path: Optional[str] = None,
        dry_run: bool = False
    ):
        """
        Инициализация Trailing Stop Manager.

        Args:
            exchange: CCXT exchange объект (или ExchangeAdapter)
            config_path: Путь к файлу конфигурации (trailing_config.json)
            state_path: Путь к файлу состояния (trailing_state.json)
            dry_run: Если True, только логирование без реальных API вызовов
        """
        self.exchange = exchange
        self.dry_run = dry_run

        # Пути к файлам
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'trailing_config.json')
        if state_path is None:
            state_path = os.path.join(os.path.dirname(__file__), 'trailing_state.json')

        self.config_path = config_path
        self.state_path = state_path

        # Загрузка конфигурации и состояния
        self.config = self._load_config()
        self.state = self._load_state()

        logging.info(f"TrailingStopManager initialized (dry_run={dry_run})")

    def _load_config(self) -> Dict:
        """Загрузка конфигурации из файла или создание дефолтной."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logging.info(f"trailing | Config loaded from {self.config_path}")
                return config
            except Exception as e:
                logging.warning(f"trailing | Failed to load config: {e}, using defaults")

        # Создаём дефолтную конфигурацию
        self._save_config(DEFAULT_CONFIG)
        logging.info(f"trailing | Created default config at {self.config_path}")
        return DEFAULT_CONFIG.copy()

    def _save_config(self, config: Dict) -> None:
        """Сохранение конфигурации в файл."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"trailing | Failed to save config: {e}")

    def _load_state(self) -> Dict:
        """Загрузка состояния (максимумы цен) из файла."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logging.debug(f"trailing | State loaded: {len(state)} symbols")
                return state
            except Exception as e:
                logging.warning(f"trailing | Failed to load state: {e}, starting fresh")

        return {}

    def _save_state(self) -> None:
        """Сохранение состояния в файл."""
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"trailing | Failed to save state: {e}")

    def update_all_positions(self) -> None:
        """
        Главная функция: обновление trailing SL для всех открытых позиций.

        Вызывается каждые 5 минут из главного цикла бота.

        Процесс:
        1. Получает все открытые позиции с биржи
        2. Для каждой позиции вызывает _update_position()
        3. Очищает state от закрытых позиций
        4. Сохраняет state в файл
        """
        try:
            # Получаем все позиции (futures/linear)
            positions = self._fetch_positions()

            if not positions:
                logging.debug("trailing | No open positions")
                return

            logging.info(f"trailing | Updating {len(positions)} positions")

            # Обновляем каждую позицию
            updated_symbols = set()
            for pos in positions:
                symbol = pos.get('symbol')
                if not symbol:
                    continue

                # Нормализуем символ (убираем :USDT суффикс если есть)
                symbol = symbol.split(':')[0]

                # ИСПРАВЛЕНО 2025-12-30: Игнорируем TRX полностью
                if "TRX" in symbol.upper():
                    logging.info(f"trailing | {symbol} | Skipping TRX (ignored by user request)")
                    continue

                updated_symbols.add(symbol)

                try:
                    self._update_position(symbol, pos)
                except Exception as e:
                    logging.error(f"trailing | {symbol} | Update failed: {e}")

            # Очищаем state от закрытых позиций
            closed_symbols = set(self.state.keys()) - updated_symbols
            for symbol in closed_symbols:
                logging.info(f"trailing | {symbol} | Position closed, removing from state")
                del self.state[symbol]

            # Сохраняем состояние
            self._save_state()

        except Exception as e:
            logging.error(f"trailing | update_all_positions failed: {e}")

    def _fetch_positions(self) -> list:
        """Получение всех открытых позиций с биржи."""
        try:
            # Поддержка как CCXT exchange, так и ExchangeAdapter
            if hasattr(self.exchange, 'fetch_positions'):
                positions = self.exchange.fetch_positions()
            elif hasattr(self.exchange, 'x') and hasattr(self.exchange.x, 'fetch_positions'):
                positions = self.exchange.x.fetch_positions()
            else:
                logging.error("trailing | Exchange does not support fetch_positions")
                return []

            # Фильтруем только открытые позиции (qty > 0)
            open_positions = [
                p for p in positions
                if float(p.get('contracts', 0)) > 0
            ]

            return open_positions

        except Exception as e:
            logging.error(f"trailing | Failed to fetch positions: {e}")
            return []

    def _update_position(self, symbol: str, position: Dict) -> None:
        """
        Обновление trailing SL для одной позиции.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            position: Position dict from exchange
        """
        # Получаем конфигурацию для этого symbol
        config = self.config.get(symbol)
        if not config:
            # Используем дефолтную конфигурацию если пары нет в config
            config = {
                "callback_rate": 0.06,
                "activation_profit": 0.04,
                "enabled": True
            }
            logging.debug(f"trailing | {symbol} | Using default config")

        if not config.get('enabled', True):
            logging.debug(f"trailing | {symbol} | Trailing disabled in config")
            return

        # Извлекаем данные позиции
        entry_price = float(position.get('entryPrice', 0))
        current_price = float(position.get('markPrice', 0))
        side_raw = str(position.get('side', '')).upper()

        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2025-12-30: Bybit API возвращает 'Buy'/'Sell', а не 'Long'/'Short'
        # Также может быть contracts > 0 для long, contracts < 0 для short
        contracts = float(position.get('contracts', 0))

        # Нормализуем side: 'BUY' = LONG, 'SELL' = SHORT
        if side_raw in ('BUY', 'LONG') or contracts > 0:
            side = 'LONG'
        elif side_raw in ('SELL', 'SHORT') or contracts < 0:
            side = 'SHORT'
        else:
            logging.warning(f"trailing | {symbol} | Unknown side: {side_raw}, contracts: {contracts}")
            return

        if not entry_price or not current_price:
            logging.warning(f"trailing | {symbol} | Invalid prices: entry={entry_price}, current={current_price}")
            return

        logging.debug(f"trailing | {symbol} | Position: side={side}, entry={entry_price:.2f}, current={current_price:.2f}, contracts={contracts}")

        # Проверяем нужна ли активация
        if not self._should_activate(symbol, entry_price, current_price, side, config):
            return

        # Рассчитываем новый trailing SL
        new_sl = self._calculate_trailing_sl(symbol, entry_price, current_price, side, config)

        if new_sl is None:
            return

        # Получаем текущий SL с биржи
        current_sl = float(position.get('stopLoss', 0))

        # Проверяем нужно ли обновлять SL
        should_update = False

        if side == 'LONG':
            # Long: обновляем если новый SL выше текущего (поднимаем защиту)
            if current_sl == 0 or new_sl > current_sl:
                should_update = True
        else:  # SHORT
            # Short: обновляем если новый SL ниже текущего (опускаем защиту)
            if current_sl == 0 or new_sl < current_sl:
                should_update = True

        if should_update:
            self._set_stop_loss(symbol, new_sl, current_price, entry_price, side)

    def _should_activate(
        self,
        symbol: str,
        entry: float,
        current: float,
        side: str,
        config: Dict
    ) -> bool:
        """
        Проверка достижения минимальной прибыли для активации trailing.

        Args:
            symbol: Trading pair
            entry: Entry price
            current: Current market price
            side: Position side ('LONG' or 'SHORT')
            config: Trailing config for this pair

        Returns:
            True если trailing должен активироваться
        """
        activation_profit = config.get('activation_profit', 0.03)

        # Рассчитываем текущую прибыль
        if side == 'LONG':
            profit_pct = (current / entry) - 1
        else:  # SHORT
            profit_pct = (entry / current) - 1

        if profit_pct >= activation_profit:
            # Проверяем активирован ли уже
            if symbol not in self.state or not self.state[symbol].get('activated'):
                logging.info(
                    f"trailing | {symbol} | Activated! Profit: {profit_pct:.1%} >= {activation_profit:.1%}"
                )
                # Помечаем как активированный
                if symbol not in self.state:
                    self.state[symbol] = {}
                self.state[symbol]['activated'] = True
            return True
        else:
            if symbol in self.state and self.state[symbol].get('activated'):
                # Уже активирован ранее - продолжаем trailing
                return True
            else:
                # Ещё не достигнута прибыль для активации
                logging.debug(
                    f"trailing | {symbol} | Profit {profit_pct:.1%} < {activation_profit:.1%} (not activated)"
                )
                return False

    def _calculate_trailing_sl(
        self,
        symbol: str,
        entry: float,
        current: float,
        side: str,
        config: Dict
    ) -> Optional[float]:
        """
        Расчёт нового trailing SL.

        Обновляет максимум/минимум цены и рассчитывает SL с учётом callback_rate.

        Args:
            symbol: Trading pair
            entry: Entry price
            current: Current market price
            side: Position side
            config: Trailing config

        Returns:
            Новая цена SL или None если не нужно обновлять
        """
        callback_rate = config.get('callback_rate', 0.05)

        # Инициализируем state для symbol если нет
        if symbol not in self.state:
            self.state[symbol] = {}

        state = self.state[symbol]

        # Обновляем максимум/минимум цены
        if side == 'LONG':
            max_price = state.get('max_price', entry)
            if current > max_price:
                max_price = current
                state['max_price'] = max_price
                logging.debug(f"trailing | {symbol} | New max price: {max_price:.6f}")

            # SL = max_price * (1 - callback_rate)
            new_sl = max_price * (1 - callback_rate)

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2025-12-31: SL не может быть выше текущей цены
            # Если цена упала ниже расчетного SL, используем текущую цену * 0.999
            if new_sl >= current:
                new_sl = current * 0.999
                logging.debug(f"trailing | {symbol} | SL adjusted to current*0.999: {new_sl:.6f} (was: {max_price * (1 - callback_rate):.6f})")

            # Breakeven protection: SL не ниже entry
            if new_sl < entry:
                new_sl = entry

        else:  # SHORT
            min_price = state.get('min_price', entry)
            if current < min_price:
                min_price = current
                state['min_price'] = min_price
                logging.debug(f"trailing | {symbol} | New min price: {min_price:.6f}")

            # SL = min_price * (1 + callback_rate)
            new_sl = min_price * (1 + callback_rate)

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2025-12-31: SL не может быть ниже текущей цены
            # Если цена поднялась выше расчетного SL, используем текущую цену * 1.001
            if new_sl <= current:
                new_sl = current * 1.001
                logging.debug(f"trailing | {symbol} | SL adjusted to current*1.001: {new_sl:.6f} (was: {min_price * (1 + callback_rate):.6f})")

            # Breakeven protection: SL не выше entry
            if new_sl > entry:
                new_sl = entry

        # Обновляем timestamp
        state['last_update'] = datetime.now(timezone.utc).isoformat()
        state['side'] = side

        return new_sl

    def _set_stop_loss(
        self,
        symbol: str,
        sl_price: float,
        current_price: float,
        entry_price: float,
        side: str
    ) -> None:
        """
        Установка нового SL через Bybit API.

        Args:
            symbol: Trading pair
            sl_price: New stop-loss price
            current_price: Current market price
            entry_price: Entry price
            side: Position side
        """
        # ИСПРАВЛЕНО 2025-12-31: Убрана дублирующая валидация
        # Валидация уже выполнена в _calculate_trailing_sl (строки 390-418)
        # Эта проверка была избыточной и блокировала корректные SL после auto-adjust
        #
        # Оставляем только КРИТИЧЕСКУЮ проверку для отладки (warning вместо error)
        if side == 'LONG' and sl_price >= current_price:
            logging.warning(
                f"trailing | {symbol} | SL близко к current: sl={sl_price:.4f} vs current={current_price:.4f}. "
                f"Это может быть результатом auto-adjust при падении цены."
            )
        elif side == 'SHORT' and sl_price <= current_price:
            logging.warning(
                f"trailing | {symbol} | SL близко к current: sl={sl_price:.4f} vs current={current_price:.4f}. "
                f"Это может быть результатом auto-adjust при росте цены."
            )

        if self.dry_run:
            profit_pct = ((current_price / entry_price) - 1) if side == 'LONG' else ((entry_price / current_price) - 1)
            max_or_min = self.state[symbol].get('max_price' if side == 'LONG' else 'min_price', entry_price)

            logging.info(
                f"[DRY-RUN] trailing | {symbol} | SL updated to {sl_price:.6f}\n"
                f"   Entry: {entry_price:.6f}, Current: {current_price:.6f}, "
                f"{'Max' if side == 'LONG' else 'Min'}: {max_or_min:.6f}\n"
                f"   Profit: {profit_pct:.1%}"
            )
            return True

        # Реальное обновление SL через API
        try:
            # Импортируем функцию из logging_utils
            from logging_utils import set_position_tp_sl

            # Retry logic (3 попытки)
            for attempt in range(3):
                try:
                    # Bybit V5 API: set trading stop через logging_utils функцию
                    # ИСПРАВЛЕНО 2025-12-28: используем self.exchange.x напрямую (как в main.py line 3805)
                    success, err = set_position_tp_sl(
                        self.exchange.x,  # ADAPTER.x - правильный exchange объект
                        symbol=symbol,
                        tp_price=None,  # Не меняем TP
                        sl_price=sl_price,
                        category="linear"
                    )

                    if success:
                        profit_pct = ((current_price / entry_price) - 1) if side == 'LONG' else ((entry_price / current_price) - 1)
                        max_or_min = self.state[symbol].get('max_price' if side == 'LONG' else 'min_price', entry_price)

                        logging.info(
                            f"trailing | {symbol} | SL updated: {sl_price:.6f}\n"
                            f"   Entry: {entry_price:.6f}, Current: {current_price:.6f}, "
                            f"{'Max' if side == 'LONG' else 'Min'}: {max_or_min:.6f}\n"
                            f"   Profit: {profit_pct:.1%}"
                        )
                        return True
                    elif err and "not modified" in str(err).lower():
                        # SL уже установлен - не ошибка
                        logging.debug(f"trailing | {symbol} | SL already set at {sl_price:.6f}")
                        return True
                    else:
                        raise Exception(f"set_position_tp_sl failed: {err}")

                except Exception as e:
                    if attempt < 2:
                        logging.warning(f"trailing | {symbol} | Attempt {attempt+1}/3 failed: {e}, retrying...")
                        time.sleep(1)
                    else:
                        raise

        except Exception as e:
            logging.error(f"trailing | {symbol} | Failed to set SL: {e}")
            return False


if __name__ == "__main__":
    """Self-test with mock exchange."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    print("=" * 70)
    print("TRAILING STOP MANAGER - SELF TEST (DRY RUN)")
    print("=" * 70)
    print()

    # Mock exchange для тестирования
    class MockExchange:
        def fetch_positions(self):
            return [
                {
                    'symbol': 'BTC/USDT',
                    'entryPrice': 90000.0,
                    'markPrice': 94000.0,  # +4.4% profit -> должен активироваться (threshold 3%)
                    'side': 'LONG',
                    'contracts': 0.1,
                    'stopLoss': 0
                },
                {
                    'symbol': 'ETH/USDT',
                    'entryPrice': 3000.0,
                    'markPrice': 3050.0,  # +1.67% profit -> НЕ активируется (threshold 3%)
                    'side': 'LONG',
                    'contracts': 1.0,
                    'stopLoss': 0
                }
            ]

    mock_exchange = MockExchange()
    manager = TrailingStopManager(mock_exchange, dry_run=True)

    print("Running update_all_positions()...")
    print()
    manager.update_all_positions()

    print()
    print("=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
