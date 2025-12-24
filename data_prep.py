"""Модуль подготовки данных для обучения модели"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

def build_feature_dataframe(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """
    Создаёт DataFrame с признаками из OHLCV данных.
    
    КРИТИЧНО: Добавляет разнообразие в признаки через:
    1. Лаговые переменные (ret_1, ret_5)
    2. Скользящие средние разных периодаов
    3. RSI, ADX и другие индикаторы
    4. Категориальные признаки (symbol_cat)
    """
    if df is None or df.empty:
        logging.warning(f"build_feature_dataframe | {symbol} | empty input")
        return pd.DataFrame()
    
    df = df.copy()
    
    # 1. Доходности (самые важные признаки!)
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    
    # 2. Скользящие средние (разные периоды создают разнообразие)
    df['sma_10'] = df['close'].rolling(10, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['ema_fast'] = df['close'].ewm(span=10, adjust=False, min_periods=1).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()
    
    # 3. RSI (14-период)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 4. ADX и DI (индикаторы тренда)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    
    plus_dm = (df['high'].diff()).clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    
    df['plus_di'] = 100 * (plus_dm.rolling(14, min_periods=1).mean() / (atr + 1e-10))
    df['minus_di'] = 100 * (minus_dm.rolling(14, min_periods=1).mean() / (atr + 1e-10))
    
    dx = 100 * ((df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10))
    df['adx'] = dx.rolling(14, min_periods=1).mean()
    
    # 5. Объём
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-10)
    
    # 6. CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20, min_periods=1).mean()
    mad = (tp - sma_tp).abs().rolling(20, min_periods=1).mean()
    df['cci'] = (tp - sma_tp) / (0.015 * (mad + 1e-10))
    
    # 7. OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # 8. Категориальный признак (разный для каждого символа!)
    df['symbol_cat'] = hash(symbol) % 17
    
    # Выбираем только нужные колонки
    feature_cols = [
        'ret_1', 'ret_5', 'sma_10', 'sma_50', 'ema_fast', 'ema_slow',
        'rsi_14', 'adx', 'plus_di', 'minus_di', 'volume_ratio',
        'cci', 'obv', 'symbol_cat'
    ]
    
    result = df[feature_cols].copy()
    
    # Заменяем inf и NaN
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result = result.ffill()  # Forward fill
    result = result.bfill()  # Backward fill
    result = result.fillna(0)  # Оставшиеся NaN -> 0
    
    logging.debug(
        f"build_feature_dataframe | {symbol} | "
        f"shape={result.shape}, ret_1_std={result['ret_1'].std():.6f}, "
        f"ret_5_std={result['ret_5'].std():.6f}"
    )
    
    return result


def fetch_and_prepare_training_data(
    adapter,
    symbols: List[str],
    base_tf: str = "15m",
    limit: int = 400
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Собирает данные для обучения модели.
    
    Returns:
        df_features: DataFrame с признаками
        df_target: Series с целевой переменной (0=hold, 1=long, 2=short)
        feature_cols: Список имён признаков
    """
    all_features = []
    all_targets = []
    
    threshold = 0.001  # 0.1% движение для классификации
    horizon = 3        # предсказываем на 3 свечи вперёд
    
    for symbol in symbols:
        try:
            # Получаем данные
            ohlcv = adapter.fetch_ohlcv(symbol, base_tf, limit=limit)
            if not ohlcv or len(ohlcv) < 100:
                logging.warning(f"fetch_and_prepare | {symbol} | insufficient data ({len(ohlcv) if ohlcv else 0} candles)")
                continue
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # КРИТИЧНО: Сохраняем close ДО обработки признаков
            close_prices = df['close'].copy()
            
            # Строим признаки
            features = build_feature_dataframe(df, symbol=symbol)
            if features.empty:
                logging.warning(f"fetch_and_prepare | {symbol} | no features")
                continue
            
            # ВАЖНО: Создаём таргет на базе ИСХОДНЫХ индексов df
            # future_return[i] = (close[i+horizon] - close[i]) / close[i]
            future_close = close_prices.shift(-horizon)
            future_return = (future_close - close_prices) / close_prices
            
            # Классификация на базе ИСХОДНОГО df
            target = pd.Series(0, index=df.index)
            target[future_return > threshold] = 1
            target[future_return < -threshold] = 2

            # После строки 147 (где future_return = ...)
            logging.info(f"DEBUG | {symbol} | future_return sample: {future_return.head()}")
            logging.info(f"DEBUG | {symbol} | future_return stats: min={future_return.min():.6f}, max={future_return.max():.6f}, mean={future_return.mean():.6f}")
            
            # КРИТИЧНО: Обрезаем последние horizon строк (нет валидного таргета)
            # И синхронизируем индексы features и target
            valid_length = len(df) - horizon
            
            # Берём только первые valid_length строк
            features = features.iloc[:valid_length].copy()
            target = target.iloc[:valid_length].copy()
            
            # Сброс индексов для синхронизации
            features = features.reset_index(drop=True)
            target = target.reset_index(drop=True)
            
            # === ДИАГНОСТИКА ===
            if len(features) > 10:
                logging.info(f"fetch_and_prepare | {symbol} | DEBUGGING:")
                
                # Проверяем первые 5 сэмплов
                for i in range(min(5, len(features))):
                    # Получаем соответствующие цены из ИСХОДНОГО df
                    current_price = close_prices.iloc[i]
                    future_price = close_prices.iloc[i + horizon] if i + horizon < len(close_prices) else None
                    actual_return = ((future_price - current_price) / current_price) if future_price else None
                    
                    logging.info(
                        f"  Sample {i}: ret_1={features.iloc[i]['ret_1']:.6f}, "
                        f"target={target.iloc[i]}, "
                        f"price[{i}]={current_price:.2f}, "
                        f"price[{i+horizon}]={future_price:.2f if future_price else 'N/A'}, "
                        f"actual_return={actual_return:.6f if actual_return else 'N/A'}"
                    )
                
                # Проверяем корреляцию
                temp_df = features.copy()
                temp_df['target'] = target.values
                corr = temp_df[['ret_1', 'ret_5', 'rsi_14']].corrwith(temp_df['target'])
                logging.info(f"  Correlations with target: {dict(corr)}")
            
            all_features.append(features)
            all_targets.append(target)
            
            logging.info(
                f"fetch_and_prepare | {symbol} | collected {len(features)} samples, "
                f"classes: {dict(target.value_counts())}"
            )
            
        except Exception as e:
            logging.error(f"fetch_and_prepare | {symbol} | error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    if not all_features:
        raise ValueError("no training data collected from any symbol")
    
    # Объединяем данные
    df_features = pd.concat(all_features, ignore_index=True)
    df_target = pd.concat(all_targets, ignore_index=True)
    
    feature_cols = df_features.columns.tolist()
    
    logging.info(
        f"fetch_and_prepare | TOTAL: {len(df_features)} samples, "
        f"features={len(feature_cols)}, classes={dict(df_target.value_counts())}"
    )
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли разнообразие в данных?
    for col in ['ret_1', 'ret_5', 'rsi_14']:
        if col in df_features.columns:
            std = df_features[col].std()
            if std < 1e-6:
                logging.error(f"fetch_and_prepare | WARNING: {col} has no variance (std={std})")
    
    return df_features, df_target, feature_cols