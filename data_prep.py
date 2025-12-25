import pandas as pd
import numpy as np
import logging
import pickle
import os
from pathlib import Path
from typing import Tuple, List

# Список признаков (14 штук)
GLOBAL_FEATURE_LIST = [
    'ret_1', 'ret_5', 'sma_10', 'sma_50', 'ema_fast', 'ema_slow', 
    'rsi_14', 'adx', 'plus_di', 'minus_di', 'volume_ratio', 'cci', 'obv', 'symbol_cat'
]

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = str(BASE_DIR / "market_data_cache.pkl")

def build_feature_dataframe(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """Генерирует чистые признаки."""
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 50:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Индикаторы
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift()).abs()
    df['tr3'] = (df['low'] - df['close'].shift()).abs()
    tr = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = tr.rolling(14).mean()
    
    df['plus_di'] = (df['high'].diff().clip(lower=0).rolling(14).mean() / (atr + 1e-10)) * 100
    df['minus_di'] = (df['low'].diff().clip(upper=0).abs().rolling(14).mean() / (atr + 1e-10)) * 100
    df['adx'] = ((df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10) * 100).rolling(14).mean()
    
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    df['symbol_cat'] = hash(symbol) % 17
    
    df_clean = df.dropna(subset=GLOBAL_FEATURE_LIST).copy()
    result = df_clean[GLOBAL_FEATURE_LIST].copy()
    result.replace([np.inf, -np.inf], 0, inplace=True)
    return result.fillna(0)

def save_current_cache(symbol_data):
    try:
        new_data = {}
        # Пробуем достать кэш из адаптера
        if hasattr(symbol_data, 'candles_cache'):
            new_data = symbol_data.candles_cache
        elif isinstance(symbol_data, dict):
            new_data = symbol_data
        
        if not new_data:
            return

        disk_data = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                disk_data = pickle.load(f)
        
        for symbol, df_new in new_data.items():
            if symbol in disk_data:
                disk_data[symbol] = pd.concat([disk_data[symbol], df_new]).drop_duplicates().sort_index()
            else:
                disk_data[symbol] = df_new
        
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(disk_data, f)
        
        logging.info(f"💾 КЭШ СОХРАНЕН: {len(disk_data)} монет в файле {CACHE_FILE}")
    except Exception as e:
        logging.error(f"Ошибка сохранения: {e}")

def prepare_training_data(symbol_data, horizon: int = 5, threshold: float = 0.002, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Подготавливает данные, используя склеенный кэш (диск + RAM)."""
    
    # Сначала вызываем сохранение, чтобы учесть самые свежие данные из текущего цикла
    save_current_cache(symbol_data)
    
    # Теперь читаем уже обновленный полный кэш с диска
    if not os.path.exists(CACHE_FILE):
        raise ValueError("Кэш данных пуст. Подождите завершения хотя бы одного цикла сбора данных.")

    with open(CACHE_FILE, "rb") as f:
        final_data_dict = pickle.load(f)

    all_features, all_targets = [], []
    logging.info(f"Подготовка обучения на истории {len(final_data_dict)} символов...")

    for symbol, df in final_data_dict.items():
        try:
            if df is None or len(df) < 100:
                continue
            
            features = build_feature_dataframe(df, symbol=symbol)
            if features.empty:
                continue

            future_return = (df['close'].shift(-horizon) - df['close']) / (df['close'] + 1e-10)
            valid_idx = features.index.intersection(future_return.dropna().index)
            
            X = features.loc[valid_idx]
            y_ret = future_return.loc[valid_idx]

            y = pd.Series(1, index=valid_idx) # Hold
            y[y_ret > threshold] = 2          # Buy
            y[y_ret < -threshold] = 0         # Sell
            
            all_features.append(X)
            all_targets.append(y)
        except Exception as e:
            logging.warning(f"Ошибка обработки {symbol}: {e}")
            continue
    
    if not all_features:
        raise ValueError("Недостаточно данных во всех источниках для обучения.")
    
    X_final = pd.concat(all_features)
    y_final = pd.concat(all_targets)
    
    logging.info(f"🚀 ОБУЧЕНИЕ: {X_final.shape} строк | Классы: {y_final.value_counts().to_dict()}")
    return X_final, y_final

fetch_and_prepare_training_data = prepare_training_data