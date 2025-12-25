#!/usr/bin/env python3
"""
Скрипт для переобучения модели с исправленным таргетом.
Использует данные из Binance (без авторизации) для обучения.
"""

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from logging_utils import setup_logging
setup_logging()

import logging
import pandas as pd
import numpy as np
import ccxt
from utils.data_prep import build_feature_dataframe
from retrain_utils import retrain_global_model
from model_utils import load_global_bundle, BUNDLE_PATH

# Список топовых монет для обучения
TRAINING_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "LTC/USDT",
    "LINK/USDT",
    "UNI/USDT",
    "ATOM/USDT",
    "ETC/USDT",
]

def main():
    logging.info("=" * 80)
    logging.info("TRAINING SCRIPT START - Fixing Model with Correct Future Targets")
    logging.info("=" * 80)

    # Создаём ccxt Binance напрямую
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        logging.info("OK Binance CCXT adapter created successfully")
    except Exception as e:
        logging.error(f"X Failed to create Binance adapter: {e}")
        return

    # Собираем данные для обучения
    logging.info(f"Fetching training data for {len(TRAINING_SYMBOLS)} symbols...")
    rows = []
    for symbol in TRAINING_SYMBOLS:
        try:
            logging.info(f"  Fetching {symbol}...")
            ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=500)

            # Конвертируем в DataFrame
            df_raw = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Строим фичи
            feat = build_feature_dataframe(df_raw, symbol, thr=0.002)
            if len(feat) > 0:
                rows.append(feat)
                logging.info(f"    OK {symbol}: {len(feat)} rows")
            else:
                logging.warning(f"    Skip {symbol}: no features generated")
        except Exception as e:
            logging.warning(f"    Failed {symbol}: {e}")

    if not rows:
        logging.error("X No training data collected from any symbol")
        return

    # Объединяем все данные
    try:
        df_all = pd.concat(rows, ignore_index=True)
        df_features = df_all.drop(columns=["target"])
        df_target = df_all["target"]
        feature_cols = list(df_features.columns)

        logging.info(f"OK Training data collected:")
        logging.info(f"  - Features shape: {df_features.shape}")
        logging.info(f"  - Target shape: {df_target.shape}")
        logging.info(f"  - Feature columns: {feature_cols}")

        # Проверяем распределение классов
        from collections import Counter
        class_dist = Counter(df_target.tolist())
        logging.info(f"  - Class distribution: {dict(class_dist)}")

        # Проверяем что данных достаточно
        if len(df_features) < 100:
            logging.error(f"X Not enough training data: {len(df_features)} rows (minimum 100 required)")
            return

        # Проверяем что все классы представлены
        unique_classes = set(df_target.unique())
        required_classes = {0, 1, 2}
        if unique_classes != required_classes:
            logging.warning(f"! Not all classes present: {unique_classes} vs {required_classes}")

    except Exception as e:
        logging.error(f"X Failed to fetch training data: {e}", exc_info=True)
        return

    # Обучаем модель
    logging.info("Starting model training...")
    try:
        model, scaler, features, classes = retrain_global_model(
            df_features=df_features,
            df_target=df_target,
            feature_cols=feature_cols,
        )

        logging.info("OK Model trained successfully!")
        logging.info(f"  - Model type: {type(model).__name__}")
        logging.info(f"  - Model classes: {classes}")
        logging.info(f"  - Features: {len(features)}")

    except Exception as e:
        logging.error(f"X Model training failed: {e}", exc_info=True)
        return

    # Проверяем что модель сохранена
    if not BUNDLE_PATH.exists():
        logging.error(f"X Model bundle NOT found at {BUNDLE_PATH}")
        return

    logging.info(f"OK Model bundle saved at {BUNDLE_PATH}")
    logging.info(f"  - File size: {BUNDLE_PATH.stat().st_size / 1024:.1f} KB")

    # Тестируем загрузку и предсказание
    logging.info("Testing model loading and prediction...")
    try:
        loaded_model, loaded_scaler, loaded_features, loaded_classes = load_global_bundle()

        # Берём несколько случайных сэмплов для теста
        test_indices = np.random.choice(len(df_features), size=min(5, len(df_features)), replace=False)
        X_test = df_features.iloc[test_indices][loaded_features].values
        X_test_scaled = loaded_scaler.transform(X_test)

        predictions = loaded_model.predict_proba(X_test_scaled)

        logging.info("OK Test predictions:")
        for i, (pred, actual) in enumerate(zip(predictions, df_target.iloc[test_indices])):
            logging.info(f"  Sample {i+1}: proba={pred}, actual_class={actual}")

        # Проверяем что вероятности НЕ равномерные
        avg_std = np.mean([np.std(p) for p in predictions])
        if avg_std < 0.05:
            logging.warning(f"! Model predictions are too uniform (std={avg_std:.4f})")
            logging.warning("  This suggests the model is not learning properly")
        else:
            logging.info(f"OK Model predictions look good (std={avg_std:.4f})")

    except Exception as e:
        logging.error(f"X Model testing failed: {e}", exc_info=True)
        return

    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE - Model ready for trading!")
    logging.info("=" * 80)
    logging.info("")
    logging.info("Next steps:")
    logging.info("1. Run your bot with: python main.py")
    logging.info("2. Check logs for non-uniform predictions")
    logging.info("3. Monitor trading decisions")

if __name__ == "__main__":
    main()
