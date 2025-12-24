import numpy as np, logging, os, pandas as pd  
from typing import Iterable
from collections import Counter  

# флаги доступности библиотек
SKLEARN_OK = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception as e:  # pragma: no cover - optional dependency
    SKLEARN_OK = False
    logging.info(
        "retrain_utils | sklearn unavailable: %s; will use XGBoost fallback", e
    )
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore
    XGB_OK = False
    logging.warning(
        "retrain_utils | xgboost unavailable: %s; ExtraTrees only (needs sklearn)", e
    )

from model_utils import save_global_bundle, SimpleScaler, REQUIRED_CLASSES

# Counter for consecutive feature mismatches during inference
FEATURE_MISMATCH_COUNT = 0


def _ensure_all_classes(X: np.ndarray, y: np.ndarray, min_count: int = 30):
    """
    Увеличивает количество наблюдений каждого класса до ``min_count``.
    Если класс отсутствует, функция лишь пишет предупреждение и пропускает
    апсемплинг для него, избегая ``np.random.choice`` на пустом массиве.
    """

    if X.size == 0 or y.size == 0:
        raise ValueError("training dataset is empty (no rows)")

    Xb, yb = X.copy(), y.copy()
    uniq = set(np.unique(yb).tolist())

    for c in REQUIRED_CLASSES:
        if c not in uniq:
            logging.warning(
                "retrain_utils | class %s absent in data; skipping upsample", c
            )
            continue
        idx = np.where(yb == c)[0]
        need = max(0, min_count - idx.size)
        if need > 0 and idx.size > 0:
            take = np.random.choice(idx, size=need, replace=True)
            Xb = np.vstack([Xb, Xb[take]])
            yb = np.hstack([yb, yb[take]])
    return Xb, yb


def _make_extratrees_calibrated():
    base = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    return CalibratedClassifierCV(base_estimator=base, cv=3, method="isotonic")


def _make_xgb_softprob():
    if not XGB_OK:
        raise RuntimeError("XGBoost not installed")
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )


def retrain_global_model(df_features, df_target, feature_cols):
    # === ДИАГНОСТИКА В НАЧАЛЕ ===
    logging.info("=" * 60)
    logging.info("retrain_global_model | ENTRY POINT")
    logging.info(f"retrain | df_features type: {type(df_features)}")
    logging.info(f"retrain | df_features shape: {df_features.shape if hasattr(df_features, 'shape') else 'N/A'}")
    logging.info(f"retrain | First 3 samples of features:")
    for i in range(min(3, len(df_features))):
        logging.info(f"  Row {i}: ret_1={df_features.iloc[i]['ret_1']:.6f}, target={df_target.iloc[i]}")
    logging.info("=" * 60)
    """Retrain global model with XGBoost."""
    import numpy as np
    import pandas as pd
    from collections import Counter
    from model_utils import save_global_bundle, SimpleScaler
    
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False
        logging.error("retrain_utils | XGBoost not available")
        raise RuntimeError("XGBoost required for training")
    
    # Проверяем наличие всех классов
    classes_present = set(df_target.unique())
    logging.info(f"retrain | initial classes: {classes_present}")
    
    if classes_present != {0, 1, 2}:
        logging.warning(
            "retrain_utils | classes=%s, adding synthetic samples",
            classes_present
        )
        # Добавляем по 5 синтетических семплов для каждого отсутствующего класса
        for cls in [0, 1, 2]:
            if cls not in classes_present:
                sample_indices = df_features.sample(n=min(5, len(df_features)), replace=True).index
                for idx in sample_indices:
                    new_row = df_features.loc[idx].copy()
                    df_features = pd.concat([df_features, new_row.to_frame().T], ignore_index=True)
                    df_target = pd.concat([df_target, pd.Series([cls])], ignore_index=True)
        
        logging.info(f"retrain | after augmentation: {Counter(df_target.tolist())}")
    
    logging.info(f"retrain | features shape: {df_features.shape}")
    logging.info(f"retrain | target shape: {df_target.shape}")
    logging.info(f"retrain | class distribution: {Counter(df_target.tolist())}")
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли вариативность в данных?
    for col in ['ret_1', 'ret_5', 'sma_10']:
        if col in df_features.columns:
            col_std = df_features[col].std()
            col_mean = df_features[col].mean()
            logging.info(f"retrain | {col}: mean={col_mean:.6f}, std={col_std:.6f}")
            if col_std < 1e-8:
                logging.error(f"retrain | CRITICAL: {col} has ZERO variance! Model will fail.")
    
    # Проверяем, что данные не все одинаковые
    first_row = df_features.iloc[0].values
    all_same = all(np.allclose(row, first_row, atol=1e-6) for _, row in df_features.iterrows())
    if all_same:
        logging.error("retrain | CRITICAL: All rows are IDENTICAL! This will cause uniform predictions.")
    
    # Scale features
    scaler = SimpleScaler().fit(df_features.values)
    X_scaled = scaler.transform(df_features.values)
    
    # Проверяем scaled данные
    X_scaled_std = X_scaled.std(axis=0)
    zero_variance_cols = np.where(X_scaled_std < 1e-8)[0]
    if len(zero_variance_cols) > 0:
        logging.error(f"retrain | CRITICAL: {len(zero_variance_cols)} features have zero variance after scaling!")
        logging.error(f"retrain | Zero variance columns: {[feature_cols[i] for i in zero_variance_cols]}")
    
    # Train XGBoost с более агрессивными параметрами
    model = XGBClassifier(
        n_estimators=500,           # 🔥 Больше деревьев
        max_depth=8,                # 🔥 Глубже (для 14 признаков это OK)
        learning_rate=0.05,         # 🔥 Медленнее, но стабильнее
        subsample=1.0,              # 🔥 Используем ВСЕ данные (их мало)
        colsample_bytree=1.0,       # 🔥 Используем ВСЕ признаки (их 14)
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="auto",         # 🔥 Пусть XGBoost сам выберет
        random_state=42,
        use_label_encoder=False,
        min_child_weight=1,         # 🔥 Минимальные ограничения
        gamma=0,                    # 🔥 Нет регуляризации на splits
        reg_alpha=0,                # 🔥 Нет L1
        reg_lambda=0.1,             # 🔥 Минимальная L2
        scale_pos_weight=1.0,       # 🔥 Равный вес классам
    )
    # Обучаем
    model.fit(X_scaled, df_target.values, verbose=False)

    import numpy as np
    logging.info(f"retrain | Model has {model.n_estimators} trees built")
    logging.info(f"retrain | Model feature importances: {model.feature_importances_}")
    logging.info(f"retrain | Non-zero importances: {np.count_nonzero(model.feature_importances_)}")

    # Проверяем, что модель вообще училась
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        trees_text = booster.get_dump()
        logging.info(f"retrain | First tree has {len(trees_text[0].split('leaf'))} leaves")

    # Проверяем, что модель действительно обучилась
    train_accuracy = (model.predict(X_scaled) == df_target.values).mean()
    logging.info(f"retrain | Train accuracy: {train_accuracy:.3f}")

    # Если точность ~33%, модель не обучилась
    if train_accuracy < 0.40:
        logging.error(f"retrain | CRITICAL: Train accuracy too low ({train_accuracy:.3f})!")
        logging.error("retrain | Model is guessing randomly. Check feature/target alignment.")
    
    # Проверяем на тренировочных данных
    train_proba = model.predict_proba(X_scaled[:10])
    logging.info(f"retrain | train predictions (first 10):")
    for i, proba in enumerate(train_proba[:5]):
        logging.info(f"  sample {i}: {proba}, true_class={df_target.values[i]}")
    
    # Проверяем feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        logging.info(f"retrain | Top 5 features: {top_features}")
        
        if all(imp < 0.01 for _, imp in top_features):
            logging.error("retrain | CRITICAL: All features have LOW importance! Model won't work.")
    
    # Финальный тест
    test_proba = model.predict_proba(X_scaled[:1])
    logging.info(f"retrain | test prediction: {test_proba}")
    
    if np.allclose(test_proba[0], [0.33, 0.33, 0.33], atol=0.02):
        logging.error("retrain | Model STILL outputs uniform probabilities after training!")
        logging.error("retrain | This means features have NO predictive power.")
        logging.error("retrain | Check data_prep.py - features may be constant or target may be random.")
    else:
        logging.info("retrain | ✓ Model shows non-uniform predictions - training successful!")
    
    # Ensure classes_ attribute
    model.classes_ = np.array([0, 1, 2])
    
    # Save bundle
    try:
        save_global_bundle(model, scaler, feature_cols, model.classes_)
        logging.info("retrain_utils | saved XGB model with classes=%s", model.classes_)
    except Exception as e:
        logging.error("retrain_utils | save bundle failed: %s", e)
        raise
    
    return model, scaler, feature_cols, model.classes_

def record_feature_mismatch(threshold: int = 3, *args, **kwargs) -> None:
    """Increment mismatch counter and trigger retraining after ``threshold`` hits."""
    global FEATURE_MISMATCH_COUNT
    FEATURE_MISMATCH_COUNT += 1
    if FEATURE_MISMATCH_COUNT >= threshold:
        FEATURE_MISMATCH_COUNT = 0
        try:
            retrain_global_model(*args, **kwargs)
        except Exception:  # pragma: no cover - best effort
            pass
