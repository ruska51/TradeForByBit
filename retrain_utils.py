import numpy as np, logging, os, pandas as pd  
from typing import Iterable
from collections import Counter  


# === УЛУЧШЕННЫЙ БЛОК ИМПОРТОВ ===
SKLEARN_OK = False
XGB_OK = False

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Пытаемся импортировать калибровку отдельно
    try:
        from sklearn.calibration import CalibratedClassifierCV
        logging.info("[OK] [retrain_utils] CalibratedClassifierCV loaded")
    except ImportError:
        logging.warning("[WARN] [retrain_utils] Calibration missing, using raw ensemble")
        CalibratedClassifierCV = None

    SKLEARN_OK = True
    try:
        import sklearn.base
        logging.info("[OK] [retrain_utils] Sklearn base detected")
    except:
        logging.info("[OK] [retrain_utils] Sklearn detected")
except ImportError as e:
    logging.warning(f"[ERROR] [retrain_utils] Core Sklearn import failed: {e}")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
    logging.info("[OK] [retrain_utils] XGBoost loaded")
except ImportError:
    logging.warning("[ERROR] [retrain_utils] XGBoost unavailable")
# ===============================
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


def retrain_global_model(df_features: pd.DataFrame, df_target: pd.Series, feature_cols: list):
    logging.info("="*60)
    logging.info("retrain_global_model | START")
    
    X = df_features[feature_cols].values
    y = df_target.values
    
    # 1. Апсемплинг (твоя страховка)
    X, y = _ensure_all_classes(X, y, min_count=100)
    
    # 2. Масштабирование (раз StandardScaler бесил, используем SimpleScaler)
    scaler = SimpleScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # 3. Работа с весами (теперь безопасно)
    sample_weights = None
    try:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        logging.info("[OK] [retrain] Sample weights computed successfully")
    except Exception as e:
        logging.warning(f"[WARN] [retrain] Could not compute weights ({e}), training without them")

    # 4. Обучение XGBoost
    logging.info(f"retrain | Training on {X_scaled.shape} samples...")

    # Вычисляем веса классов
    from collections import Counter
    c = Counter(y)
    total = len(y)
    weights_map = {cls: total / (count * len(c)) for cls, count in c.items()}
    sample_weights = np.array([weights_map[label] for label in y])
    logging.info(f"retrain | Class weights: {weights_map}")

    # Создаём и обучаем модель
    model = XGBClassifier(
        n_estimators=100,  # Уменьшили для скорости
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1,  # Используем 1 поток для отладки
        verbosity=1,  # Включаем логи XGBoost
    )

    logging.info(f"retrain | Starting XGBoost fit...")
    try:
        model.fit(
            X_scaled,
            y,
            sample_weight=sample_weights,
            verbose=True,  # Логируем процесс обучения
        )
        logging.info(f"retrain | XGBoost fit completed!")
    except Exception as e:
        logging.error(f"retrain | XGBoost fit FAILED: {e}", exc_info=True)
        raise

    # Проверка что модель обучена
    if not hasattr(model, '_Booster') and not hasattr(model, 'get_booster'):
        logging.error("retrain | Model has no booster - training failed!")
        raise RuntimeError("XGBoost model not trained properly")

    # Проверка вероятностей в середине данных
    idx = len(X_scaled) // 2
    sample_preds = model.predict_proba(X_scaled[idx:idx+3])
    logging.info(f"retrain | Probabilities sample (mid-training-data):\n{sample_preds}")

    # Проверка на первых 3 сэмплах
    first_preds = model.predict_proba(X_scaled[:3])
    logging.info(f"retrain | Probabilities sample (first 3):\n{first_preds}")

    # Проверяем что вероятности НЕ равномерные
    avg_std = np.mean([np.std(p) for p in sample_preds])
    if avg_std < 0.01:
        logging.warning(f"retrain | WARNING: Model predictions too uniform (std={avg_std:.4f})")
        logging.warning(f"retrain | This indicates training failed or data has no signal")
    else:
        logging.info(f"retrain | Model predictions variance OK (std={avg_std:.4f})")

    # НЕ перезаписываем classes_, используем то что XGBoost установил
    if not hasattr(model, 'classes_'):
        logging.warning("retrain | Model has no classes_, setting manually")
        model.classes_ = np.array([0, 1, 2])
    save_global_bundle(model, scaler, feature_cols, model.classes_)
    
    logging.info("retrain | SUCCESS! Model saved.")
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
