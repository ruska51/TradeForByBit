import numpy as np, logging, os
from typing import Iterable

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
    """Retrain global model with XGBoost fallback."""
    import logging
    import numpy as np
    from model_utils import save_global_bundle, SimpleScaler  # ← ДОБАВЬ ЭТУ СТРОКУ!
    
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False
        logging.error("retrain_utils | XGBoost not available")
        raise RuntimeError("XGBoost required for training")
    
    # Ensure all 3 classes present
    classes_present = set(df_target.unique())
    if classes_present != {0, 1, 2}:
        logging.warning(
            "retrain_utils | classes=%s, adding synthetic samples",
            classes_present
        )
        for cls in [0, 1, 2]:
            if cls not in classes_present:
                # Add 3 synthetic samples per missing class
                for _ in range(3):
                    df_features = df_features.append(
                        df_features.iloc[0], ignore_index=True
                    )
                    df_target = df_target.append(
                        pd.Series([cls]), ignore_index=True
                    )
    
    # Scale features
    scaler = SimpleScaler().fit(df_features.values)
    X_scaled = scaler.transform(df_features.values)
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )
    
    model.fit(X_scaled, df_target.values)
    
    # Ensure classes_ attribute
    model.classes_ = np.array([0, 1, 2])
    
    # Save bundle
    try:
        save_global_bundle(model, scaler, feature_cols, model.classes_)
        logging.info("retrain_utils | saved XGB softprob model with classes=%s", model.classes_)
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
