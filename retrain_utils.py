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
    """
    Обучает модель с гарантированным ``predict_proba``. Предпочитает
    калиброванный ExtraTrees, при проблемах откатывается на XGBoost.
    """

    import numpy as np
    import pandas as pd
    from retrain_utils import (
        SKLEARN_OK,
        XGB_OK,
        _make_extratrees_calibrated,
        _make_xgb_softprob,
    )

    if df_features is None or df_target is None or not feature_cols:
        raise ValueError("retrain_global_model | input features/target empty")

    X_df = df_features.copy()
    y_s = df_target.copy()

    # убрать NaN и infinities
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = (~X_df.isna().any(axis=1)) & (~y_s.isna())
    X_df = X_df.loc[mask]
    y_s = y_s.loc[mask]
    if X_df.empty or y_s.empty:
        raise ValueError("retrain_global_model | dataset empty after cleaning")

    X = X_df[feature_cols].astype("float32").values
    y = y_s.astype("int32").values

    if X.shape[0] == 0:
        raise ValueError("retrain_global_model | no training rows available")

    # апсемплинг классов
    X, y = _ensure_all_classes(X, y, min_count=30)

    if SKLEARN_OK:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        model = _make_extratrees_calibrated()
        model.fit(Xtr_s, ytr)
        classes = getattr(model, "classes_", np.array([], dtype=int))
        if (
            not hasattr(model, "predict_proba")
            or len(classes) != len(REQUIRED_CLASSES)
            or set(classes.tolist()) != set(REQUIRED_CLASSES.tolist())
        ):
            logging.warning(
                "retrain_utils | ExtraTrees invalid (proba=%s, classes=%s); fallback to XGB",
                hasattr(model, "predict_proba"),
                classes,
            )
            if not XGB_OK:
                raise RuntimeError("ExtraTrees invalid and XGBoost not installed")
            model = _make_xgb_softprob()
            model.fit(Xtr_s, ytr)
            classes = np.array([0, 1, 2])
        try:
            model.classes_ = classes
            save_global_bundle(model, scaler, feature_cols, classes)
        except Exception as e:
            logging.error("retrain_utils | save bundle failed: %s", e)
            raise
        logging.info(
            "retrain_utils | saved calibrated model with classes=%s",
            classes.tolist(),
        )
        return model, scaler, feature_cols, classes

    if XGB_OK:
        scaler = SimpleScaler().fit(X)
        model = _make_xgb_softprob()
        model.fit(scaler.transform(X), y)
        classes = np.array([0, 1, 2])
        try:
            model.classes_ = classes
            save_global_bundle(model, scaler, feature_cols, classes)
        except Exception as e:
            logging.error("retrain_utils | save bundle failed: %s", e)
            raise
        logging.info(
            "retrain_utils | saved XGB softprob model with classes=%s",
            classes.tolist(),
        )
        return model, scaler, feature_cols, classes

    raise RuntimeError("No ML libraries available. Install sklearn or xgboost.")


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
