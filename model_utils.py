from __future__ import annotations

"""Helpers for loading and creating models used by the trading bot."""

# [ANCHOR:MODEL_IMPORTS]
import os
import logging
import importlib
import pkgutil
from pathlib import Path

import joblib
from joblib import dump, load
import numpy as np
import pandas as pd

from exchange_adapter import ExchangeAdapter

# ---------------------------------------------------------------------------
# Unified model bundle helpers
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = MODEL_DIR / "global_model.joblib"


class SimpleScaler:
    """Минимальный скейлер для XGB-fallback (pickle-friendly)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        import numpy as np

        X = np.asarray(X)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-9
        return self

    def transform(self, X):
        import numpy as np

        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        import numpy as np

        X = np.asarray(X)
        return X * self.std_ + self.mean_

REQUIRED_CLASSES = np.array([0, 1, 2])  # 0=hold,1=long,2=short


def save_global_bundle(model, scaler, features, classes):
    """Persist model bundle with features and classes."""

    dump(
        {
            "model": model,
            "scaler": scaler,
            "features": list(features),
            "classes": np.array(classes),
        },
        BUNDLE_PATH,
    )
    logging.info("model_utils | saved bundle at %s", BUNDLE_PATH)


def load_global_bundle():
    """Load model bundle ensuring required classes and predict_proba support."""

    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"global bundle not found at {BUNDLE_PATH}")
    bundle = load(BUNDLE_PATH)
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    features = bundle.get("features") or []
    classes_obj = bundle.get("classes")
    classes = np.array(classes_obj if classes_obj is not None else [])
    if model is None or scaler is None or len(features) == 0:
        raise ValueError("incomplete bundle (model/scaler/features)")
    if not hasattr(model, "predict_proba") and getattr(model, "objective", "") != "multi:softprob":
        raise ValueError("bundle model lacks predict_proba")
    if classes.size < 3 or set(classes.tolist()) != set(REQUIRED_CLASSES.tolist()):
        raise ValueError(f"bundle classes invalid: {classes}")
    # Ensure model exposes the expected class labels for health checks
    try:
        model.classes_ = classes
    except Exception:
        pass
    return model, scaler, features, classes


ADAPTER: ExchangeAdapter | None = None
# Cache for the loaded global model.  The tuple stores ``(model, scaler, features)``
# to avoid confusion with similarly named globals in :mod:`main`.
GLOBAL_MODEL_CACHE: tuple | None = None


def _get_adapter() -> ExchangeAdapter:
    """Lazy-load the global :class:`ExchangeAdapter` from :mod:`main`."""
    global ADAPTER
    if ADAPTER is None:  # pragma: no cover - import side effect
        from main import ADAPTER as MAIN_ADAPTER  # type: ignore
        ADAPTER = MAIN_ADAPTER
    return ADAPTER


def _check_sklearn_install() -> tuple[bool, str]:
    """Проверяет, что установлена именно scikit-learn с подмодулем ensemble."""
    try:
        sk = importlib.import_module("sklearn")
        if pkgutil.find_loader("sklearn.ensemble") is None:
            return False, "sklearn package without 'ensemble' (likely installed 'sklearn' instead of 'scikit-learn')"
        return True, "ok"
    except Exception as e:
        return False, f"sklearn import failed: {e}"


def build_recent_dataset_light(
    symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 200
) -> pd.DataFrame:
    """Return a dataframe with basic features for recent bars."""

    try:
        data = _get_adapter().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:  # pragma: no cover - network errors
        logging.warning("model | %s | ohlcv fetch failed: %s", symbol, exc)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["ret"] = df["close"].pct_change()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    plus_dm = (df["high"].diff()).clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    tr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df[["ret", "atr", "rsi", "adx"]].dropna().reset_index(drop=True)


def build_recent_dataset(
    symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 200
):
    """Return ``(X, y, scaler, features)`` for quick model training."""

    try:
        df = build_recent_dataset_light(symbol, timeframe, limit)
        if df.empty or len(df) <= 1:
            raise ValueError("not enough data")
        features = list(df.columns)
        X = df.iloc[:-1].values
        y = (df["ret"].shift(-1).iloc[:-1] > 0).astype(int).values
        try:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        except Exception:  # pragma: no cover - optional dependency
            scaler = None
        return X, y, scaler, features
    except Exception as e:
        logging.warning("model | build_recent_dataset failed: %s; using constant hold model", e)
        X = np.zeros((1, 4), dtype=float)
        y = np.zeros((1,), dtype=int)
        try:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(X)
        except Exception:  # pragma: no cover - optional dependency
            scaler = None
        features = ["ret", "atr", "rsi", "adx"]
        return X, y, scaler, features


def load_global_model() -> tuple:
    """Load the persisted global model.

    The model is always stored as ``{"model": clf, "scaler": scaler,
    "features": features}``.  If the model file is missing or cannot be read a
    retrain is triggered.  As a last resort an XGBoost model is created so that
    callers always receive a usable model object.
    """

    import joblib

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "global_model.joblib")
    xgb_path = os.path.join(model_dir, "global_xgb.json")
    scaler_path = os.path.join(model_dir, "global_scaler.pkl")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        try:
            from retrain_utils import retrain_global_model

            retrain_global_model()
        except Exception as retrain_exc:
            logging.error(
                "model | retrain failed: %s; falling back to XGBoost", retrain_exc
            )
            return _load_xgb_fallback(model_path, xgb_path, scaler_path)

    try:
        meta = joblib.load(model_path)
    except Exception as exc:
        logging.warning("model | load failed: %s; retraining", exc)
        try:
            from retrain_utils import retrain_global_model

            retrain_global_model()
            meta = joblib.load(model_path)
        except Exception as retrain_exc:
            logging.error(
                "model | retrain failed: %s; falling back to XGBoost", retrain_exc
            )
            return _load_xgb_fallback(model_path, xgb_path, scaler_path)

    if not isinstance(meta, dict):
        logging.error("model | unsupported model format; falling back to XGBoost")
        return _load_xgb_fallback(model_path, xgb_path, scaler_path)

    model = meta.get("model")
    scaler = meta.get("scaler")
    features = meta.get("features", [])
    if model is None:
        logging.error("model | global load returned None; using fallback")
        return _load_xgb_fallback(model_path, xgb_path, scaler_path)

    classes = getattr(model, "classes_", [])
    has_proba = hasattr(model, "predict_proba") or (
        hasattr(model, "predict") and getattr(model, "objective", "") == "multi:softprob"
    )
    need_retrain = False
    if len(classes) < 3:
        logging.warning("model | classes=%s <3", classes)
        need_retrain = True
    if not has_proba:
        logging.warning("model | model lacks predict_proba")
        need_retrain = True
    if need_retrain:
        try:
            from retrain_utils import retrain_global_model

            retrain_global_model()
            meta = joblib.load(model_path)
            model = meta.get("model")
            scaler = meta.get("scaler")
            features = meta.get("features", [])
        except Exception as exc:
            logging.error("model | retrain failed: %s", exc)
            return _load_xgb_fallback(model_path, xgb_path, scaler_path)

    return model, scaler, features


def _load_xgb_fallback(model_path: str, xgb_path: str, scaler_path: str) -> tuple:
    """Fallback model loader using :class:`xgboost.XGBClassifier`."""

    from xgboost import XGBClassifier
    import joblib

    if os.path.exists(model_path):
        try:
            meta = joblib.load(model_path)
            return meta["model"], meta.get("scaler"), meta.get("features", [])
        except Exception:
            pass

    if os.path.exists(scaler_path):
        scaler, features = joblib.load(scaler_path)
    else:
        X, y, scaler, features = build_recent_dataset()
        joblib.dump((scaler, features), scaler_path)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=2,
        tree_method="hist",
    )

    if os.path.exists(xgb_path):
        model.load_model(xgb_path)
    else:
        X, y, scaler, features = build_recent_dataset()
        model.fit(X, y)
        model.save_model(xgb_path)

    joblib.dump({"model": model, "scaler": scaler, "features": features}, model_path)
    return model, scaler, features


def predict_with_fallback(symbol: str, timeframe: str = "5m", limit: int = 200) -> str:
    """Lightweight prediction using recent data.

    If scikit-learn is available a small :class:`LogisticRegression` model is
    trained on recent bars.  Otherwise a rule-based decision based on RSI/ADX is
    returned.
    """

    df = build_recent_dataset_light(symbol, timeframe, limit)
    if df.empty:
        logging.warning("model | %s | no data for fallback", symbol)
        return "hold"

    global GLOBAL_MODEL_CACHE
    if GLOBAL_MODEL_CACHE is None:  # pragma: no cover - cache model
        try:
            GLOBAL_MODEL_CACHE = load_global_model()
        except Exception as exc:
            logging.warning("model | load_global_model failed: %s", exc)
            GLOBAL_MODEL_CACHE = None

    if GLOBAL_MODEL_CACHE:
        model, scaler, features = GLOBAL_MODEL_CACHE
        try:
            X = df[features].iloc[-1:].values
            if scaler is not None:
                X = scaler.transform(X)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0][1]
                if prob > 0.51:
                    return "long"
                if prob < 0.49:
                    return "short"
                return "hold"
            pred = model.predict(X)[0]
            return "long" if int(pred) > 0 else "short"
        except Exception as exc:
            logging.warning("model | global model prediction failed: %s", exc)

    ok, _ = _check_sklearn_install()
    if ok and len(df) > 20:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            X = df[["ret", "atr", "rsi", "adx"]].iloc[:-1].values
            y = (df["ret"].shift(-1).iloc[:-1] > 0).astype(int).values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            last = scaler.transform(df[["ret", "atr", "rsi", "adx"]].iloc[-1:].values)
            prob = model.predict_proba(last)[0][1]
            if prob > 0.51:
                return "long"
            if prob < 0.49:
                return "short"
            return "hold"
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("model | sklearn fallback failed: %s", exc)

    rsi = df["rsi"].iloc[-1]
    adx = df["adx"].iloc[-1]
    if adx > 12:
        if rsi < 30:
            return "long"
        if rsi > 70:
            return "short"
    return "hold"

def make_xgb_classifier(n_classes: int, random_state: int = 42):
    """Return an :class:`xgboost.XGBClassifier` for binary or multi-class tasks."""

    from xgboost import XGBClassifier

    if n_classes <= 2:
        return XGBClassifier(random_state=random_state, eval_metric="logloss")
    return XGBClassifier(
        random_state=random_state,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
    )

