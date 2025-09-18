"""Very small stub of :mod:`xgboost` used for testing.

The real project depends on ``xgboost`` which is heavy.  For the unit tests we
provide a tiny stand‑in that mimics just enough of the real API for the code to
run.  The fallback logic for loading a global model relies on ``load_model`` and
``save_model`` methods, so lightweight implementations are provided here as
well.  Models are serialised via :mod:`joblib` with a simple dictionary of
parameters.
"""

from __future__ import annotations

import joblib


class XGBClassifier:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.best_iteration = kwargs.get("n_estimators", 100)

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):  # pragma: no cover - trivial stub
        import numpy as np

        n = len(X)
        k = self.params.get("num_class", 2)
        return np.full((n, k), 1.0 / k)

    def get_params(self):
        return self.params

    def get_xgb_params(self):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

    # ------------------------------------------------------------------
    # persistence helpers used by the new fallback model loader
    def save_model(self, path: str) -> None:  # pragma: no cover - trivial
        """Persist parameters to ``path`` using :mod:`joblib`."""

        joblib.dump({"params": self.params}, path)

    def load_model(self, path: str) -> None:  # pragma: no cover - trivial
        """Load parameters from ``path`` if it exists."""

        try:
            data = joblib.load(path)
            if isinstance(data, dict) and "params" in data:
                self.params.update(data["params"])
        except FileNotFoundError:
            # If the file does not exist we simply keep default parameters.
            pass

