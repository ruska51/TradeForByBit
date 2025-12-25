class SelectFromModel:
    """Very small stub of scikit-learn's :class:`SelectFromModel`.

    It keeps API-compatibility for the parts of the project that rely on it
    without pulling in the full dependency.  Feature selection is not actually
    performed; all features are retained.
    """

    def __init__(self, estimator=None, *args, **kwargs):
        self.estimator = estimator
        self.support_ = None

    def fit(self, X, y=None):
        if self.estimator is not None and hasattr(self.estimator, "fit"):
            try:
                self.estimator.fit(X, y)
            except Exception:
                pass
        if hasattr(X, "shape"):
            n_features = X.shape[1]
        elif getattr(X, "__len__", None) and len(X):
            first = X[0]
            n_features = len(first) if getattr(first, "__len__", None) else 1
        else:
            n_features = 0
        self.support_ = [True] * n_features
        return self

    def transform(self, X):
        return X

    def get_support(self):
        return self.support_
