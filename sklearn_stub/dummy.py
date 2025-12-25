class DummyClassifier:
    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy

    def fit(self, X, y):
        self.constant_ = 1 if y and sum(y) else 0
        return self

    def predict(self, X):
        return [getattr(self, "constant_", 0)] * len(X)

    def predict_proba(self, X):
        p = [0.0, 0.0, 0.0]
        idx = getattr(self, "constant_", 0)
        if 0 <= idx < len(p):
            p[idx] = 1.0
        return [p[:] for _ in range(len(X))]
