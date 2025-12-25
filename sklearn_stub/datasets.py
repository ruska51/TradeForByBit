import numpy as np


def make_classification(
    n_samples: int = 100,
    n_features: int = 20,
    n_classes: int = 2,
    n_informative: int = 2,
    n_redundant: int = 0,
    random_state: int | None = None,
):
    """Minimal stub of ``sklearn.datasets.make_classification`` for tests."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    return X, y

