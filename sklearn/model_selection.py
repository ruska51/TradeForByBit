class TimeSeriesSplit:
    def split(self, X, y=None):
        return []


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        n = len(X)
        idx = list(range(n))
        yield idx, idx


def train_test_split(*arrays, **kwargs):
    """Basic ``train_test_split`` implementation.

    The stub mimics the scikit-learn function sufficiently for the tests.  It
    splits each input array into a train and test portion and returns the
    results in the same order as the real function.
    """

    if not arrays:
        return []

    test_size = kwargs.get("test_size", 0.25)
    shuffle = kwargs.get("shuffle", True)
    random_state = kwargs.get("random_state")

    n = len(arrays[0])
    n_test = max(1, int(n * test_size))
    n_train = n - n_test

    indices = list(range(n))
    if shuffle:
        import random

        rng = random.Random(random_state)
        rng.shuffle(indices)

    test_idx = set(indices[:n_test])

    result = []
    for arr in arrays:
        train = [arr[i] for i in range(n) if i not in test_idx]
        test = [arr[i] for i in range(n) if i in test_idx]
        result.extend([train, test])
    return result
