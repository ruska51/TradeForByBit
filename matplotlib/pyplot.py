"""Very small ``matplotlib.pyplot`` stub used for tests.

The real project optionally depends on ``matplotlib`` for chart generation.
During testing the heavy dependency is replaced with this lightweight stub
which mimics just enough of the API for the rest of the code to run without
emitting warnings.  Previously the stub only provided ``plot``/``savefig`` and
``close`` functions which meant calls to :func:`subplots` raised an
``AttributeError``.  The surrounding code interpreted that failure as missing
figure support and logged a warning.  To keep the log clean we implement a
minimal ``subplots`` helper returning dummy objects that provide the methods
used by ``save_candle_chart``.
"""


class _DummyAx:
    def plot(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass


class _DummyFig:
    def tight_layout(self, *args, **kwargs):
        pass

    def savefig(self, *args, **kwargs):
        pass


def subplots(*args, **kwargs):  # pragma: no cover - trivial
    """Return dummy figure and axis objects."""
    return _DummyFig(), _DummyAx()


def plot(*args, **kwargs):  # pragma: no cover - trivial
    pass


def savefig(*args, **kwargs):  # pragma: no cover - trivial
    pass


def close(*args, **kwargs):  # pragma: no cover - trivial
    pass
