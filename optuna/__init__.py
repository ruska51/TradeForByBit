import pandas as pd


class trial:
    class Trial:
        def suggest_float(self, name, low, high):
            return (low + high) / 2

        def suggest_int(self, name, low, high):
            return int((low + high) / 2)


class study:
    class Study:
        def __init__(self):
            self.best_value = float("-inf")
            self.best_params = {}

        def enqueue_trial(self, params):
            pass

        def optimize(self, func, n_trials=1):
            for _ in range(n_trials):
                t = trial.Trial()
                val = func(t)
                if val > self.best_value:
                    self.best_value = val
                    self.best_params = {}

        def trials_dataframe(self):
            return pd.DataFrame()


class logging:
    WARNING = 0
    INFO = 1

    @staticmethod
    def set_verbosity(level):
        pass


def create_study(direction="maximize"):
    return study.Study()
