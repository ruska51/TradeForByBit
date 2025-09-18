import pandas as pd
import optuna
from main import run_optuna, save_optuna_study, load_optuna_study


def test_optuna_study_save_and_load(tmp_path):
    df = pd.DataFrame({'profit': [1, -1, 2]})
    study = run_optuna(df, n_trials=1)
    file = tmp_path / 'study.pkl'
    save_optuna_study(study, str(file))
    assert file.exists()
    loaded = load_optuna_study(str(file))
    assert isinstance(loaded, optuna.study.Study)
    assert loaded.best_params == study.best_params
