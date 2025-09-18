
from memory_utils import MemoryManager, normalize_param_keys


def test_normalize_param_keys():
    params = {'threshold': 0.1, 'SL_pct': 0.02}
    norm = normalize_param_keys(params)
    assert norm == {'THRESHOLD': 0.1, 'SL_PCT': 0.02}


def test_last_best_params_uppercase(tmp_path):
    path = tmp_path / 'mem.json'
    mm = MemoryManager(str(path))
    mm.add_event('optimize', {'best_params': {'threshold': 1, 'sl_pct': 2}})
    assert mm.last_best_params() == {'THRESHOLD': 1, 'SL_PCT': 2}

