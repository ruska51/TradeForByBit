import retrain_utils


def test_record_feature_mismatch_triggers(monkeypatch):
    calls = []
    monkeypatch.setattr(retrain_utils, "retrain_global_model", lambda: calls.append(True))
    import types, sys
    dummy_main = types.SimpleNamespace(
        GLOBAL_MODEL_PATH="/tmp/nonexistent.pkl",
        GLOBAL_MODEL=None,
        GLOBAL_SCALER=None,
        GLOBAL_FEATURES=None,
    )
    monkeypatch.setitem(sys.modules, "main", dummy_main)
    monkeypatch.setattr(retrain_utils.os.path, "exists", lambda p: False)
    retrain_utils.FEATURE_MISMATCH_COUNT = 2
    retrain_utils.record_feature_mismatch()
    assert calls == [True]
    assert retrain_utils.FEATURE_MISMATCH_COUNT == 0
