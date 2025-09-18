def test_model_fallback(monkeypatch):
    import model_utils

    monkeypatch.setattr(model_utils, "_check_sklearn_install", lambda: (False, "forced"))
    model, scaler, features = model_utils.load_global_model()
    assert model is not None and hasattr(model, "predict")
