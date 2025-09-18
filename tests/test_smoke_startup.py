def test_smoke_startup():
    import main

    assert hasattr(main, "run_bot_loop")
