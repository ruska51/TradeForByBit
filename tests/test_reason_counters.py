import importlib
import logging


def test_reason_counters_separated():
    import main
    importlib.reload(main)
    main._event_counters.clear()
    main._inc_event("hold_no_position")
    main._inc_event("holds")
    main._inc_event("proba_low")
    assert main._event_counters["hold_no_position"] == 1
    assert main._event_counters["holds"] == 1
    assert main._event_counters["proba_low"] == 1


def test_volume_reasons_separated(caplog):
    import main
    importlib.reload(main)
    main._event_counters.clear()
    with caplog.at_level(logging.INFO):
        main._inc_event("vol_missing")
        main.log_decision("SYM", "vol_missing")
        main._inc_event("vol_low")
        main.log_decision("SYM", "vol_low")
    assert main._event_counters["vol_missing"] == 1
    assert main._event_counters["vol_low"] == 1
    assert "vol_missing" in caplog.text
    assert "vol_low" in caplog.text
    assert "proba_low" not in caplog.text
    assert main._event_counters.get("proba_low", 0) == 0
    assert main._event_counters.get("hold_no_position", 0) == 0
