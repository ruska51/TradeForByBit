from main import apply_pattern_proba_bonus, MIN_PROBA_FILTER


def test_pattern_proba_bonus_reduces_threshold():
    base = 0.50
    adjusted = apply_pattern_proba_bonus(base, 0.8, True)
    assert adjusted == max(MIN_PROBA_FILTER, base - 0.20)


def test_pattern_proba_bonus_no_change():
    base = 0.50
    # low confidence
    assert apply_pattern_proba_bonus(base, 0.6, True) == base
    # trend not confirmed
    assert apply_pattern_proba_bonus(base, 0.8, False) == base

