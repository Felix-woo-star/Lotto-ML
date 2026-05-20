from lottoml.reporting.scoring import classify, score_ticket

WIN = (5, 11, 17, 26, 32, 39)
BONUS = 20


def test_six_match_is_first() -> None:
    hits, tier = classify((5, 11, 17, 26, 32, 39), WIN, BONUS)
    assert hits == 6
    assert tier == 1


def test_five_plus_bonus_is_second() -> None:
    hits, tier = classify((5, 11, 17, 26, 32, 20), WIN, BONUS)
    assert hits == 5
    assert tier == 2


def test_five_match_is_third() -> None:
    hits, tier = classify((5, 11, 17, 26, 32, 45), WIN, BONUS)
    assert hits == 5
    assert tier == 3


def test_four_match_is_fourth() -> None:
    hits, tier = classify((5, 11, 17, 26, 1, 2), WIN, BONUS)
    assert hits == 4
    assert tier == 4


def test_three_match_is_fifth() -> None:
    hits, tier = classify((5, 11, 17, 1, 2, 3), WIN, BONUS)
    assert hits == 3
    assert tier == 5


def test_two_match_no_tier() -> None:
    hits, tier = classify((5, 11, 1, 2, 3, 4), WIN, BONUS)
    assert hits == 2
    assert tier is None


def test_score_ticket_uses_actual_payouts_for_high_tiers() -> None:
    actuals = {
        "first_prize": 1_000_000_000, "second_prize": 50_000_000, "third_prize": 1_500_000,
    }
    payout = score_ticket(tier=1, actuals=actuals)
    assert payout == 1_000_000_000
    payout = score_ticket(tier=2, actuals=actuals)
    assert payout == 50_000_000


def test_score_ticket_uses_fixed_payouts_for_lower_tiers() -> None:
    payout = score_ticket(tier=4, actuals={})
    assert payout == 50_000
    payout = score_ticket(tier=5, actuals={})
    assert payout == 5_000


def test_score_ticket_returns_zero_for_no_tier() -> None:
    assert score_ticket(tier=None, actuals={}) == 0
