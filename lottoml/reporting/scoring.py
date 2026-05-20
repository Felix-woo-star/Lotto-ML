"""티켓 적중 등수 분류 + 환급금 계산."""
from __future__ import annotations

DEFAULT_PAYOUTS = {
    1: 2_000_000_000,
    2: 50_000_000,
    3: 1_500_000,
    4: 50_000,
    5: 5_000,
}


def classify(
    ticket: tuple[int, ...], winning: tuple[int, ...], bonus: int
) -> tuple[int, int | None]:
    """티켓의 적중 개수와 등수를 반환."""
    hits = len(set(ticket) & set(winning))
    bonus_hit = bonus in ticket
    if hits == 6:
        return hits, 1
    if hits == 5 and bonus_hit:
        return hits, 2
    if hits == 5:
        return hits, 3
    if hits == 4:
        return hits, 4
    if hits == 3:
        return hits, 5
    return hits, None


def score_ticket(*, tier: int | None, actuals: dict) -> int:
    """등수와 회차별 실 당첨금(있으면)을 받아 환급금 반환."""
    if tier is None:
        return 0
    if tier == 1 and actuals.get("first_prize"):
        return int(actuals["first_prize"])
    if tier == 2 and actuals.get("second_prize"):
        return int(actuals["second_prize"])
    if tier == 3 and actuals.get("third_prize"):
        return int(actuals["third_prize"])
    return DEFAULT_PAYOUTS[tier]
