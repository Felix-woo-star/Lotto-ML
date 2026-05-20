import pytest

from lottoml.selection.covering import select_covering, CoveringResult


def test_select_covering_returns_four_unique_tickets() -> None:
    pool = list(range(1, 31))
    probs = {n: 1.0 / 30 for n in pool}
    result = select_covering(pool, probs, num_tickets=4, seed=42, time_limit_seconds=20)
    assert isinstance(result, CoveringResult)
    assert len(result.tickets) == 4
    assert len({frozenset(t) for t in result.tickets}) == 4
    for ticket in result.tickets:
        assert len(ticket) == 6
        assert ticket == tuple(sorted(ticket))


def test_select_covering_is_deterministic() -> None:
    pool = list(range(1, 31))
    probs = {n: 1.0 / 30 for n in pool}
    r1 = select_covering(pool, probs, num_tickets=4, seed=7, time_limit_seconds=15)
    r2 = select_covering(pool, probs, num_tickets=4, seed=7, time_limit_seconds=15)
    assert r1.tickets == r2.tickets


def test_select_covering_uses_high_probability_numbers() -> None:
    pool = list(range(1, 31))
    # 1~10번에 매우 높은 가중치
    probs = {n: 10.0 if n <= 10 else 0.01 for n in pool}
    result = select_covering(pool, probs, num_tickets=4, seed=42, time_limit_seconds=15)
    high_prob_numbers = set(range(1, 11))
    union = set()
    for ticket in result.tickets:
        union.update(ticket)
    # 4 tickets * 6 numbers = 24, 그 중 적어도 8개는 1~10에서 와야 함
    assert len(union & high_prob_numbers) >= 8
