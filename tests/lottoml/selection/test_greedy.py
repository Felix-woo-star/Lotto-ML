from itertools import combinations

from lottoml.selection.greedy import greedy_covering
from lottoml.selection.types import Ticket


def _uniform_pool(size: int = 30) -> list[int]:
    return list(range(1, size + 1))


def _uniform_probs(numbers: list[int]) -> dict[int, float]:
    return {n: 1.0 / len(numbers) for n in numbers}


def test_greedy_returns_unique_six_number_tickets() -> None:
    pool = _uniform_pool(30)
    probs = _uniform_probs(pool)
    tickets = greedy_covering(pool, probs, num_tickets=4, seed=42)
    assert len(tickets) == 4
    for ticket in tickets:
        assert len(set(ticket)) == 6
        assert all(1 <= n <= 45 for n in ticket)
        assert ticket == tuple(sorted(ticket))


def test_greedy_minimizes_three_subset_overlap() -> None:
    pool = _uniform_pool(30)
    probs = _uniform_probs(pool)
    tickets = greedy_covering(pool, probs, num_tickets=4, seed=42)
    seen: set = set()
    duplicate_count = 0
    for ticket in tickets:
        for sub in combinations(ticket, 3):
            if sub in seen:
                duplicate_count += 1
            seen.add(sub)
    # 4 tickets contribute at most 4*C(6,3) = 80 subsets; greedy should keep duplicates minimal
    assert duplicate_count < 20


def test_greedy_is_deterministic() -> None:
    pool = _uniform_pool(30)
    probs = _uniform_probs(pool)
    a = greedy_covering(pool, probs, num_tickets=4, seed=99)
    b = greedy_covering(pool, probs, num_tickets=4, seed=99)
    assert a == b
