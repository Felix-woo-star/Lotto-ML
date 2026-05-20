"""Greedy 4티켓 covering 선택 (ILP 폴백 + baseline)."""
from __future__ import annotations

import random
from itertools import combinations

from .types import Ticket

TICKET_SIZE = 6


def greedy_covering(
    pool: list[int],
    probs: dict[int, float],
    *,
    num_tickets: int,
    seed: int,
    num_candidates: int = 5000,
) -> list[Ticket]:
    """가중 샘플링으로 후보 티켓 생성 → greedy 선택 (3-subset 중복 최소화)."""
    if len(pool) < TICKET_SIZE:
        raise ValueError("풀이 6개 미만입니다")

    rng = random.Random(seed)
    weights = [max(probs.get(n, 0.0), 1e-9) for n in pool]

    candidates: dict[Ticket, float] = {}
    while len(candidates) < num_candidates:
        chosen = _weighted_sample_without_replacement(rng, pool, weights, TICKET_SIZE)
        ticket: Ticket = tuple(sorted(chosen))  # type: ignore[assignment]
        if ticket not in candidates:
            score = sum(probs.get(n, 0.0) for n in ticket)
            candidates[ticket] = score
        if len(candidates) >= num_candidates:
            break

    sorted_candidates = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))

    selected: list[Ticket] = []
    selected_set: set[Ticket] = set()
    covered_subsets: set = set()
    remaining = list(sorted_candidates)
    while remaining and len(selected) < num_tickets:
        best_idx = 0
        best_gain = -1
        best_score = float("-inf")
        for i, (ticket, score) in enumerate(remaining):
            new_subsets = sum(1 for s in combinations(ticket, 3) if s not in covered_subsets)
            if new_subsets > best_gain or (new_subsets == best_gain and score > best_score):
                best_gain = new_subsets
                best_score = score
                best_idx = i
        ticket, _ = remaining.pop(best_idx)
        if ticket in selected_set:
            continue
        selected.append(ticket)
        selected_set.add(ticket)
        covered_subsets.update(combinations(ticket, 3))

    return selected[:num_tickets]


def _weighted_sample_without_replacement(
    rng: random.Random, numbers: list[int], weights: list[float], k: int
) -> list[int]:
    pool = list(numbers)
    pool_weights = list(weights)
    picks: list[int] = []
    for _ in range(k):
        total = sum(pool_weights)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            target = rng.random() * total
            cum = 0.0
            idx = len(pool) - 1
            for i, w in enumerate(pool_weights):
                cum += w
                if cum >= target:
                    idx = i
                    break
        picks.append(pool.pop(idx))
        pool_weights.pop(idx)
    return picks
