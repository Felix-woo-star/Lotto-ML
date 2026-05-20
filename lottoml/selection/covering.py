"""ortools CP-SAT 기반 4티켓 covering 선택."""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from ortools.sat.python import cp_model

from .greedy import _weighted_sample_without_replacement
from .types import Ticket

TICKET_SIZE = 6


@dataclass(frozen=True)
class CoveringResult:
    tickets: list[Ticket]
    fallback_used: bool
    objective: float


def select_covering(
    pool: list[int],
    probs: dict[int, float],
    *,
    num_tickets: int,
    seed: int,
    num_candidates: int = 5000,
    time_limit_seconds: float = 30.0,
) -> CoveringResult:
    """ILP로 num_tickets 선택; 실패 시 greedy 폴백."""
    if len(pool) < TICKET_SIZE:
        raise ValueError("풀이 6개 미만입니다")

    candidate_tickets = _generate_candidate_tickets(pool, probs, num_candidates, seed)
    subset_weights = {
        sub: probs[sub[0]] * probs[sub[1]] * probs[sub[2]]
        for sub in combinations(sorted(pool), 3)
    }

    model = cp_model.CpModel()
    x = {t: model.NewBoolVar(f"x_{i}") for i, t in enumerate(candidate_tickets)}
    y = {s: model.NewBoolVar(f"y_{i}") for i, s in enumerate(subset_weights)}

    model.Add(sum(x.values()) == num_tickets)
    for sub, var in y.items():
        covering_tickets = [
            x[t] for t in candidate_tickets if set(sub).issubset(t)
        ]
        if covering_tickets:
            model.Add(var <= sum(covering_tickets))
        else:
            model.Add(var == 0)

    # log-weight to avoid float precision issues
    scale = 1_000_000
    coeffs = {s: int(round(scale * w)) for s, w in subset_weights.items()}
    model.Maximize(sum(coeffs[s] * y[s] for s in subset_weights))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tickets = _greedy_fallback(pool, probs, num_tickets=num_tickets, seed=seed)
        return CoveringResult(tickets=tickets, fallback_used=True, objective=0.0)

    chosen = [t for t in candidate_tickets if solver.Value(x[t]) == 1]
    chosen.sort()
    return CoveringResult(
        tickets=chosen,
        fallback_used=False,
        objective=float(solver.ObjectiveValue()) / scale,
    )


def _generate_candidate_tickets(
    pool: list[int],
    probs: dict[int, float],
    num_candidates: int,
    seed: int,
) -> list[Ticket]:
    rng = random.Random(seed)
    weights = [max(probs.get(n, 0.0), 1e-9) for n in pool]
    seen: set = set()
    out: list[Ticket] = []
    attempts = 0
    max_attempts = max(num_candidates * 10, 1000)
    while len(out) < num_candidates and attempts < max_attempts:
        attempts += 1
        chosen = _weighted_sample_without_replacement(rng, pool, weights, TICKET_SIZE)
        ticket: Ticket = tuple(sorted(chosen))  # type: ignore[assignment]
        if ticket not in seen:
            seen.add(ticket)
            out.append(ticket)
    return out


def _greedy_fallback(
    pool: list[int], probs: dict[int, float], *, num_tickets: int, seed: int
) -> list[Ticket]:
    from .greedy import greedy_covering
    return greedy_covering(pool, probs, num_tickets=num_tickets, seed=seed)
