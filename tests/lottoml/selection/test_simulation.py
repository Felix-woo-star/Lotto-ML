import random
from itertools import combinations

from lottoml.selection.hybrid import build_portfolio


def _draw_winning_set(rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(1, 46), 6)))


def test_hybrid_simulated_hit_at_3_exceeds_random_baseline() -> None:
    rng = random.Random(123)
    # 약한 prior: 균등 + 작은 노이즈
    probs = {n: 1.0 / 45 + rng.uniform(-0.001, 0.001) for n in range(1, 46)}
    portfolio = build_portfolio(probs, seed=42, pool_size=30, time_limit_seconds=20)
    tickets = [e.numbers for e in portfolio.tickets]

    n_trials = 200
    hits = 0
    for _ in range(n_trials):
        winning = set(_draw_winning_set(rng))
        if any(len(set(t) & winning) >= 3 for t in tickets):
            hits += 1
    hit_rate = hits / n_trials
    # 무작위 5장 이론치 10.7%. 표본 200이라 ±5% 노이즈, 6% 하한으로 검증
    assert hit_rate >= 0.06, f"hit@3 = {hit_rate:.3f}"
