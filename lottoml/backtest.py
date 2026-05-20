"""v2 vs 무작위 5장 hit@3 비교 백테스트."""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

from .data.types import Draw
from .model.predict import predict_probabilities
from .model.train import train_ranker
from .selection.hybrid import build_portfolio


def run_backtest(
    draws: list[Draw],
    *,
    test_draws: int = 50,
    time_limit_seconds: float = 30.0,
    pool_size: int = 30,
) -> dict:
    """마지막 test_draws 구간을 테스트 회차로 백테스트."""
    if len(draws) <= test_draws + 100:
        raise ValueError("학습 + 테스트에 충분한 회차가 없습니다")

    train_end = len(draws) - test_draws

    v2_hits: list[int] = []
    random_hits: list[int] = []
    rng = random.Random(0)

    with tempfile.TemporaryDirectory() as tmp:
        out_model = Path(tmp) / "ranker.pkl"
        out_meta = Path(tmp) / "ranker_meta.json"
        train_ranker(
            draws[:train_end],
            out_model=out_model,
            out_meta=out_meta,
            holdout=20,
        )

        for idx in range(train_end, len(draws)):
            history = draws[:idx]
            winning = set(draws[idx].numbers)

            probs = predict_probabilities(history, model_path=out_model)
            portfolio = build_portfolio(
                probs,
                seed=draws[idx].draw_no,
                pool_size=pool_size,
                time_limit_seconds=time_limit_seconds,
            )
            v2_best = max(len(set(e.numbers) & winning) for e in portfolio.tickets)
            v2_hits.append(v2_best)

            random_tickets = [
                tuple(sorted(rng.sample(range(1, 46), 6))) for _ in range(5)
            ]
            random_best = max(len(set(t) & winning) for t in random_tickets)
            random_hits.append(random_best)

    return {
        "v2_avg_best_hits": sum(v2_hits) / len(v2_hits),
        "v2_hit3": sum(1 for h in v2_hits if h >= 3) / len(v2_hits),
        "v2_hit4": sum(1 for h in v2_hits if h >= 4) / len(v2_hits),
        "random_avg_best_hits": sum(random_hits) / len(random_hits),
        "random_hit3": sum(1 for h in random_hits if h >= 3) / len(random_hits),
        "random_hit4": sum(1 for h in random_hits if h >= 4) / len(random_hits),
        "test_draws": test_draws,
    }
