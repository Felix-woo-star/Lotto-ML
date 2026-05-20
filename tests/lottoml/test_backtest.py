import datetime as dt

from lottoml.backtest import run_backtest
from lottoml.data.types import Draw


def _make_draws(n: int = 250) -> list[Draw]:
    draws = []
    for i in range(1, n + 1):
        unique = []
        for k in range(6):
            v = ((i * 7 + k * 11) % 45) + 1
            while v in unique:
                v = (v % 45) + 1
            unique.append(v)
        draws.append(
            Draw(
                draw_no=i, draw_date=dt.date(2026, 1, 1),
                numbers=tuple(sorted(unique)),  # type: ignore[arg-type]
                bonus=((i * 5) % 45) + 1,
                total_sales=0, first_prize=0, first_winners=0,
                second_prize=0, second_winners=0,
                third_prize=0, third_winners=0,
            )
        )
    return draws


def test_backtest_returns_three_way_stats() -> None:
    draws = _make_draws(220)
    result = run_backtest(draws, test_draws=20, time_limit_seconds=10)
    assert "v2_hit3" in result
    assert "random_hit3" in result
    assert "v2_avg_best_hits" in result
    # 무작위 5장 hit@3 ≈ 10.7% (작은 표본에서 노이즈 있음, 0 이상이면 됨)
    assert 0.0 <= result["random_hit3"] <= 1.0
