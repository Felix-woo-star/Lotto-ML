import datetime as dt
from pathlib import Path

from lottoml.data.types import Draw
from lottoml.model.train import train_ranker
from lottoml.model.predict import predict_probabilities


def _make_synthetic_draws(n: int = 250) -> list[Draw]:
    draws = []
    for i in range(1, n + 1):
        numbers = tuple(((i * 7 + k * 11) % 45) + 1 for k in range(6))
        # ensure unique 6 numbers
        unique = []
        for v in numbers:
            if v not in unique:
                unique.append(v)
        while len(unique) < 6:
            unique.append((max(unique) % 45) + 1)
        draws.append(
            Draw(
                draw_no=i,
                draw_date=dt.date(2026, 1, 1),
                numbers=tuple(sorted(unique[:6])),  # type: ignore[arg-type]
                bonus=((i * 5) % 45) + 1,
                total_sales=0, first_prize=0, first_winners=0,
                second_prize=0, second_winners=0,
                third_prize=0, third_winners=0,
            )
        )
    return draws


def test_train_ranker_produces_calibrated_model(tmp_path: Path) -> None:
    draws = _make_synthetic_draws(250)
    out_path = tmp_path / "ranker.pkl"
    meta_path = tmp_path / "ranker_meta.json"
    metrics = train_ranker(draws, out_model=out_path, out_meta=meta_path, holdout=20)
    assert out_path.exists()
    assert meta_path.exists()
    # 메트릭 필드 존재 확인
    assert "brier" in metrics
    assert "ece" in metrics
    assert "top6_hit" in metrics


def test_predict_probabilities_returns_45_values(tmp_path: Path) -> None:
    draws = _make_synthetic_draws(250)
    out_path = tmp_path / "ranker.pkl"
    meta_path = tmp_path / "ranker_meta.json"
    train_ranker(draws, out_model=out_path, out_meta=meta_path, holdout=20)

    probs = predict_probabilities(draws, model_path=out_path)
    assert len(probs) == 45
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert set(probs.keys()) == set(range(1, 46))
