import datetime as dt
from pathlib import Path

from click.testing import CliRunner

from lottoml.cli import main
from lottoml.data.storage import save_draws
from lottoml.data.types import Draw
from lottoml.model.train import train_ranker


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


def test_recommend_writes_portfolio_and_markdown(tmp_path: Path, monkeypatch) -> None:
    save_draws(tmp_path / "data" / "draws.csv", _make_draws(250))
    train_ranker(
        _make_draws(250),
        out_model=tmp_path / "data" / "models" / "ranker.pkl",
        out_meta=tmp_path / "data" / "models" / "ranker_meta.json",
        holdout=20,
    )
    # discover_latest_draw returns 250 - 새 회차 없음
    monkeypatch.setattr("lottoml.cli.discover_latest_draw", lambda **_: 250)

    runner = CliRunner()
    result = runner.invoke(
        main, ["recommend", "--root", str(tmp_path), "--time-limit", "20"]
    )
    assert result.exit_code == 0, result.output

    portfolio_csv = tmp_path / "data" / "portfolio.csv"
    assert portfolio_csv.exists()
    weekly_md = tmp_path / "reports" / "weekly" / "251.md"
    assert weekly_md.exists()
    content = weekly_md.read_text(encoding="utf-8")
    assert "251회 추천" in content


def test_recommend_updates_weekly_markdown_after_score(tmp_path: Path, monkeypatch) -> None:
    """이전 회차 추천이 채점되면 해당 weekly.md의 결과 섹션이 TBD에서 실값으로 바뀐다."""
    save_draws(tmp_path / "data" / "draws.csv", _make_draws(250))
    train_ranker(
        _make_draws(250),
        out_model=tmp_path / "data" / "models" / "ranker.pkl",
        out_meta=tmp_path / "data" / "models" / "ranker_meta.json",
        holdout=20,
    )
    monkeypatch.setattr("lottoml.cli.discover_latest_draw", lambda **_: 250)

    runner = CliRunner()
    # 첫 호출: 251회 추천 생성 (markdown은 TBD)
    runner.invoke(main, ["recommend", "--root", str(tmp_path), "--time-limit", "15"])
    md_path = tmp_path / "reports" / "weekly" / "251.md"
    assert "TBD" in md_path.read_text(encoding="utf-8")

    # 251회 draw 결과를 draws.csv에 추가
    draws_after = _make_draws(251)
    save_draws(tmp_path / "data" / "draws.csv", draws_after)
    monkeypatch.setattr("lottoml.cli.discover_latest_draw", lambda **_: 251)

    # 두 번째 호출: 251회 채점 + 252회 추천
    result = runner.invoke(main, ["recommend", "--root", str(tmp_path), "--time-limit", "15"])
    assert result.exit_code == 0, result.output
    content = md_path.read_text(encoding="utf-8")
    assert "TBD" not in content
    assert "회차 손익" in content
