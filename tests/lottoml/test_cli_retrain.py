import datetime as dt
from pathlib import Path

from click.testing import CliRunner

from lottoml.cli import main
from lottoml.data.storage import save_draws
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


def test_retrain_creates_model(tmp_path: Path) -> None:
    save_draws(tmp_path / "data" / "draws.csv", _make_draws(250))
    runner = CliRunner()
    result = runner.invoke(main, ["retrain", "--root", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "models" / "ranker.pkl").exists()
    assert (tmp_path / "data" / "models" / "ranker_meta.json").exists()
