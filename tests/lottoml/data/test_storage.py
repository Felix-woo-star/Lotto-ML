import datetime as dt
from pathlib import Path

from lottoml.data.storage import load_draws, save_draws
from lottoml.data.types import Draw


def test_storage_roundtrip(tmp_path: Path) -> None:
    draws = [
        Draw(
            draw_no=1,
            draw_date=dt.date(2002, 12, 7),
            numbers=(10, 23, 29, 33, 37, 40),
            bonus=16,
            total_sales=3681782000,
            first_prize=0,
            first_winners=0,
            second_prize=143934100,
            second_winners=1,
            third_prize=5140500,
            third_winners=28,
        ),
        Draw(
            draw_no=2,
            draw_date=dt.date(2002, 12, 14),
            numbers=(9, 13, 21, 25, 32, 42),
            bonus=2,
            total_sales=4904274000,
            first_prize=2002006800,
            first_winners=1,
            second_prize=94866800,
            second_winners=2,
            third_prize=1842000,
            third_winners=103,
        ),
    ]
    csv_path = tmp_path / "draws.csv"
    save_draws(csv_path, draws)
    loaded = load_draws(csv_path)
    assert loaded == draws


def test_load_empty_returns_empty_list(tmp_path: Path) -> None:
    csv_path = tmp_path / "draws.csv"
    assert load_draws(csv_path) == []
