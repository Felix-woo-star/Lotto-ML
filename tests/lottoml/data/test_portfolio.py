import datetime as dt
from pathlib import Path

from lottoml.data.portfolio import (
    PortfolioRecord,
    append_recommendation,
    load_portfolio_records,
    record_result,
)


def test_append_and_load_records(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.csv"
    records = [
        PortfolioRecord(
            draw_no=1225, ticket_idx=1,
            numbers=(5, 11, 17, 26, 32, 39),
            source="covering",
            recommended_at=dt.datetime(2026, 5, 19, 9, 0, 0),
        ),
        PortfolioRecord(
            draw_no=1225, ticket_idx=2,
            numbers=(8, 14, 22, 28, 35, 42),
            source="covering",
            recommended_at=dt.datetime(2026, 5, 19, 9, 0, 0),
        ),
    ]
    for record in records:
        append_recommendation(path, record)

    loaded = load_portfolio_records(path)
    assert len(loaded) == 2
    assert loaded[0].numbers == (5, 11, 17, 26, 32, 39)


def test_record_result_updates_existing(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.csv"
    append_recommendation(
        path,
        PortfolioRecord(
            draw_no=1225, ticket_idx=1,
            numbers=(5, 11, 17, 26, 32, 39),
            source="covering",
            recommended_at=dt.datetime(2026, 5, 19, 9, 0, 0),
        ),
    )
    record_result(
        path,
        draw_no=1225,
        ticket_idx=1,
        actual_numbers=(5, 11, 17, 23, 32, 39),
        actual_bonus=20,
        hits=5,
        tier=3,
        payout=1_500_000,
        recorded_at=dt.datetime(2026, 5, 26, 9, 0, 0),
    )
    rows = load_portfolio_records(path)
    assert rows[0].hits == 5
    assert rows[0].tier == 3
    assert rows[0].payout == 1_500_000
