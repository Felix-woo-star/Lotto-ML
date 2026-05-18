"""draws.csv 영속 계층."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from .types import Draw

FIELDNAMES = [
    "draw_no", "draw_date",
    "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
    "total_sales",
    "first_prize", "first_winners",
    "second_prize", "second_winners",
    "third_prize", "third_winners",
]


def load_draws(path: Path) -> list[Draw]:
    """draws.csv에서 회차 목록을 회차번호 오름차순으로 반환한다."""
    if not path.exists():
        return []
    rows: list[Draw] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_row_to_draw(row))
    return sorted(rows, key=lambda draw: draw.draw_no)


def save_draws(path: Path, draws: list[Draw]) -> None:
    """회차 목록을 CSV로 저장한다 (덮어쓰기)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for draw in sorted(draws, key=lambda d: d.draw_no):
            writer.writerow(_draw_to_row(draw))


def _draw_to_row(draw: Draw) -> dict[str, str]:
    n1, n2, n3, n4, n5, n6 = draw.numbers
    return {
        "draw_no": str(draw.draw_no),
        "draw_date": draw.draw_date.isoformat(),
        "n1": str(n1), "n2": str(n2), "n3": str(n3),
        "n4": str(n4), "n5": str(n5), "n6": str(n6),
        "bonus": str(draw.bonus),
        "total_sales": str(draw.total_sales),
        "first_prize": str(draw.first_prize),
        "first_winners": str(draw.first_winners),
        "second_prize": str(draw.second_prize),
        "second_winners": str(draw.second_winners),
        "third_prize": str(draw.third_prize),
        "third_winners": str(draw.third_winners),
    }


def _row_to_draw(row: dict[str, str]) -> Draw:
    return Draw(
        draw_no=int(row["draw_no"]),
        draw_date=dt.date.fromisoformat(row["draw_date"]),
        numbers=(
            int(row["n1"]), int(row["n2"]), int(row["n3"]),
            int(row["n4"]), int(row["n5"]), int(row["n6"]),
        ),
        bonus=int(row["bonus"]),
        total_sales=int(row["total_sales"]),
        first_prize=int(row["first_prize"]),
        first_winners=int(row["first_winners"]),
        second_prize=int(row["second_prize"]),
        second_winners=int(row["second_winners"]),
        third_prize=int(row["third_prize"]),
        third_winners=int(row["third_winners"]),
    )
