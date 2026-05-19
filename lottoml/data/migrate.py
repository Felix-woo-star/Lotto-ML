"""v1 lotto_draws.csv → v2 draws.csv 변환."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from .storage import save_draws
from .types import Draw


def migrate_v1_csv(v1_path: Path, v2_path: Path) -> int:
    """v1 정제 CSV를 v2 스키마로 변환한다. 변환된 회차 수를 반환."""
    draws: list[Draw] = []
    with v1_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draws.append(
                Draw(
                    draw_no=int(row["draw_no"]),
                    draw_date=dt.date.fromisoformat(row["draw_date"]),
                    numbers=(
                        int(row["n1"]), int(row["n2"]), int(row["n3"]),
                        int(row["n4"]), int(row["n5"]), int(row["n6"]),
                    ),
                    bonus=int(row["bonus"]),
                    total_sales=_parse_int(row.get("total_sell_amount")),
                    first_prize=_parse_int(row.get("first_prize_amount")),
                    first_winners=_parse_int(row.get("first_prize_winner_count")),
                    second_prize=_parse_int(row.get("second_prize_amount")),
                    second_winners=_parse_int(row.get("second_prize_winner_count")),
                    third_prize=_parse_int(row.get("third_prize_amount")),
                    third_winners=_parse_int(row.get("third_prize_winner_count")),
                )
            )
    save_draws(v2_path, draws)
    return len(draws)


def _parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))
