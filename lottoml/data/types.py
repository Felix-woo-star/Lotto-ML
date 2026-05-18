"""Domain types for the lotto data layer."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Draw:
    """단일 회차 추첨 결과."""

    draw_no: int
    draw_date: dt.date
    numbers: tuple[int, int, int, int, int, int]
    bonus: int
    total_sales: int
    first_prize: int
    first_winners: int
    second_prize: int
    second_winners: int
    third_prize: int
    third_winners: int
