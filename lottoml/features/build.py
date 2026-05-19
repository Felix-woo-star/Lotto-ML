"""번호별 피처 테이블 생성 (8 피처)."""
from __future__ import annotations

from collections import deque
from typing import Iterable

import pandas as pd

from ..data.types import Draw

NUMBER_MIN = 1
NUMBER_MAX = 45
NUMBERS = tuple(range(NUMBER_MIN, NUMBER_MAX + 1))
WINDOWS = (20, 50, 100)
MIN_HISTORY = max(WINDOWS)


def build_features(draws: Iterable[Draw]) -> pd.DataFrame:
    """회차 목록을 (draw_no, number, label, 8피처) 테이블로 변환한다."""
    sorted_draws = sorted(draws, key=lambda d: d.draw_no)
    if len(sorted_draws) <= MIN_HISTORY:
        return pd.DataFrame()

    window_counts = {w: {n: 0 for n in NUMBERS} for w in WINDOWS}
    window_queues = {w: deque() for w in WINDOWS}
    bonus_count_50 = {n: 0 for n in NUMBERS}
    bonus_queue_50: deque[int] = deque()
    last_seen: dict[int, int] = {n: 0 for n in NUMBERS}

    rows: list[dict] = []

    for idx, draw in enumerate(sorted_draws):
        if idx >= MIN_HISTORY:
            label_set = set(draw.numbers)
            for number in NUMBERS:
                gap = draw.draw_no - last_seen[number] if last_seen[number] else 0
                rows.append(
                    {
                        "draw_no": draw.draw_no,
                        "number": number,
                        "label": 1 if number in label_set else 0,
                        "freq_20": window_counts[20][number],
                        "freq_50": window_counts[50][number],
                        "freq_100": window_counts[100][number],
                        "last_seen_gap": gap,
                        "bonus_freq_50": bonus_count_50[number],
                        "bucket": ((number - 1) // 15) + 1,
                        "is_odd": number % 2,
                        "mod_10": number % 10,
                    }
                )

        # Update windows after emitting row for current draw
        for window in WINDOWS:
            queue = window_queues[window]
            queue.append(draw.numbers)
            for number in draw.numbers:
                window_counts[window][number] += 1
            if len(queue) > window:
                outgoing = queue.popleft()
                for number in outgoing:
                    window_counts[window][number] -= 1

        bonus_queue_50.append(draw.bonus)
        if NUMBER_MIN <= draw.bonus <= NUMBER_MAX:
            bonus_count_50[draw.bonus] += 1
        if len(bonus_queue_50) > 50:
            outgoing_bonus = bonus_queue_50.popleft()
            if NUMBER_MIN <= outgoing_bonus <= NUMBER_MAX:
                bonus_count_50[outgoing_bonus] -= 1

        for number in draw.numbers:
            last_seen[number] = draw.draw_no

    return pd.DataFrame(rows)
