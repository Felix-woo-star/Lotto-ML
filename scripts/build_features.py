#!/usr/bin/env python3
"""Build number-level feature dataset for lotto draws."""
import argparse
import csv
import datetime as dt
import json
import os
from collections import deque
from dataclasses import dataclass
from itertools import combinations

import pyarrow as pa
import pyarrow.parquet as pq

NUMBER_MIN = 1
NUMBER_MAX = 45


@dataclass(frozen=True)
class Draw:
    draw_no: int
    draw_date: dt.date
    numbers: tuple[int, int, int, int, int, int]
    bonus: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lotto feature dataset.")
    parser.add_argument(
        "--in-processed",
        default="data/processed/lotto_draws.csv",
        help="Input processed CSV path.",
    )
    parser.add_argument(
        "--out-parquet",
        default="data/features/lotto_features.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--out-config",
        default="data/features/feature_config.json",
        help="Feature config output path.",
    )
    parser.add_argument(
        "--windows",
        default="5,10,20,50,100",
        help="Frequency window sizes (comma-separated).",
    )
    parser.add_argument(
        "--ewm",
        default="0.9,0.95",
        help="Exponential decay rates (comma-separated).",
    )
    parser.add_argument(
        "--cooc-windows",
        default="20,50",
        help="Co-occurrence windows against previous draw numbers (comma-separated).",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=None,
        help="Minimum history length to start feature generation.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def normalize_windows(values: list[int]) -> list[int]:
    return sorted({value for value in values if value > 0})


def parse_draw_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        if len(value) == 8 and value.isdigit():
            return dt.date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
        raise


def load_draws(path: str) -> list[Draw]:
    draws: list[Draw] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numbers = tuple(int(row[f"n{i}"]) for i in range(1, 7))
            draws.append(
                Draw(
                    draw_no=int(row["draw_no"]),
                    draw_date=parse_draw_date(row["draw_date"]),
                    numbers=numbers,
                    bonus=int(row.get("bonus", 0) or 0),
                )
            )
    return sorted(draws, key=lambda item: item.draw_no)


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def ewm_column_name(value: float) -> str:
    normalized = str(value).replace(".", "_")
    return f"ewm_{normalized}"


def build_feature_rows(
    draws: list[Draw],
    windows: list[int],
    ewm_rates: list[float],
    cooc_windows: list[int],
    min_history: int,
) -> tuple[list[dict], dict]:
    window_counts = {window: [0] * (NUMBER_MAX + 1) for window in windows}
    window_queues = {window: deque() for window in windows}
    window_odd_totals = {window: 0 for window in windows}
    window_high_totals = {window: 0 for window in windows}
    window_sum_totals = {window: 0 for window in windows}
    window_consecutive_totals = {window: 0 for window in windows}
    window_bucket_counts = {window: [0, 0, 0, 0] for window in windows}
    window_mod_counts = {window: [0] * 10 for window in windows}
    pair_window_counts = {
        window: [[0] * (NUMBER_MAX + 1) for _ in range(NUMBER_MAX + 1)]
        for window in windows
    }

    bonus_window_counts = {window: [0] * (NUMBER_MAX + 1) for window in windows}
    bonus_window_queues = {window: deque() for window in windows}
    last_seen = [0] * (NUMBER_MAX + 1)
    bonus_last_seen = [0] * (NUMBER_MAX + 1)
    ewm_values = {rate: [0.0] * (NUMBER_MAX + 1) for rate in ewm_rates}

    rows: list[dict] = []
    numbers_range = list(range(NUMBER_MIN, NUMBER_MAX + 1))
    max_window = max(windows) if windows else 0
    max_cooc_window = max(cooc_windows) if cooc_windows else 0
    start_index = max(min_history, max_window, max_cooc_window, 1)
    draw_sets = [set(draw.numbers) for draw in draws]

    trend_pairs = []
    if 10 in windows and 50 in windows:
        trend_pairs.append((10, 50))
    if 20 in windows and 100 in windows:
        trend_pairs.append((20, 100))

    rate_diff_pairs = []
    if 5 in windows and 20 in windows:
        rate_diff_pairs.append((5, 20))
    if 10 in windows and 50 in windows:
        rate_diff_pairs.append((10, 50))

    for idx, draw in enumerate(draws):
        if idx >= start_index:
            labels = set(draw.numbers)
            prev_numbers = draw_sets[idx - 1]
            prev_bonus = draws[idx - 1].bonus

            cooc_counts = {
                cooc_window: [0] * (NUMBER_MAX + 1) for cooc_window in cooc_windows
            }
            for cooc_window in cooc_windows:
                start = max(0, idx - cooc_window)
                for history_idx in range(start, idx):
                    history_numbers = draw_sets[history_idx]
                    if history_numbers.intersection(prev_numbers):
                        for number in history_numbers:
                            cooc_counts[cooc_window][number] += 1

            prev_sum = sum(prev_numbers)
            prev_odd_count = sum(1 for value in prev_numbers if value % 2 == 1)
            prev_high_count = sum(1 for value in prev_numbers if value > 22)

            for number in numbers_range:
                bucket = ((number - 1) // 15) + 1
                row = {
                    "draw_no": draw.draw_no,
                    "number": number,
                    "label": 1 if number in labels else 0,
                    "number_bucket": bucket,
                    "number_is_odd": 1 if number % 2 == 1 else 0,
                    "number_is_high": 1 if number > 22 else 0,
                    "number_mod_10": number % 10,
                    "draw_month": draw.draw_date.month,
                    "draw_weekday": draw.draw_date.weekday(),
                    "prev_draw_sum": prev_sum,
                    "prev_draw_odd_count": prev_odd_count,
                    "prev_draw_high_count": prev_high_count,
                    "is_prev_draw_number": 1 if number in prev_numbers else 0,
                    "is_prev_bonus_number": 1 if number == prev_bonus else 0,
                    "adjacent_prev_count": sum(
                        1 for prev_value in prev_numbers if abs(number - prev_value) == 1
                    ),
                    "distance_to_prev_min": min(
                        abs(number - prev_value) for prev_value in prev_numbers
                    ),
                }
                for window in windows:
                    freq = window_counts[window][number]
                    recent_mean_number = window_sum_totals[window] / (window * 6)
                    pair_values = [
                        pair_window_counts[window][number][prev_number]
                        for prev_number in prev_numbers
                        if prev_number != number
                    ]
                    row[f"freq_{window}"] = freq
                    row[f"freq_rate_{window}"] = freq / window
                    row[f"bucket_freq_{window}"] = window_bucket_counts[window][bucket]
                    row[f"bucket_rate_{window}"] = (
                        window_bucket_counts[window][bucket] / (window * 6)
                    )
                    row[f"mod_freq_{window}"] = window_mod_counts[window][number % 10]
                    row[f"mod_rate_{window}"] = (
                        window_mod_counts[window][number % 10] / (window * 6)
                    )
                    row[f"bonus_freq_{window}"] = bonus_window_counts[window][number]
                    row[f"bonus_freq_rate_{window}"] = (
                        bonus_window_counts[window][number] / window
                    )
                    row[f"distance_to_recent_mean_{window}"] = abs(
                        number - recent_mean_number
                    )
                    row[f"recent_mean_gap_signed_{window}"] = (
                        number - recent_mean_number
                    )
                    row[f"recent_odd_rate_{window}"] = (
                        window_odd_totals[window] / (window * 6)
                    )
                    row[f"recent_high_rate_{window}"] = (
                        window_high_totals[window] / (window * 6)
                    )
                    row[f"recent_sum_avg_{window}"] = window_sum_totals[window] / window
                    row[f"recent_consecutive_rate_{window}"] = (
                        window_consecutive_totals[window] / window
                    )
                    row[f"pair_with_prev_sum_{window}"] = sum(pair_values)
                    row[f"pair_with_prev_avg_{window}"] = (
                        sum(pair_values) / len(pair_values) if pair_values else 0.0
                    )
                    row[f"pair_with_prev_max_{window}"] = (
                        max(pair_values) if pair_values else 0
                    )

                last_seen_draw_no = last_seen[number]
                row["last_seen_draw_no"] = last_seen_draw_no
                row["last_seen_gap"] = (
                    draw.draw_no - last_seen_draw_no if last_seen_draw_no else 0
                )
                row["recency_inv"] = (
                    1.0 / (1.0 + row["last_seen_gap"]) if row["last_seen_gap"] > 0 else 1.0
                )

                bonus_last_seen_draw_no = bonus_last_seen[number]
                row["bonus_last_seen_draw_no"] = bonus_last_seen_draw_no
                row["bonus_last_seen_gap"] = (
                    draw.draw_no - bonus_last_seen_draw_no
                    if bonus_last_seen_draw_no
                    else 0
                )

                for recent, long in trend_pairs:
                    recent_freq = window_counts[recent][number]
                    long_freq = window_counts[long][number]
                    row[f"trend_{recent}_{long}"] = recent_freq - (
                        long_freq * recent / long
                    )
                    expected_recent = (long_freq * recent / long) + 1e-6
                    row[f"trend_ratio_{recent}_{long}"] = recent_freq / expected_recent

                for short, long in rate_diff_pairs:
                    row[f"freq_rate_diff_{short}_{long}"] = (
                        row[f"freq_rate_{short}"] - row[f"freq_rate_{long}"]
                    )

                for rate in ewm_rates:
                    row[ewm_column_name(rate)] = ewm_values[rate][number]

                for cooc_window in cooc_windows:
                    row[f"cooc_prev_{cooc_window}"] = cooc_counts[cooc_window][number]

                rows.append(row)

        for window in windows:
            queue = window_queues[window]
            odd_count = sum(1 for value in draw.numbers if value % 2 == 1)
            high_count = sum(1 for value in draw.numbers if value > 22)
            has_consecutive = 0
            sorted_numbers = sorted(draw.numbers)
            for left, right in zip(sorted_numbers, sorted_numbers[1:]):
                if right - left == 1:
                    has_consecutive = 1
                    break
            draw_sum = sum(draw.numbers)

            queue.append(
                (
                    draw.numbers,
                    odd_count,
                    high_count,
                    draw_sum,
                    has_consecutive,
                )
            )
            for number in draw.numbers:
                window_counts[window][number] += 1
                window_bucket_counts[window][((number - 1) // 15) + 1] += 1
                window_mod_counts[window][number % 10] += 1
            for left, right in combinations(sorted_numbers, 2):
                pair_window_counts[window][left][right] += 1
                pair_window_counts[window][right][left] += 1
            window_odd_totals[window] += odd_count
            window_high_totals[window] += high_count
            window_sum_totals[window] += draw_sum
            window_consecutive_totals[window] += has_consecutive
            if len(queue) > window:
                (
                    outgoing,
                    outgoing_odd,
                    outgoing_high,
                    outgoing_sum,
                    outgoing_consecutive,
                ) = queue.popleft()
                for number in outgoing:
                    window_counts[window][number] -= 1
                    window_bucket_counts[window][((number - 1) // 15) + 1] -= 1
                    window_mod_counts[window][number % 10] -= 1
                outgoing_sorted = sorted(outgoing)
                for left, right in combinations(outgoing_sorted, 2):
                    pair_window_counts[window][left][right] -= 1
                    pair_window_counts[window][right][left] -= 1
                window_odd_totals[window] -= outgoing_odd
                window_high_totals[window] -= outgoing_high
                window_sum_totals[window] -= outgoing_sum
                window_consecutive_totals[window] -= outgoing_consecutive

            bonus_queue = bonus_window_queues[window]
            bonus_queue.append(draw.bonus)
            if NUMBER_MIN <= draw.bonus <= NUMBER_MAX:
                bonus_window_counts[window][draw.bonus] += 1
            if len(bonus_queue) > window:
                outgoing_bonus = bonus_queue.popleft()
                if NUMBER_MIN <= outgoing_bonus <= NUMBER_MAX:
                    bonus_window_counts[window][outgoing_bonus] -= 1

        for rate in ewm_rates:
            updated = ewm_values[rate]
            for number in numbers_range:
                updated[number] *= rate
            for number in draw.numbers:
                updated[number] += 1.0

        for number in draw.numbers:
            last_seen[number] = draw.draw_no
        if NUMBER_MIN <= draw.bonus <= NUMBER_MAX:
            bonus_last_seen[draw.bonus] = draw.draw_no

    stats = {
        "total_draws": len(draws),
        "rows": len(rows),
        "min_history": min_history,
        "windows": windows,
        "ewm_rates": ewm_rates,
        "cooc_windows": cooc_windows,
        "trend_pairs": trend_pairs,
        "rate_diff_pairs": rate_diff_pairs,
        "feature_groups": {
            "number": [
                "number_bucket",
                "number_is_odd",
                "number_is_high",
                "number_mod_10",
            ],
            "distribution": [
                "bucket_freq_*",
                "bucket_rate_*",
                "mod_freq_*",
                "mod_rate_*",
                "distance_to_recent_mean_*",
                "recent_mean_gap_signed_*",
            ],
            "recency": ["last_seen_gap", "bonus_last_seen_gap", "recency_inv"],
            "context": [
                "prev_draw_sum",
                "prev_draw_odd_count",
                "prev_draw_high_count",
                "is_prev_draw_number",
                "is_prev_bonus_number",
                "adjacent_prev_count",
                "distance_to_prev_min",
            ],
            "pairwise": [
                "pair_with_prev_sum_*",
                "pair_with_prev_avg_*",
                "pair_with_prev_max_*",
            ],
        },
    }
    return rows, stats


def write_config(path: str, payload: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_parquet(path: str, rows: list[dict]) -> None:
    ensure_parent_dir(path)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.in_processed):
        print(f"입력 파일이 없습니다: {args.in_processed}")
        return 1

    windows = normalize_windows(parse_int_list(args.windows))
    ewm_rates = parse_float_list(args.ewm)
    cooc_windows = normalize_windows(parse_int_list(args.cooc_windows))
    if args.min_history is not None:
        min_history = args.min_history
    elif windows:
        min_history = max(windows)
    else:
        min_history = 0

    draws = load_draws(args.in_processed)
    rows, stats = build_feature_rows(
        draws, windows, ewm_rates, cooc_windows, min_history
    )

    if not rows:
        print("생성된 피처 행이 없습니다. min-history와 입력 데이터 크기를 확인하세요.")
        return 1

    write_parquet(args.out_parquet, rows)
    write_config(
        args.out_config,
        {
            "source": args.in_processed,
            "out_parquet": args.out_parquet,
            "windows": windows,
            "ewm_rates": ewm_rates,
            "cooc_windows": cooc_windows,
            "min_history": min_history,
            "stats": stats,
        },
    )
    print(f"피처 {len(rows)}행을 저장했습니다: {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
