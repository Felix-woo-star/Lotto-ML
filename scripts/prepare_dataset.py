#!/usr/bin/env python3
"""로또 원천 데이터를 검증/정제해 가공 CSV를 만든다."""
import argparse
import csv
import datetime as dt
import os

OUTPUT_FIELDS = [
    "draw_no",
    "draw_date",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "bonus",
    "total_sell_amount",
    "first_prize_amount",
    "first_prize_winner_count",
    "first_accum_amount",
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Validate and clean lotto data.")
    parser.add_argument(
        "--in-raw",
        default="data/raw/lotto_draws.csv",
        help="Input raw CSV path.",
    )
    parser.add_argument(
        "--out-processed",
        default="data/processed/lotto_draws.csv",
        help="Output processed CSV path.",
    )
    return parser.parse_args()


def parse_int(value: str, field: str, errors: list[str]) -> int | None:
    """필수 정수 필드를 파싱하고 오류를 누적한다."""
    if value is None or value == "":
        errors.append(f"missing {field}")
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"invalid int for {field}: {value}")
        return None


def parse_optional_int(value: str) -> int | str:
    """선택 정수 필드를 파싱하며 실패 시 빈 문자열을 반환한다."""
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        return ""


def validate_row(row: dict) -> tuple[dict | None, list[str]]:
    """행 단위로 검증하고 정제된 레코드를 반환한다."""
    errors: list[str] = []

    draw_no = parse_int(row.get("draw_no"), "draw_no", errors)
    draw_date = row.get("draw_date", "")
    if not draw_date:
        errors.append("missing draw_date")
    else:
        try:
            dt.date.fromisoformat(draw_date)
        except ValueError:
            errors.append(f"invalid draw_date: {draw_date}")

    numbers = []
    for idx in range(1, 7):
        value = parse_int(row.get(f"n{idx}"), f"n{idx}", errors)
        if value is not None:
            numbers.append(value)

    bonus = parse_int(row.get("bonus"), "bonus", errors)

    if len(numbers) == 6:
        if len(set(numbers)) != 6:
            errors.append("duplicate numbers")
        if any(number < 1 or number > 45 for number in numbers):
            errors.append("numbers out of range")
    if bonus is not None:
        if bonus < 1 or bonus > 45:
            errors.append("bonus out of range")
        if bonus in numbers:
            errors.append("bonus overlaps with main numbers")

    if draw_no is None or errors:
        return None, errors

    numbers = sorted(numbers)
    cleaned = {
        "draw_no": draw_no,
        "draw_date": draw_date,
        "n1": numbers[0],
        "n2": numbers[1],
        "n3": numbers[2],
        "n4": numbers[3],
        "n5": numbers[4],
        "n6": numbers[5],
        "bonus": bonus if bonus is not None else "",
        "total_sell_amount": parse_optional_int(row.get("total_sell_amount")),
        "first_prize_amount": parse_optional_int(row.get("first_prize_amount")),
        "first_prize_winner_count": parse_optional_int(
            row.get("first_prize_winner_count")
        ),
        "first_accum_amount": parse_optional_int(row.get("first_accum_amount")),
    }
    return cleaned, []


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.in_raw):
        print(f"Missing input file: {args.in_raw}")
        return 1

    seen = set()
    rows: list[dict] = []
    invalid = 0
    duplicates = 0

    with open(args.in_raw, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned, errors = validate_row(row)
            if cleaned is None:
                invalid += 1
                continue
            draw_no = cleaned["draw_no"]
            if draw_no in seen:
                duplicates += 1
                continue
            seen.add(draw_no)
            rows.append(cleaned)

    rows.sort(key=lambda item: item["draw_no"])
    os.makedirs(os.path.dirname(args.out_processed) or ".", exist_ok=True)
    with open(args.out_processed, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        min_no = rows[0]["draw_no"]
        max_no = rows[-1]["draw_no"]
        print(
            f"Processed {len(rows)} draws ({min_no}..{max_no}). "
            f"Invalid: {invalid}, duplicates: {duplicates}."
        )
    else:
        print(f"No valid rows found. Invalid: {invalid}, duplicates: {duplicates}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
