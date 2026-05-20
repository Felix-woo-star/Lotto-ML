"""portfolio.csv 추천 이력 + 사후 결과 영속."""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

PORTFOLIO_FIELDS = [
    "draw_no", "ticket_idx", "n1", "n2", "n3", "n4", "n5", "n6",
    "source", "recommended_at",
    "actual_n1", "actual_n2", "actual_n3", "actual_n4", "actual_n5", "actual_n6",
    "actual_bonus", "hits", "tier", "payout", "result_recorded_at",
]


@dataclass(frozen=True)
class PortfolioRecord:
    draw_no: int
    ticket_idx: int
    numbers: tuple[int, int, int, int, int, int]
    source: str
    recommended_at: dt.datetime
    actual_numbers: tuple[int, ...] = field(default=())
    actual_bonus: int | None = None
    hits: int | None = None
    tier: int | None = None
    payout: int | None = None
    result_recorded_at: dt.datetime | None = None


def append_recommendation(path: Path, record: PortfolioRecord) -> None:
    """새 추천을 portfolio.csv 끝에 추가."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PORTFOLIO_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(_record_to_row(record))


def load_portfolio_records(path: Path) -> list[PortfolioRecord]:
    """portfolio.csv 전체 레코드를 읽어 반환."""
    if not path.exists():
        return []
    records: list[PortfolioRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(_row_to_record(row))
    return records


def record_result(
    path: Path,
    *,
    draw_no: int,
    ticket_idx: int,
    actual_numbers: tuple[int, ...],
    actual_bonus: int,
    hits: int,
    tier: int | None,
    payout: int,
    recorded_at: dt.datetime,
) -> None:
    """추천된 티켓에 사후 결과 컬럼을 채워 갱신한다."""
    rows = load_portfolio_records(path)
    updated = False
    for idx, record in enumerate(rows):
        if record.draw_no == draw_no and record.ticket_idx == ticket_idx:
            rows[idx] = PortfolioRecord(
                draw_no=record.draw_no,
                ticket_idx=record.ticket_idx,
                numbers=record.numbers,
                source=record.source,
                recommended_at=record.recommended_at,
                actual_numbers=actual_numbers,
                actual_bonus=actual_bonus,
                hits=hits,
                tier=tier,
                payout=payout,
                result_recorded_at=recorded_at,
            )
            updated = True
            break
    if not updated:
        raise KeyError(f"draw {draw_no} ticket {ticket_idx} 추천 기록을 찾지 못함")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PORTFOLIO_FIELDS)
        writer.writeheader()
        for record in rows:
            writer.writerow(_record_to_row(record))


def _record_to_row(record: PortfolioRecord) -> dict[str, str]:
    actual = list(record.actual_numbers) + [""] * (6 - len(record.actual_numbers))
    return {
        "draw_no": str(record.draw_no),
        "ticket_idx": str(record.ticket_idx),
        "n1": str(record.numbers[0]), "n2": str(record.numbers[1]),
        "n3": str(record.numbers[2]), "n4": str(record.numbers[3]),
        "n5": str(record.numbers[4]), "n6": str(record.numbers[5]),
        "source": record.source,
        "recommended_at": record.recommended_at.isoformat(),
        "actual_n1": str(actual[0]) if actual[0] != "" else "",
        "actual_n2": str(actual[1]) if actual[1] != "" else "",
        "actual_n3": str(actual[2]) if actual[2] != "" else "",
        "actual_n4": str(actual[3]) if actual[3] != "" else "",
        "actual_n5": str(actual[4]) if actual[4] != "" else "",
        "actual_n6": str(actual[5]) if actual[5] != "" else "",
        "actual_bonus": str(record.actual_bonus) if record.actual_bonus is not None else "",
        "hits": str(record.hits) if record.hits is not None else "",
        "tier": str(record.tier) if record.tier is not None else "",
        "payout": str(record.payout) if record.payout is not None else "",
        "result_recorded_at": (
            record.result_recorded_at.isoformat() if record.result_recorded_at else ""
        ),
    }


def _row_to_record(row: dict[str, str]) -> PortfolioRecord:
    actual_nums = tuple(
        int(row[f"actual_n{i}"]) for i in range(1, 7) if row.get(f"actual_n{i}")
    )
    return PortfolioRecord(
        draw_no=int(row["draw_no"]),
        ticket_idx=int(row["ticket_idx"]),
        numbers=(
            int(row["n1"]), int(row["n2"]), int(row["n3"]),
            int(row["n4"]), int(row["n5"]), int(row["n6"]),
        ),
        source=row["source"],
        recommended_at=dt.datetime.fromisoformat(row["recommended_at"]),
        actual_numbers=actual_nums,
        actual_bonus=int(row["actual_bonus"]) if row.get("actual_bonus") else None,
        hits=int(row["hits"]) if row.get("hits") else None,
        tier=int(row["tier"]) if row.get("tier") else None,
        payout=int(row["payout"]) if row.get("payout") else None,
        result_recorded_at=(
            dt.datetime.fromisoformat(row["result_recorded_at"])
            if row.get("result_recorded_at")
            else None
        ),
    )
