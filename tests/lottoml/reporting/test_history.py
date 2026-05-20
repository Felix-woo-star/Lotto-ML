import datetime as dt

from lottoml.data.portfolio import PortfolioRecord
from lottoml.reporting.history import render_history_markdown


def _record(draw_no: int, ticket_idx: int, hits: int, tier: int | None, payout: int) -> PortfolioRecord:
    return PortfolioRecord(
        draw_no=draw_no, ticket_idx=ticket_idx,
        numbers=(1, 2, 3, 4, 5, 6), source="covering",
        recommended_at=dt.datetime(2026, 5, 19, 9),
        actual_numbers=(1, 2, 3, 4, 5, 6),
        actual_bonus=7,
        hits=hits, tier=tier, payout=payout,
        result_recorded_at=dt.datetime(2026, 5, 26, 9),
    )


def test_history_summarises_basic_stats() -> None:
    records = [_record(1225, i, hits=2, tier=None, payout=0) for i in range(1, 6)]
    records += [_record(1226, 1, hits=3, tier=5, payout=5000)]
    records += [_record(1226, i, hits=1, tier=None, payout=0) for i in range(2, 6)]
    md = render_history_markdown(records)
    assert "총 베팅: 10,000원" in md
    assert "총 환급: 5,000원" in md
    assert "5등: 1회" in md
    assert "hit@3 누적 적중률" in md


def test_history_handles_empty() -> None:
    md = render_history_markdown([])
    assert "이력 없음" in md
