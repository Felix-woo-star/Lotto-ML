"""누적 통계 markdown."""
from __future__ import annotations

from collections import defaultdict
from io import StringIO

from ..data.portfolio import PortfolioRecord

TICKET_PRICE = 1000


def render_history_markdown(records: list[PortfolioRecord]) -> str:
    scored = [r for r in records if r.hits is not None]
    if not scored:
        return "# 누적 통계\n\n이력 없음 - 아직 채점된 추천이 없습니다.\n"

    out = StringIO()
    by_draw: dict[int, list[PortfolioRecord]] = defaultdict(list)
    for record in scored:
        by_draw[record.draw_no].append(record)

    weeks = len(by_draw)
    total_bet = len(scored) * TICKET_PRICE
    total_payout = sum(r.payout or 0 for r in scored)
    profit = total_payout - total_bet
    payback_rate = (total_payout / total_bet) if total_bet else 0.0

    tier_counts = {tier: 0 for tier in range(1, 6)}
    for record in scored:
        if record.tier is not None:
            tier_counts[record.tier] += 1

    hit3_weeks = sum(
        1 for draw_records in by_draw.values()
        if any((r.hits or 0) >= 3 for r in draw_records)
    )

    out.write("# 누적 통계\n\n")
    draw_nos = sorted(by_draw)
    out.write(f"## 전체 ({draw_nos[0]}~{draw_nos[-1]}회, {weeks}주)\n\n")
    out.write(f"- 총 베팅: {total_bet:,}원\n")
    out.write(f"- 총 환급: {total_payout:,}원\n")
    out.write(f"- 순손익: {profit:+,}원\n")
    out.write(f"- 환급률: {payback_rate * 100:.1f}%  (이론 상한 50%)\n\n")

    out.write("## 등수별 적중\n\n")
    for tier in range(1, 6):
        out.write(f"- {tier}등: {tier_counts[tier]}회\n")

    out.write(f"\n## 핵심 메트릭\n\n")
    out.write(f"- hit@3 누적 적중률: {hit3_weeks / weeks * 100:.1f}% ({hit3_weeks}/{weeks})\n")
    out.write(f"- 회당 평균 best_hits: {_avg_best_hits(by_draw):.2f}\n\n")

    out.write("ℹ️  표본이 50주 미만이면 통계 노이즈가 큽니다.\n")
    return out.getvalue()


def _avg_best_hits(by_draw: dict[int, list[PortfolioRecord]]) -> float:
    bests = [max(r.hits or 0 for r in records) for records in by_draw.values()]
    return sum(bests) / len(bests) if bests else 0.0
