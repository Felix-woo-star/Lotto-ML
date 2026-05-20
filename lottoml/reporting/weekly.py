"""주간 추천 markdown 렌더."""
from __future__ import annotations

from io import StringIO

from ..selection.hybrid import Portfolio


def render_weekly_markdown(
    *,
    draw_no: int,
    draw_date: str,
    portfolio: Portfolio,
    probs: dict[int, float],
    pool: list[int],
) -> str:
    """1주차 회차 추천 markdown 문자열을 만든다."""
    out = StringIO()
    out.write(f"# {draw_no}회 추천 ({draw_date} 추첨)\n\n")
    out.write("## 추천 포트폴리오\n\n")
    out.write("| 티켓 | 번호 | 출처 |\n")
    out.write("|------|------|------|\n")
    for entry in portfolio.tickets:
        nums = " ".join(str(n) for n in entry.numbers)
        out.write(f"| {entry.ticket_idx} | {nums} | {entry.source} |\n")

    out.write("\n## ML 확률 Top-10\n\n")
    out.write("| 순위 | 번호 | 확률 |\n")
    out.write("|------|------|------|\n")
    top10 = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    for rank, (number, p) in enumerate(top10, start=1):
        out.write(f"| {rank} | {number} | {p:.4f} |\n")

    out.write("\n## 후보 풀\n\n")
    out.write(" ".join(str(n) for n in sorted(pool)))
    out.write("\n\n## 결과 (추첨 후 자동 갱신)\n\n")
    out.write("- **당첨번호**: TBD\n")
    out.write("- **포트폴리오 best hits**: TBD\n")
    out.write("- **환급**: TBD원\n")

    return out.getvalue()


def render_weekly_results_section(
    *,
    winning: tuple[int, ...],
    bonus: int,
    per_ticket: list[dict],
    total_payout: int,
    ticket_price: int = 1_000,
) -> str:
    """추첨 결과로 채운 결과 섹션 markdown."""
    out = StringIO()
    out.write("## 결과\n\n")
    out.write(f"- **당첨번호**: {' '.join(str(n) for n in winning)} (보너스 {bonus})\n\n")
    out.write("| 티켓 | 번호 | 적중 | 등수 | 환급 |\n")
    out.write("|------|------|------|------|------|\n")
    best_hits = 0
    for row in per_ticket:
        nums = " ".join(str(n) for n in row["numbers"])
        tier_str = f"{row['tier']}등" if row["tier"] else "-"
        hit_str = f"{row['hits']}개 적중"
        payout_str = f"{row['payout']:,}원"
        out.write(f"| {row['idx']}번 ({row['source']}) | {nums} | {hit_str} | {tier_str} | {payout_str} |\n")
        best_hits = max(best_hits, row["hits"])
    total_cost = len(per_ticket) * ticket_price
    profit = total_payout - total_cost
    out.write(f"\n- **포트폴리오 best hits**: {best_hits}개\n")
    out.write(f"- **회차 손익**: {profit:+,}원 (지출 {total_cost:,} / 환급 {total_payout:,})\n")
    return out.getvalue()


def update_weekly_markdown_with_results(
    path, section_markdown: str, *, draw_date: str | None = None
) -> None:
    """기존 weekly.md의 '## 결과' 섹션을 새 결과로 교체한다."""
    text = path.read_text(encoding="utf-8")
    if draw_date is not None:
        text = text.replace("(TBD 추첨)", f"({draw_date} 추첨)", 1)
    marker = "\n## 결과"
    idx = text.find(marker)
    if idx == -1:
        path.write_text(text.rstrip() + "\n\n" + section_markdown, encoding="utf-8")
        return
    path.write_text(text[:idx].rstrip() + "\n\n" + section_markdown, encoding="utf-8")
