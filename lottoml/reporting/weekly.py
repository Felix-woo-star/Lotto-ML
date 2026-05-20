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
