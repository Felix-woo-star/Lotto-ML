from lottoml.reporting.weekly import render_weekly_markdown
from lottoml.selection.hybrid import Portfolio, PortfolioEntry
from lottoml.selection.covering import CoveringResult


def _fake_portfolio() -> Portfolio:
    entries = [
        PortfolioEntry(1, (5, 11, 17, 26, 32, 39), "covering"),
        PortfolioEntry(2, (8, 14, 22, 28, 35, 42), "covering"),
        PortfolioEntry(3, (3, 19, 25, 31, 37, 44), "covering"),
        PortfolioEntry(4, (12, 16, 24, 29, 33, 40), "covering"),
        PortfolioEntry(5, (7, 18, 23, 30, 36, 41), "dream"),
    ]
    covering = CoveringResult(tickets=[e.numbers for e in entries[:4]], fallback_used=False, objective=0.0)
    return Portfolio(tickets=entries, covering=covering)


def test_render_weekly_markdown_contains_all_tickets() -> None:
    probs = {n: 0.1 + n / 1000 for n in range(1, 46)}
    pool = sorted(range(16, 46))
    md = render_weekly_markdown(
        draw_no=1225,
        draw_date="2026-05-24",
        portfolio=_fake_portfolio(),
        probs=probs,
        pool=pool,
    )
    assert "# 1225회 추천" in md
    assert "2026-05-24" in md
    assert "5 11 17 26 32 39" in md
    assert "dream" in md
    assert "## 후보 풀" in md


def test_render_weekly_markdown_has_result_placeholder() -> None:
    probs = {n: 0.1 for n in range(1, 46)}
    md = render_weekly_markdown(
        draw_no=1225, draw_date="2026-05-24",
        portfolio=_fake_portfolio(), probs=probs, pool=list(range(1, 31)),
    )
    assert "**당첨번호**: TBD" in md


from lottoml.reporting.weekly import render_weekly_results_section


def test_render_results_section_with_completed_draw() -> None:
    section = render_weekly_results_section(
        winning=(5, 11, 17, 26, 32, 39),
        bonus=20,
        per_ticket=[
            {"idx": 1, "source": "covering", "numbers": (5, 11, 17, 23, 32, 39), "hits": 5, "tier": 3, "payout": 1_500_000},
            {"idx": 2, "source": "covering", "numbers": (1, 2, 3, 4, 5, 6), "hits": 1, "tier": None, "payout": 0},
        ],
        total_payout=1_500_000,
        ticket_price=1_000,
    )
    assert "5 11 17 26 32 39" in section
    assert "보너스 20" in section
    assert "1번" in section
    assert "5개 적중" in section
    assert "1,500,000원" in section
    assert "회차 손익" in section
