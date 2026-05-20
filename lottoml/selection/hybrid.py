"""4 covering + 1 dream 포트폴리오 조립."""
from __future__ import annotations

from dataclasses import dataclass

from .covering import CoveringResult, select_covering
from .types import Ticket


@dataclass(frozen=True)
class PortfolioEntry:
    ticket_idx: int
    numbers: Ticket
    source: str  # "covering" or "dream"


@dataclass(frozen=True)
class Portfolio:
    tickets: list[PortfolioEntry]
    covering: CoveringResult


def build_portfolio(
    probs: dict[int, float],
    *,
    seed: int,
    pool_size: int = 30,
    num_candidates: int = 5000,
    time_limit_seconds: float = 30.0,
) -> Portfolio:
    """45개 확률 → 풀 선정 → covering 4티켓 → dream 1티켓 합쳐서 5티켓 반환."""
    if len(probs) != 45 or any(n not in probs for n in range(1, 46)):
        raise ValueError("45개 번호의 확률이 모두 필요합니다")

    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    pool = sorted(n for n, _ in ordered[:pool_size])

    covering = select_covering(
        pool,
        probs,
        num_tickets=4,
        seed=seed,
        num_candidates=num_candidates,
        time_limit_seconds=time_limit_seconds,
    )

    top6: Ticket = tuple(sorted(n for n, _ in ordered[:6]))  # type: ignore[assignment]
    if top6 in covering.tickets:
        # 7번째 번호로 대체
        alt: Ticket = tuple(sorted(
            [n for n, _ in ordered[:5]] + [ordered[6][0]]
        ))  # type: ignore[assignment]
        dream = alt
    else:
        dream = top6

    entries: list[PortfolioEntry] = []
    for idx, ticket in enumerate(covering.tickets, start=1):
        entries.append(PortfolioEntry(ticket_idx=idx, numbers=ticket, source="covering"))
    entries.append(PortfolioEntry(ticket_idx=5, numbers=dream, source="dream"))
    return Portfolio(tickets=entries, covering=covering)
