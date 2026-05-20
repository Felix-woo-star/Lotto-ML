from lottoml.selection.hybrid import build_portfolio


def test_build_portfolio_returns_five_tickets() -> None:
    probs = {n: 0.13 + (n / 1000) for n in range(1, 46)}
    portfolio = build_portfolio(probs, seed=42, pool_size=30, time_limit_seconds=20)
    assert len(portfolio.tickets) == 5
    sources = [entry.source for entry in portfolio.tickets]
    assert sources.count("covering") == 4
    assert sources.count("dream") == 1


def test_dream_ticket_is_top6() -> None:
    probs = {n: float(n) for n in range(1, 46)}  # 큰 번호일수록 높은 확률
    portfolio = build_portfolio(probs, seed=42, pool_size=30, time_limit_seconds=20)
    dream = next(e.numbers for e in portfolio.tickets if e.source == "dream")
    assert set(dream) == {40, 41, 42, 43, 44, 45}


def test_dream_falls_back_when_duplicate() -> None:
    # 모든 covering 후보가 top6를 강제로 포함하도록 만들기는 어렵지만,
    # 함수가 중복 케이스에서도 5개의 고유 티켓을 반환하는지 확인
    probs = {n: 1.0 / 45 for n in range(1, 46)}
    portfolio = build_portfolio(probs, seed=42, pool_size=30, time_limit_seconds=15)
    unique = {frozenset(e.numbers) for e in portfolio.tickets}
    assert len(unique) == 5
