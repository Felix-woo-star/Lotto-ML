#!/usr/bin/env python3
"""Utilities for lotto candidate generation, portfolio selection, and payout analysis."""

from __future__ import annotations

import csv
import math
import os
import random
from itertools import combinations

NUMBER_MIN = 1
NUMBER_MAX = 45
TICKET_SIZE = 6

DEFAULT_TIER_PAYOUTS = {
    2: 50_000_000.0,
    3: 1_500_000.0,
    4: 50_000.0,
    5: 5_000.0,
}
DEFAULT_TICKET_PRICE = 1_000.0
DEFAULT_TIER1_ESTIMATE = 2_000_000_000.0


def normalize_ticket(numbers: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a ticket into a sorted tuple of six unique ints."""
    return tuple(sorted(int(number) for number in numbers))


def overlap_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the number of shared values between two tickets."""
    return len(set(left).intersection(right))


def load_draw_contexts(path: str) -> dict[int, dict]:
    """Load draw metadata needed for bonus and payout evaluation."""
    if not os.path.exists(path):
        return {}

    contexts: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draw_no_text = row.get("draw_no", "")
            if not draw_no_text:
                continue
            try:
                draw_no = int(draw_no_text)
            except ValueError:
                continue
            numbers = []
            valid_numbers = True
            for idx in range(1, 7):
                try:
                    numbers.append(int(row.get(f"n{idx}", "") or 0))
                except ValueError:
                    valid_numbers = False
                    break
            if not valid_numbers:
                continue

            contexts[draw_no] = {
                "draw_no": draw_no,
                "numbers": normalize_ticket(numbers),
                "bonus": parse_int(row.get("bonus")),
                "first_prize_amount": parse_float(row.get("first_prize_amount")),
                "first_prize_winner_count": parse_int(row.get("first_prize_winner_count")),
                "first_accum_amount": parse_float(row.get("first_accum_amount")),
            }
    return contexts


def parse_int(value: object) -> int:
    """Parse an integer-like value, returning 0 on failure."""
    try:
        return int(value) if value not in ("", None) else 0
    except (TypeError, ValueError):
        return 0


def parse_float(value: object) -> float:
    """Parse a float-like value, returning 0.0 on failure."""
    try:
        return float(value) if value not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_probability_map(
    numbers: list[int] | tuple[int, ...],
    probabilities: list[float] | tuple[float, ...],
) -> dict[int, float]:
    """Zip numbers and probabilities into a plain dict."""
    return {int(number): max(float(probability), 0.0) for number, probability in zip(numbers, probabilities)}


def weighted_sample_without_replacement(
    rng: random.Random,
    numbers: list[int],
    weights: list[float],
    k: int = TICKET_SIZE,
) -> tuple[int, ...]:
    """Sample a single lotto ticket using weighted sampling without replacement."""
    pool_numbers = numbers[:]
    pool_weights = weights[:]
    picks: list[int] = []

    for _ in range(min(k, len(pool_numbers))):
        total = sum(pool_weights)
        if total <= 0:
            index = rng.randrange(len(pool_numbers))
        else:
            target = rng.random() * total
            cumulative = 0.0
            index = len(pool_weights) - 1
            for idx, weight in enumerate(pool_weights):
                cumulative += weight
                if cumulative >= target:
                    index = idx
                    break
        picks.append(pool_numbers.pop(index))
        pool_weights.pop(index)

    return normalize_ticket(picks)


def transform_weights(probability_by_number: dict[int, float], temperature: float) -> dict[int, float]:
    """Convert probabilities to positive sampling weights."""
    safe_temperature = max(float(temperature), 1e-3)
    transformed: dict[int, float] = {}
    for number in range(NUMBER_MIN, NUMBER_MAX + 1):
        probability = max(probability_by_number.get(number, 0.0), 1e-9)
        transformed[number] = math.pow(probability, 1.0 / safe_temperature)
    return transformed


def candidate_score(numbers: tuple[int, ...], probability_by_number: dict[int, float]) -> float:
    """Score a ticket by the summed marginal probabilities of its numbers."""
    return float(sum(probability_by_number.get(number, 0.0) for number in numbers))


def generate_candidates(
    probability_by_number: dict[int, float],
    num_candidates: int,
    *,
    top_pool_size: int,
    temperature: float,
    seed: int,
) -> list[dict]:
    """Generate unique ticket candidates from per-number probabilities."""
    num_candidates = max(1, int(num_candidates))
    top_pool_size = max(TICKET_SIZE, int(top_pool_size))
    rng = random.Random(seed)

    ranked_numbers = sorted(
        probability_by_number.items(),
        key=lambda item: (-item[1], item[0]),
    )
    candidate_pool = ranked_numbers[: min(top_pool_size, len(ranked_numbers))]
    if len(candidate_pool) < TICKET_SIZE:
        candidate_pool = ranked_numbers

    pool_numbers = [number for number, _ in candidate_pool]
    transformed = transform_weights(dict(candidate_pool), temperature)
    pool_weights = [transformed[number] for number in pool_numbers]

    unique_tickets: dict[tuple[int, ...], dict] = {}

    top_ticket = normalize_ticket([number for number, _ in ranked_numbers[:TICKET_SIZE]])
    unique_tickets[top_ticket] = {
        "numbers": top_ticket,
        "base_score": candidate_score(top_ticket, probability_by_number),
        "sum_probability": candidate_score(top_ticket, probability_by_number),
        "source": "top6",
    }

    max_attempts = max(num_candidates * 30, 200)
    attempts = 0
    while len(unique_tickets) < num_candidates and attempts < max_attempts:
        attempts += 1
        ticket = weighted_sample_without_replacement(
            rng,
            pool_numbers,
            pool_weights,
            k=TICKET_SIZE,
        )
        if ticket in unique_tickets:
            continue
        score = candidate_score(ticket, probability_by_number)
        unique_tickets[ticket] = {
            "numbers": ticket,
            "base_score": score,
            "sum_probability": score,
            "source": "sampled",
        }

    candidates = sorted(
        unique_tickets.values(),
        key=lambda item: (-item["base_score"], item["numbers"]),
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["candidate_rank"] = rank
    return candidates


def select_portfolio(
    candidates: list[dict],
    portfolio_size: int,
    *,
    overlap_penalty: float,
    unique_bonus: float,
) -> list[dict]:
    """Select a diversified portfolio from candidate tickets with a greedy score."""
    target_size = max(1, int(portfolio_size))
    if not candidates:
        return []

    selected: list[dict] = []
    covered_numbers: set[int] = set()
    remaining = [dict(candidate) for candidate in candidates]

    while remaining and len(selected) < target_size:
        best_index = 0
        best_value = float("-inf")
        best_candidate: dict | None = None

        for idx, candidate in enumerate(remaining):
            overlaps = [
                overlap_count(candidate["numbers"], existing["numbers"])
                for existing in selected
            ]
            overlap_cost = sum(value * value for value in overlaps)
            new_unique = len(set(candidate["numbers"]) - covered_numbers)
            adjusted_score = (
                float(candidate["base_score"])
                - float(overlap_penalty) * overlap_cost
                + float(unique_bonus) * new_unique
            )
            if adjusted_score > best_value:
                best_value = adjusted_score
                best_index = idx
                best_candidate = candidate

        assert best_candidate is not None
        picked = remaining.pop(best_index)
        picked["portfolio_rank"] = len(selected) + 1
        picked["selection_score"] = float(best_value)
        picked["new_unique_numbers"] = len(set(picked["numbers"]) - covered_numbers)
        picked["max_overlap_with_previous"] = (
            max(
                overlap_count(picked["numbers"], existing["numbers"])
                for existing in selected
            )
            if selected
            else 0
        )
        selected.append(picked)
        covered_numbers.update(picked["numbers"])

    return selected


def portfolio_overlap_summary(portfolio: list[dict]) -> dict:
    """Summarize overlap characteristics of a selected portfolio."""
    tickets = [candidate["numbers"] for candidate in portfolio]
    unique_numbers = sorted({number for ticket in tickets for number in ticket})
    overlaps = [overlap_count(left, right) for left, right in combinations(tickets, 2)]
    return {
        "ticket_count": len(tickets),
        "unique_number_count": len(unique_numbers),
        "unique_numbers": unique_numbers,
        "average_pairwise_overlap": float(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "max_pairwise_overlap": max(overlaps) if overlaps else 0,
    }


def classify_ticket(
    ticket: tuple[int, ...],
    winning_numbers: tuple[int, ...],
    bonus_number: int,
) -> tuple[int, bool, int | None]:
    """Return hit count, bonus match, and prize tier for a ticket."""
    hits = len(set(ticket).intersection(winning_numbers))
    bonus_hit = bonus_number > 0 and bonus_number in ticket
    tier = None
    if hits == 6:
        tier = 1
    elif hits == 5 and bonus_hit:
        tier = 2
    elif hits == 5:
        tier = 3
    elif hits == 4:
        tier = 4
    elif hits == 3:
        tier = 5
    return hits, bonus_hit, tier


def tier_payout(
    tier: int,
    draw_context: dict | None,
    tier_payouts: dict[int, float],
    tier1_estimate: float,
) -> float:
    """Return the payout amount used for the given tier."""
    if tier == 1:
        if draw_context:
            actual = parse_float(draw_context.get("first_prize_amount"))
            if actual > 0:
                return actual
        return float(tier1_estimate)
    return float(tier_payouts.get(tier, 0.0))


def evaluate_portfolio(
    portfolio: list[dict],
    winning_numbers: tuple[int, ...],
    bonus_number: int,
    *,
    draw_context: dict | None,
    tier_payouts: dict[int, float],
    ticket_price: float,
    tier1_estimate: float,
) -> dict:
    """Evaluate a selected portfolio against a winning draw."""
    counts = {tier: 0 for tier in range(1, 6)}
    payout_by_tier = {tier: 0.0 for tier in range(1, 6)}
    best_hits = 0
    total_hits = 0

    for candidate in portfolio:
        hits, _, tier = classify_ticket(candidate["numbers"], winning_numbers, bonus_number)
        best_hits = max(best_hits, hits)
        total_hits += hits
        candidate["actual_hits"] = hits
        candidate["actual_tier"] = tier if tier is not None else ""
        if tier is None:
            continue
        counts[tier] += 1
        payout_by_tier[tier] += tier_payout(
            tier,
            draw_context,
            tier_payouts,
            tier1_estimate,
        )

    total_payout = float(sum(payout_by_tier.values()))
    total_cost = float(len(portfolio)) * float(ticket_price)
    total_profit = total_payout - total_cost
    overlap_summary = portfolio_overlap_summary(portfolio)

    return {
        "best_hits": best_hits,
        "total_hits": total_hits,
        "tier_counts": counts,
        "payout_by_tier": payout_by_tier,
        "total_payout": total_payout,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "roi": (total_profit / total_cost) if total_cost > 0 else 0.0,
        "unique_number_count": overlap_summary["unique_number_count"],
        "average_pairwise_overlap": overlap_summary["average_pairwise_overlap"],
        "max_pairwise_overlap": overlap_summary["max_pairwise_overlap"],
        "ticket_count": overlap_summary["ticket_count"],
    }


def summarize_portfolio_results(results: list[dict], thresholds: list[int]) -> dict:
    """Aggregate per-draw portfolio backtest results into summary metrics."""
    if not results:
        return {
            "draws": 0,
            "portfolio_size": 0,
            "average_best_hits": 0.0,
            "best_hit_rates": {threshold: 0.0 for threshold in thresholds},
            "average_unique_numbers": 0.0,
            "average_pairwise_overlap": 0.0,
            "expected_ticket_count": 0.0,
            "expected_payout": 0.0,
            "expected_cost": 0.0,
            "expected_profit": 0.0,
            "expected_payout_by_tier": {tier: 0.0 for tier in range(1, 6)},
            "expected_tier_hits": {tier: 0.0 for tier in range(1, 6)},
            "tier_hit_rates": {tier: 0.0 for tier in range(1, 6)},
            "roi": 0.0,
        }

    draws = len(results)
    best_hits = [int(result["best_hits"]) for result in results]
    unique_counts = [int(result["unique_number_count"]) for result in results]
    avg_overlaps = [float(result["average_pairwise_overlap"]) for result in results]
    ticket_counts = [int(result["ticket_count"]) for result in results]

    total_cost = float(sum(float(result["total_cost"]) for result in results))
    total_payout = float(sum(float(result["total_payout"]) for result in results))
    total_profit = float(sum(float(result["total_profit"]) for result in results))

    expected_payout_by_tier = {
        tier: float(sum(float(result["payout_by_tier"][tier]) for result in results) / draws)
        for tier in range(1, 6)
    }
    expected_tier_hits = {
        tier: float(sum(int(result["tier_counts"][tier]) for result in results) / draws)
        for tier in range(1, 6)
    }
    tier_hit_rates = {
        tier: float(sum(1 for result in results if int(result["tier_counts"][tier]) > 0) / draws)
        for tier in range(1, 6)
    }

    return {
        "draws": draws,
        "portfolio_size": int(round(sum(ticket_counts) / draws)),
        "average_best_hits": float(sum(best_hits) / draws),
        "best_hit_rates": {
            threshold: float(sum(1 for value in best_hits if value >= threshold) / draws)
            for threshold in thresholds
        },
        "average_unique_numbers": float(sum(unique_counts) / draws),
        "average_pairwise_overlap": float(sum(avg_overlaps) / draws),
        "expected_ticket_count": float(sum(ticket_counts) / draws),
        "expected_payout": total_payout / draws,
        "expected_cost": total_cost / draws,
        "expected_profit": total_profit / draws,
        "expected_payout_by_tier": expected_payout_by_tier,
        "expected_tier_hits": expected_tier_hits,
        "tier_hit_rates": tier_hit_rates,
        "roi": (total_profit / total_cost) if total_cost > 0 else 0.0,
    }


def build_portfolio(
    probability_by_number: dict[int, float],
    *,
    num_candidates: int,
    portfolio_size: int,
    top_pool_size: int,
    temperature: float,
    overlap_penalty: float,
    unique_bonus: float,
    seed: int,
) -> dict:
    """Generate candidate tickets and select a final diversified portfolio."""
    candidates = generate_candidates(
        probability_by_number,
        num_candidates,
        top_pool_size=top_pool_size,
        temperature=temperature,
        seed=seed,
    )
    portfolio = select_portfolio(
        candidates,
        portfolio_size,
        overlap_penalty=overlap_penalty,
        unique_bonus=unique_bonus,
    )
    summary = portfolio_overlap_summary(portfolio)
    return {
        "candidates": candidates,
        "portfolio": portfolio,
        "summary": summary,
    }
