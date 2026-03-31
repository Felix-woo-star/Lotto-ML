#!/usr/bin/env python3
"""Print a concise terminal summary for pipeline outputs."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print lotto pipeline summary.")
    parser.add_argument(
        "--baseline-json",
        default="reports/baseline.json",
        help="Baseline metrics JSON path.",
    )
    parser.add_argument(
        "--model-jsons",
        default="reports/model.json",
        help="Comma-separated model metrics JSON paths.",
    )
    parser.add_argument(
        "--rolling-json",
        default="reports/rolling_validation.json",
        help="Rolling validation JSON path.",
    )
    parser.add_argument(
        "--predictions-json",
        default="reports/predictions.json",
        help="Predictions JSON path.",
    )
    parser.add_argument(
        "--portfolio-ticket-limit",
        type=int,
        default=3,
        help="Number of portfolio tickets to print.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0.0))


def hit_rate(metrics: dict, threshold: int) -> float:
    hit_rates = metrics.get("hit_rates", {})
    value = hit_rates.get(threshold)
    if value is None:
        value = hit_rates.get(str(threshold), 0.0)
    return float(value)


def portfolio_value(metrics: dict, key: str) -> float:
    portfolio = metrics.get("portfolio", {})
    return float(portfolio.get(key, 0.0))


def summary_metric(summary: dict, key: int | str) -> dict:
    value = summary.get(key)
    if value is None:
        value = summary.get(str(key), {})
    return value or {}


def composite_score(metrics: dict, thresholds: list[int]) -> float:
    score = metric_value(metrics, "average_hits")
    weight_by_threshold = {1: 0.03, 2: 0.07, 3: 0.20, 4: 0.05, 5: 0.02}
    for threshold in thresholds:
        score += hit_rate(metrics, threshold) * weight_by_threshold.get(threshold, 0.01)
    score += metric_value(metrics, "mrr") * 0.15
    mean_min_rank = metric_value(metrics, "mean_min_rank")
    if mean_min_rank > 0:
        score += (1.0 / mean_min_rank) * 0.40
    score -= metric_value(metrics, "brier") * 0.10
    score -= metric_value(metrics, "log_loss") * 0.05
    score -= metric_value(metrics, "ece") * 0.10
    if "portfolio" in metrics:
        score += portfolio_value(metrics, "average_best_hits") * 0.25
        score += portfolio_value(metrics, "roi") * 0.02
    return float(score)


def collect_thresholds(payloads: list[dict]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for payload in payloads:
        if not payload:
            continue
        config = payload.get("config", {})
        raw = str(config.get("hit_thresholds", ""))
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            value = int(item)
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        metrics = payload.get("metrics", payload)
        for key in metrics.get("hit_rates", {}).keys():
            try:
                value = int(key)
            except (TypeError, ValueError):
                continue
            if value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered or [1, 2, 3, 4, 5]


def print_baseline_summary(payload: dict | None, thresholds: list[int]) -> None:
    if not payload:
        return
    metrics = payload.get("metrics", {}).get("baseline", {})
    if not metrics:
        return
    print("Baseline")
    print(f"- average_hits: {metric_value(metrics, 'average_hits'):.4f}")
    if "portfolio" in metrics:
        print(f"- portfolio_best_hits: {portfolio_value(metrics, 'average_best_hits'):.4f}")
        print(f"- portfolio_roi: {portfolio_value(metrics, 'roi'):.6f}")
    else:
        for threshold in thresholds[:3]:
            print(f"- hit_rate_{threshold}: {hit_rate(metrics, threshold):.4f}")


def print_model_summary(payloads: list[dict], thresholds: list[int]) -> None:
    ranked: list[tuple[str, dict, float]] = []
    for payload in payloads:
        if not payload:
            continue
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        if not metrics:
            continue
        label = config.get("model") or os.path.basename(config.get("data_path", "")) or "model"
        ranked.append((label, metrics, composite_score(metrics, thresholds)))

    if not ranked:
        return

    ranked.sort(key=lambda item: item[2], reverse=True)
    best_name, best_metrics, best_score = ranked[0]
    print("Best Model")
    print(f"- model: {best_name}")
    print(f"- score: {best_score:.4f}")
    print(f"- average_hits: {metric_value(best_metrics, 'average_hits'):.4f}")
    print(f"- mrr: {metric_value(best_metrics, 'mrr'):.4f}")
    if "portfolio" in best_metrics:
        print(f"- portfolio_best_hits: {portfolio_value(best_metrics, 'average_best_hits'):.4f}")
        print(f"- portfolio_profit: {portfolio_value(best_metrics, 'expected_profit'):.2f}")
        print(f"- portfolio_roi: {portfolio_value(best_metrics, 'roi'):.6f}")


def print_rolling_summary(payload: dict | None) -> None:
    if not payload:
        return
    summary = payload.get("summary", {})
    if not summary:
        return
    thresholds = payload.get("config", {}).get("hit_thresholds", [1, 2, 3, 4, 5])
    ranked = sorted(
        summary.items(),
        key=lambda item: composite_score(
            {
                "average_hits": item[1]["average_hits"]["mean"],
                "hit_rates": {
                    threshold: summary_metric(item[1]["hit_rates"], threshold).get("mean", 0.0)
                    for threshold in thresholds
                },
                "mrr": item[1].get("mrr", {}).get("mean", 0.0),
                "mean_min_rank": item[1].get("mean_min_rank", {}).get("mean", 0.0),
                "brier": item[1].get("brier", {}).get("mean", 0.0),
                "log_loss": item[1].get("log_loss", {}).get("mean", 0.0),
                "ece": item[1].get("ece", {}).get("mean", 0.0),
                "portfolio": {
                    "average_best_hits": item[1].get("portfolio", {})
                    .get("average_best_hits", {})
                    .get("mean", 0.0),
                    "roi": item[1].get("portfolio", {}).get("roi", {}).get("mean", 0.0),
                },
            },
            thresholds,
        ),
        reverse=True,
    )
    if not ranked:
        return
    model_name, metrics = ranked[0]
    print("Rolling Validation")
    print(f"- winner: {model_name}")
    print(f"- average_hits_mean: {metrics['average_hits']['mean']:.4f}")
    portfolio = metrics.get("portfolio", {})
    if portfolio:
        print(f"- portfolio_best_hits_mean: {portfolio['average_best_hits']['mean']:.4f}")
        print(f"- portfolio_roi_mean: {portfolio['roi']['mean']:.6f}")


def print_predictions_summary(payload: dict | None, ticket_limit: int) -> None:
    if not payload:
        return
    recommendations = payload.get("recommendations", [])
    portfolio = payload.get("portfolio", [])
    portfolio_summary = payload.get("portfolio_summary", {})
    print("Next Draw Prediction")
    print(f"- latest_draw: {payload.get('latest_draw', '')}")
    print(f"- predicted_draw: {payload.get('predicted_draw', '')}")
    print(
        "- top_numbers: "
        + ", ".join(str(item["number"]) for item in recommendations[: payload.get("top_k", 6)])
    )
    if portfolio_summary:
        print(
            f"- portfolio_unique_numbers: "
            f"{portfolio_summary.get('unique_number_count', 0)}"
        )
        print(
            f"- portfolio_avg_overlap: "
            f"{float(portfolio_summary.get('average_pairwise_overlap', 0.0)):.4f}"
        )
    for candidate in portfolio[: max(0, ticket_limit)]:
        numbers = ", ".join(str(number) for number in candidate.get("numbers", []))
        print(f"- ticket_{candidate.get('ticket_rank', '?')}: {numbers}")


def main() -> int:
    args = parse_args()
    model_paths = [item.strip() for item in args.model_jsons.split(",") if item.strip()]

    baseline_payload = load_json(args.baseline_json)
    model_payloads = [load_json(path) for path in model_paths]
    rolling_payload = load_json(args.rolling_json)
    predictions_payload = load_json(args.predictions_json)
    thresholds = collect_thresholds(
        [payload for payload in [baseline_payload, *model_payloads] if payload]
    )

    print()
    print("==> 실행 요약")
    print_baseline_summary(baseline_payload, thresholds)
    print_model_summary([payload for payload in model_payloads if payload], thresholds)
    print_rolling_summary(rolling_payload)
    print_predictions_summary(predictions_payload, args.portfolio_ticket_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
