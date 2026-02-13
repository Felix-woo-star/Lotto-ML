#!/usr/bin/env python3
"""베이스라인 추천 모델을 평가하는 스크립트."""
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass

NUMBER_MIN = 1
NUMBER_MAX = 45


@dataclass(frozen=True)
class Draw:
    draw_no: int
    numbers: tuple[int, int, int, int, int, int]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Evaluate baseline lotto model.")
    parser.add_argument(
        "--in-processed",
        default="data/processed/lotto_draws.csv",
        help="정제된 CSV 경로.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="테스트로 사용할 마지막 회차 수.",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=None,
        help="훈련에 사용할 마지막 회차 번호. 지정 시 test-size는 무시된다.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100,
        help="회차당 추천 조합 개수 K(max_of_candidates 모드에서 사용).",
    )
    parser.add_argument(
        "--eval-protocol",
        choices=["single_top6", "max_of_candidates"],
        default="single_top6",
        help=(
            "평가 방식. single_top6: 회차당 추천 1세트(모델과 동일 기준), "
            "max_of_candidates: K개 중 최대 적중."
        ),
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=None,
        help="최근 N회차만 사용해 빈도를 계산한다.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1.0,
        help="회차 간 가중치 감소율(1.0이면 동일 가중치).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="빈도 0 방지를 위한 스무딩 값.",
    )
    parser.add_argument(
        "--hit-thresholds",
        default="1,2,3,4,5",
        help="Top-K 적중 기준(쉼표 구분).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="난수 시드.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="평가 결과 JSON 저장 경로.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="평가 결과 CSV 저장 경로.",
    )
    return parser.parse_args()


def load_draws(path: str) -> list[Draw]:
    """정제된 CSV를 로드한다."""
    draws: list[Draw] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numbers = tuple(int(row[f"n{i}"]) for i in range(1, 7))
            draws.append(Draw(draw_no=int(row["draw_no"]), numbers=numbers))
    return draws


def split_draws(
    draws: list[Draw], train_end: int | None, test_size: int
) -> tuple[list[Draw], list[Draw]]:
    """시간 순으로 훈련/테스트를 분리한다."""
    if not draws:
        return [], []

    draws = sorted(draws, key=lambda item: item.draw_no)
    if train_end is not None:
        train = [draw for draw in draws if draw.draw_no <= train_end]
        test = [draw for draw in draws if draw.draw_no > train_end]
        return train, test

    if test_size <= 0:
        return draws, []
    split_index = max(len(draws) - test_size, 0)
    return draws[:split_index], draws[split_index:]


def compute_weights(
    draws: list[Draw], recent_window: int | None, decay: float, smoothing: float
) -> list[float]:
    """훈련 구간의 숫자 빈도로 가중치를 계산한다."""
    if recent_window is not None and recent_window > 0:
        draws = draws[-recent_window:]

    weights = [float(smoothing) for _ in range(NUMBER_MAX - NUMBER_MIN + 1)]
    if not draws:
        return weights

    if decay <= 0 or decay > 1:
        decay = 1.0

    last_index = len(draws) - 1
    for idx, draw in enumerate(draws):
        distance = last_index - idx
        weight = math.pow(decay, distance)
        for number in draw.numbers:
            weights[number - NUMBER_MIN] += weight
    return weights


def weighted_sample(
    rng: random.Random, numbers: list[int], weights: list[float], k: int
) -> list[int]:
    """가중치 기반으로 중복 없이 k개를 샘플링한다."""
    pool_numbers = numbers[:]
    pool_weights = weights[:]
    picks: list[int] = []

    for _ in range(k):
        total = sum(pool_weights)
        if total <= 0:
            choice = rng.choice(pool_numbers)
            picks.append(choice)
            index = pool_numbers.index(choice)
        else:
            target = rng.random() * total
            cumulative = 0.0
            index = 0
            for index, weight in enumerate(pool_weights):
                cumulative += weight
                if cumulative >= target:
                    break
            picks.append(pool_numbers[index])
        pool_numbers.pop(index)
        pool_weights.pop(index)

    return sorted(picks)


def generate_candidates(
    rng: random.Random, weights: list[float], num_candidates: int
) -> list[tuple[int, int, int, int, int, int]]:
    """추천 조합 K개를 생성한다."""
    numbers = list(range(NUMBER_MIN, NUMBER_MAX + 1))
    candidates = []
    for _ in range(num_candidates):
        picks = weighted_sample(rng, numbers, weights, 6)
        candidates.append(tuple(picks))
    return candidates


def top_numbers_from_weights(weights: list[float], k: int = 6) -> tuple[int, ...]:
    """가중치 상위 k개 숫자를 정렬해 반환한다."""
    scored = list(enumerate(weights, start=NUMBER_MIN))
    scored.sort(key=lambda item: (-item[1], item[0]))
    picks = [number for number, _ in scored[:k]]
    return tuple(sorted(picks))


def hits(target: tuple[int, int, int, int, int, int], candidate: tuple[int, ...]) -> int:
    """두 조합의 일치 개수를 계산한다."""
    return len(set(target).intersection(candidate))


def evaluate_max_of_candidates(
    test_draws: list[Draw],
    weights: list[float],
    num_candidates: int,
    thresholds: list[int],
    seed: int,
) -> dict:
    """K개 추천 조합 중 최대 적중을 사용하는 평가를 수행한다."""
    rng = random.Random(seed)
    max_hits: list[int] = []
    for draw in test_draws:
        candidates = generate_candidates(rng, weights, num_candidates)
        draw_hits = [hits(draw.numbers, candidate) for candidate in candidates]
        max_hits.append(max(draw_hits) if draw_hits else 0)

    if not max_hits:
        return {
            "average_hits": 0.0,
            "hit_rates": {threshold: 0.0 for threshold in thresholds},
            "draws": 0,
        }

    average_hits = sum(max_hits) / len(max_hits)
    hit_rates = {
        threshold: sum(1 for value in max_hits if value >= threshold) / len(max_hits)
        for threshold in thresholds
    }
    return {"average_hits": average_hits, "hit_rates": hit_rates, "draws": len(max_hits)}


def evaluate_single_top6(
    test_draws: list[Draw],
    weights: list[float],
    thresholds: list[int],
    *,
    random_pick: bool,
    seed: int,
) -> dict:
    """회차당 추천 1세트 기준으로 평가한다."""
    if not test_draws:
        return {
            "average_hits": 0.0,
            "hit_rates": {threshold: 0.0 for threshold in thresholds},
            "draws": 0,
        }

    rng = random.Random(seed)
    numbers = list(range(NUMBER_MIN, NUMBER_MAX + 1))
    fixed_candidate = top_numbers_from_weights(weights, k=6)

    result_hits: list[int] = []
    for draw in test_draws:
        candidate = (
            tuple(weighted_sample(rng, numbers, weights, 6))
            if random_pick
            else fixed_candidate
        )
        result_hits.append(hits(draw.numbers, candidate))

    average_hits = sum(result_hits) / len(result_hits)
    hit_rates = {
        threshold: sum(1 for value in result_hits if value >= threshold) / len(result_hits)
        for threshold in thresholds
    }
    return {
        "average_hits": average_hits,
        "hit_rates": hit_rates,
        "draws": len(result_hits),
    }


def improvement(baseline: float, random_value: float) -> float:
    """랜덤 대비 개선율을 계산한다."""
    if random_value == 0:
        return 0.0
    return (baseline - random_value) / random_value


def draw_range(draws: list[Draw]) -> tuple[int, int] | None:
    """회차 번호 범위를 반환한다."""
    if not draws:
        return None
    return draws[0].draw_no, draws[-1].draw_no


def metrics_to_flat(metrics: dict, thresholds: list[int]) -> dict:
    """메트릭을 평탄화된 딕셔너리로 변환한다."""
    flattened = {"average_hits": metrics["average_hits"]}
    for threshold in thresholds:
        flattened[f"hit_rate_{threshold}"] = metrics["hit_rates"][threshold]
    return flattened


def build_config(
    args: argparse.Namespace,
    thresholds: list[int],
    train: list[Draw],
    test: list[Draw],
) -> dict:
    """평가 설정 정보를 구성한다."""
    train_span = draw_range(train)
    test_span = draw_range(test)
    return {
        "data_path": args.in_processed,
        "train_draws": len(train),
        "test_draws": len(test),
        "train_span": train_span,
        "test_span": test_span,
        "train_end": args.train_end if args.train_end is not None else "",
        "test_size_param": args.test_size,
        "eval_protocol": args.eval_protocol,
        "num_candidates": args.num_candidates,
        "recent_window": args.recent_window if args.recent_window is not None else "",
        "decay": args.decay,
        "smoothing": args.smoothing,
        "seed": args.seed,
        "hit_thresholds": ",".join(str(value) for value in thresholds),
    }


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전 상위 디렉터리를 만든다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    """JSON 파일로 평가 결과를 저장한다."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """CSV 파일로 평가 결과를 저장한다."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    draws = load_draws(args.in_processed)
    train, test = split_draws(draws, args.train_end, args.test_size)
    if not train or not test:
        print("훈련/테스트 데이터가 부족합니다.")
        return 1

    thresholds = [
        int(value.strip()) for value in args.hit_thresholds.split(",") if value.strip()
    ]
    weights = compute_weights(train, args.recent_window, args.decay, args.smoothing)
    uniform_weights = [1.0 for _ in range(NUMBER_MAX - NUMBER_MIN + 1)]
    if args.eval_protocol == "single_top6":
        baseline = evaluate_single_top6(
            test, weights, thresholds, random_pick=False, seed=args.seed
        )
        random_baseline = evaluate_single_top6(
            test, uniform_weights, thresholds, random_pick=True, seed=args.seed + 1
        )
    else:
        baseline = evaluate_max_of_candidates(
            test, weights, args.num_candidates, thresholds, seed=args.seed
        )
        random_baseline = evaluate_max_of_candidates(
            test, uniform_weights, args.num_candidates, thresholds, seed=args.seed + 1
        )
    improvement_metrics = {
        "average_hits": improvement(
            baseline["average_hits"], random_baseline["average_hits"]
        ),
        "hit_rates": {
            threshold: improvement(
                baseline["hit_rates"][threshold], random_baseline["hit_rates"][threshold]
            )
            for threshold in thresholds
        },
    }

    print("Baseline metrics")
    print(f"- average_hits: {baseline['average_hits']:.4f}")
    for threshold in thresholds:
        print(f"- hit_rate_{threshold}: {baseline['hit_rates'][threshold]:.4f}")

    print("Random baseline")
    print(f"- average_hits: {random_baseline['average_hits']:.4f}")
    for threshold in thresholds:
        print(f"- hit_rate_{threshold}: {random_baseline['hit_rates'][threshold]:.4f}")

    print("Improvement vs random")
    print(
        f"- average_hits: {improvement(baseline['average_hits'], random_baseline['average_hits']):.4f}"
    )
    for threshold in thresholds:
        baseline_rate = baseline["hit_rates"][threshold]
        random_rate = random_baseline["hit_rates"][threshold]
        print(
            f"- hit_rate_{threshold}: {improvement(baseline_rate, random_rate):.4f}"
        )

    config = build_config(args, thresholds, train, test)
    if args.out_json:
        payload = {
            "config": config,
            "metrics": {
                "baseline": baseline,
                "random": random_baseline,
                "improvement": improvement_metrics,
            },
        }
        write_json(args.out_json, payload)
        print(f"Saved JSON metrics to {args.out_json}")

    if args.out_csv:
        base_fields = ["metric_type", "average_hits"] + [
            f"hit_rate_{threshold}" for threshold in thresholds
        ]
        config_fields = list(config.keys())
        fieldnames = base_fields + config_fields

        rows = []
        for metric_type, metrics in (
            ("baseline", baseline),
            ("random", random_baseline),
            ("improvement", improvement_metrics),
        ):
            row = {"metric_type": metric_type}
            row.update(metrics_to_flat(metrics, thresholds))
            row.update(config)
            rows.append(row)

        write_csv(args.out_csv, rows, fieldnames)
        print(f"Saved CSV metrics to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
