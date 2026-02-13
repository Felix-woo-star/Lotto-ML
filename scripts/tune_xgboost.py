#!/usr/bin/env python3
"""XGBoost 하이퍼파라미터를 롤링 검증으로 탐색한다."""

import argparse
import itertools
import json
import os

import pandas as pd
import train_model as tm


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters.")
    parser.add_argument(
        "--in-parquet",
        default="data/features/lotto_features.parquet",
        help="피처 Parquet 경로.",
    )
    parser.add_argument(
        "--hit-thresholds",
        default="1,2,3,4,5",
        help="Top-K 적중 기준(쉼표 구분).",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드.")
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="ECE 계산에 사용할 confidence bin 개수.",
    )
    parser.add_argument(
        "--min-train-draws",
        type=int,
        default=900,
        help="첫 폴드 최소 학습 회차 수.",
    )
    parser.add_argument(
        "--fold-test-size",
        type=int,
        default=20,
        help="각 폴드 테스트 회차 수.",
    )
    parser.add_argument(
        "--fold-step-size",
        type=int,
        default=20,
        help="폴드 시작점 이동 간격.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=4,
        help="최근 기준 최대 폴드 수.",
    )
    parser.add_argument(
        "--grid-n-estimators",
        default="200,400,600",
        help="n_estimators 탐색 값(쉼표 구분).",
    )
    parser.add_argument(
        "--grid-max-depth",
        default="4,6,8",
        help="max_depth 탐색 값(쉼표 구분).",
    )
    parser.add_argument(
        "--grid-learning-rate",
        default="0.03,0.05,0.1",
        help="learning_rate 탐색 값(쉼표 구분).",
    )
    parser.add_argument(
        "--grid-subsample",
        default="0.7,0.85,1.0",
        help="subsample 탐색 값(쉼표 구분).",
    )
    parser.add_argument(
        "--grid-colsample-bytree",
        default="0.7,0.85,1.0",
        help="colsample_bytree 탐색 값(쉼표 구분).",
    )
    parser.add_argument(
        "--out-json",
        default="reports/tuning_xgboost.json",
        help="튜닝 결과 JSON 저장 경로.",
    )
    parser.add_argument(
        "--out-md",
        default="reports/tuning_xgboost.md",
        help="튜닝 결과 Markdown 저장 경로.",
    )
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    """쉼표 구분 정수 문자열을 리스트로 변환한다."""
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_floats(raw: str) -> list[float]:
    """쉼표 구분 실수 문자열을 리스트로 변환한다."""
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_thresholds(raw: str) -> list[int]:
    """적중 임계값 문자열을 리스트로 변환한다."""
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전 상위 디렉터리를 만든다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def build_folds(draws: list[int], args: argparse.Namespace) -> list[dict]:
    """확장 윈도우 롤링 폴드 목록을 만든다."""
    if len(draws) < args.min_train_draws + args.fold_test_size:
        return []

    starts = list(
        range(
            args.min_train_draws,
            len(draws) - args.fold_test_size + 1,
            max(1, args.fold_step_size),
        )
    )
    if args.max_folds > 0 and len(starts) > args.max_folds:
        starts = starts[-args.max_folds:]

    folds = []
    for fold_idx, start_idx in enumerate(starts, start=1):
        train_draws = draws[:start_idx]
        test_draws = draws[start_idx : start_idx + args.fold_test_size]
        folds.append(
            {
                "fold": fold_idx,
                "train_draws": train_draws,
                "test_draws": test_draws,
                "train_span": (train_draws[0], train_draws[-1]),
                "test_span": (test_draws[0], test_draws[-1]),
            }
        )
    return folds


def avg(values: list[float]) -> float:
    """평균값을 반환한다."""
    return sum(values) / len(values) if values else 0.0


def score_metrics(metrics: dict, thresholds: list[int]) -> float:
    """모델 우열 판정을 위한 종합 점수를 계산한다."""
    score = float(metrics.get("average_hits", 0.0))
    hit_rates = metrics.get("hit_rates", {})
    weight_by_threshold = {1: 0.03, 2: 0.07, 3: 0.20, 4: 0.05, 5: 0.02}
    for threshold in thresholds:
        value = hit_rates.get(threshold)
        if value is None:
            value = hit_rates.get(str(threshold), 0.0)
        score += float(value) * weight_by_threshold.get(threshold, 0.01)
    score += float(metrics.get("mrr", 0.0)) * 0.15
    mean_min_rank = float(metrics.get("mean_min_rank", 0.0))
    if mean_min_rank > 0:
        score += (1.0 / mean_min_rank) * 0.40
    score -= float(metrics.get("brier", 0.0)) * 0.10
    score -= float(metrics.get("log_loss", 0.0)) * 0.05
    score -= float(metrics.get("ece", 0.0)) * 0.10
    return score


def evaluate_combo(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[dict],
    thresholds: list[int],
    args: argparse.Namespace,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> dict:
    """단일 하이퍼파라미터 조합을 롤링 폴드로 평가한다."""
    fold_metrics = []
    for fold in folds:
        train_df = df[df["draw_no"].isin(fold["train_draws"])].copy()
        test_df = df[df["draw_no"].isin(fold["test_draws"])].copy()

        model_args = argparse.Namespace(
            model="xgboost",
            seed=args.seed,
            gbdt_max_iter=200,
            gbdt_learning_rate=0.1,
            gbdt_max_depth=3,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            min_data_in_leaf=20,
            xgb_subsample=subsample,
            xgb_colsample_bytree=colsample_bytree,
            mlp_hidden_layers="64,32",
            mlp_max_iter=400,
        )

        model = tm.build_model(model_args)
        model.fit(train_df[feature_cols], train_df["label"])
        metrics = tm.evaluate(
            model,
            test_df,
            feature_cols,
            thresholds,
            calibration_bins=args.calibration_bins,
        )
        fold_metrics.append(metrics)

    avg_metrics = {
        "average_hits": avg([float(item["average_hits"]) for item in fold_metrics]),
        "hit_rates": {
            threshold: avg([float(item["hit_rates"][threshold]) for item in fold_metrics])
            for threshold in thresholds
        },
        "mrr": avg([float(item.get("mrr", 0.0)) for item in fold_metrics]),
        "mean_min_rank": avg(
            [float(item.get("mean_min_rank", 0.0)) for item in fold_metrics]
        ),
        "brier": avg([float(item.get("brier", 0.0)) for item in fold_metrics]),
        "log_loss": avg([float(item.get("log_loss", 0.0)) for item in fold_metrics]),
        "ece": avg([float(item.get("ece", 0.0)) for item in fold_metrics]),
        "folds": len(fold_metrics),
    }

    return {
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        },
        "metrics": avg_metrics,
        "score": score_metrics(avg_metrics, thresholds),
    }


def render_markdown(payload: dict, thresholds: list[int]) -> str:
    """튜닝 결과를 Markdown으로 렌더링한다."""
    config = payload["config"]
    best = payload["best"]
    trials = payload["trials"]

    lines = [
        "# XGBoost 튜닝 리포트",
        "",
        "## 설정",
        f"- 데이터: `{config['data_path']}`",
        f"- 폴드 수: {config['folds']}",
        f"- 탐색 조합 수: {config['trial_count']}",
        f"- 탐색 파라미터: n_estimators, max_depth, learning_rate, subsample, colsample_bytree",
        "",
        "## 최적 파라미터",
        f"- n_estimators: {best['params']['n_estimators']}",
        f"- max_depth: {best['params']['max_depth']}",
        f"- learning_rate: {best['params']['learning_rate']}",
        f"- subsample: {best['params']['subsample']}",
        f"- colsample_bytree: {best['params']['colsample_bytree']}",
        f"- average_hits: {best['metrics']['average_hits']:.4f}",
        f"- MRR: {best['metrics']['mrr']:.4f}",
        f"- Brier: {best['metrics']['brier']:.6f}",
        f"- ECE: {best['metrics']['ece']:.6f}",
        f"- 종합점수: {best['score']:.4f}",
    ]
    for threshold in thresholds:
        lines.append(f"- Hit@{threshold}: {best['metrics']['hit_rates'][threshold]:.4f}")

    lines.extend(["", "## 상위 결과", ""])
    headers = (
        ["순위", "종합점수", "average_hits"]
        + [f"Hit@{t}" for t in thresholds]
        + [
            "MRR",
            "Brier",
            "ECE",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        ]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, trial in enumerate(trials[:10], start=1):
        row = [
            str(idx),
            f"{trial['score']:.4f}",
            f"{trial['metrics']['average_hits']:.4f}",
        ]
        for threshold in thresholds:
            row.append(f"{trial['metrics']['hit_rates'][threshold]:.4f}")
        row.extend(
            [
                f"{trial['metrics']['mrr']:.4f}",
                f"{trial['metrics']['brier']:.6f}",
                f"{trial['metrics']['ece']:.6f}",
                str(trial["params"]["n_estimators"]),
                str(trial["params"]["max_depth"]),
                str(trial["params"]["learning_rate"]),
                str(trial["params"]["subsample"]),
                str(trial["params"]["colsample_bytree"]),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.in_parquet):
        print(f"Missing input file: {args.in_parquet}")
        return 1

    thresholds = parse_thresholds(args.hit_thresholds)
    df = pd.read_parquet(args.in_parquet).fillna(0)
    feature_cols = [col for col in df.columns if col not in ("label", "draw_no")]

    draws = sorted(int(value) for value in df["draw_no"].unique())
    folds = build_folds(draws, args)
    if not folds:
        print("튜닝용 폴드를 만들 수 없습니다. min-train-draws/fold-test-size를 조정하세요.")
        return 1

    grid_n_estimators = parse_ints(args.grid_n_estimators)
    grid_max_depth = parse_ints(args.grid_max_depth)
    grid_learning_rate = parse_floats(args.grid_learning_rate)
    grid_subsample = parse_floats(args.grid_subsample)
    grid_colsample_bytree = parse_floats(args.grid_colsample_bytree)

    combos = list(
        itertools.product(
            grid_n_estimators,
            grid_max_depth,
            grid_learning_rate,
            grid_subsample,
            grid_colsample_bytree,
        )
    )
    print(f"총 탐색 조합 수: {len(combos)}")

    trials = []
    for idx, (
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
    ) in enumerate(combos, start=1):
        try:
            result = evaluate_combo(
                df,
                feature_cols,
                folds,
                thresholds,
                args,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
            )
        except tm.OptionalDependencyError as exc:
            print(str(exc))
            return 2
        trials.append(result)
        print(
            f"[{idx}/{len(combos)}] score={result['score']:.4f}, "
            f"avg_hits={result['metrics']['average_hits']:.4f}, params={result['params']}"
        )

    trials.sort(key=lambda item: float(item["score"]), reverse=True)
    best = trials[0]

    payload = {
        "config": {
            "data_path": args.in_parquet,
            "folds": len(folds),
            "fold_spans": [
                {
                    "fold": fold["fold"],
                    "train_span": fold["train_span"],
                    "test_span": fold["test_span"],
                }
                for fold in folds
            ],
            "trial_count": len(trials),
            "thresholds": thresholds,
            "calibration_bins": args.calibration_bins,
            "grid": {
                "n_estimators": grid_n_estimators,
                "max_depth": grid_max_depth,
                "learning_rate": grid_learning_rate,
                "subsample": grid_subsample,
                "colsample_bytree": grid_colsample_bytree,
            },
        },
        "best": best,
        "trials": trials,
    }

    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(tm.to_json_compatible(payload), handle, ensure_ascii=False, indent=2)

    md_text = render_markdown(payload, thresholds)
    ensure_parent_dir(args.out_md)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write(md_text)

    print(f"최적 파라미터: {best['params']}")
    print(f"튜닝 JSON 저장: {args.out_json}")
    print(f"튜닝 리포트 저장: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
