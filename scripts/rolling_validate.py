#!/usr/bin/env python3
"""여러 모델을 시간축 롤링 백테스트로 검증한다."""

import argparse
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import lotto_portfolio as lp
import train_model as tm

DEFAULT_MODELS = [
    "logreg",
    "gbdt",
    "randomforest",
    "extratrees",
    "mlp",
    "lightgbm",
    "xgboost",
    "catboost",
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Run rolling backtest for lotto models.")
    parser.add_argument(
        "--in-parquet",
        default="data/features/lotto_features.parquet",
        help="피처 Parquet 경로.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="검증할 모델 목록(쉼표 구분).",
    )
    parser.add_argument(
        "--hit-thresholds",
        default="1,2,3,4,5",
        help="Top-K 적중 기준(쉼표 구분).",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="ECE 계산에 사용할 confidence bin 개수.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="난수 시드.",
    )
    parser.add_argument(
        "--in-processed",
        default="data/processed/lotto_draws.csv",
        help="보너스 번호/수익 평가용 정제 CSV 경로.",
    )
    parser.add_argument(
        "--min-train-draws",
        type=int,
        default=600,
        help="첫 폴드 최소 학습 회차 수.",
    )
    parser.add_argument(
        "--fold-test-size",
        type=int,
        default=40,
        help="각 폴드 테스트 회차 수.",
    )
    parser.add_argument(
        "--fold-step-size",
        type=int,
        default=20,
        help="폴드 시작점 이동 간격(회차).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=8,
        help="최근 기준 최대 폴드 수.",
    )

    parser.add_argument("--gbdt-max-iter", type=int, default=200)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.1)
    parser.add_argument("--gbdt-max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=None,
        help="LightGBM 전용 n_estimators (미지정 시 --n-estimators 사용).",
    )
    parser.add_argument(
        "--lgbm-max-depth",
        type=int,
        default=None,
        help="LightGBM 전용 max_depth (미지정 시 --max-depth 사용).",
    )
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--mlp-hidden-layers", default="64,32")
    parser.add_argument("--mlp-max-iter", type=int, default=1000)
    parser.add_argument("--mlp-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--mlp-tol", type=float, default=1e-3)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.1)
    parser.add_argument("--mlp-n-iter-no-change", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--portfolio-size", type=int, default=12)
    parser.add_argument("--candidate-pool-size", type=int, default=18)
    parser.add_argument("--sampling-temperature", type=float, default=0.9)
    parser.add_argument("--overlap-penalty", type=float, default=0.18)
    parser.add_argument("--unique-bonus", type=float, default=0.035)
    parser.add_argument("--ticket-price", type=float, default=lp.DEFAULT_TICKET_PRICE)
    parser.add_argument(
        "--tier1-payout-estimate",
        type=float,
        default=lp.DEFAULT_TIER1_ESTIMATE,
    )
    parser.add_argument("--tier2-payout", type=float, default=lp.DEFAULT_TIER_PAYOUTS[2])
    parser.add_argument("--tier3-payout", type=float, default=lp.DEFAULT_TIER_PAYOUTS[3])
    parser.add_argument("--tier4-payout", type=float, default=lp.DEFAULT_TIER_PAYOUTS[4])
    parser.add_argument("--tier5-payout", type=float, default=lp.DEFAULT_TIER_PAYOUTS[5])
    parser.add_argument(
        "--mlp-early-stopping",
        dest="mlp_early_stopping",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-mlp-early-stopping",
        dest="mlp_early_stopping",
        action="store_false",
    )
    parser.add_argument(
        "--include-ensemble",
        dest="include_ensemble",
        action="store_true",
        default=True,
        help="폴드별 모델 평균 확률 앙상블을 추가 평가한다.",
    )
    parser.add_argument(
        "--no-ensemble",
        dest="include_ensemble",
        action="store_false",
        help="앙상블 평가를 비활성화한다.",
    )

    parser.add_argument(
        "--out-json",
        default="reports/rolling_validation.json",
        help="롤링 검증 JSON 저장 경로.",
    )
    parser.add_argument(
        "--out-md",
        default="reports/rolling_validation.md",
        help="롤링 검증 Markdown 저장 경로.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전 상위 디렉터리를 만든다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def parse_models(raw: str) -> list[str]:
    """모델 목록 문자열을 파싱한다."""
    models = [value.strip() for value in raw.split(",") if value.strip()]
    return list(dict.fromkeys(models))


def parse_thresholds(raw: str) -> list[int]:
    """임계값 문자열을 정수 리스트로 변환한다."""
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def build_folds(
    draws: list[int],
    min_train_draws: int,
    fold_test_size: int,
    fold_step_size: int,
    max_folds: int,
) -> list[dict]:
    """확장 윈도우 방식의 롤링 폴드 목록을 구성한다."""
    if len(draws) < min_train_draws + fold_test_size:
        return []

    starts = list(
        range(min_train_draws, len(draws) - fold_test_size + 1, max(1, fold_step_size))
    )
    if max_folds > 0 and len(starts) > max_folds:
        starts = starts[-max_folds:]

    folds = []
    for fold_idx, start_idx in enumerate(starts, start=1):
        train_draws = draws[:start_idx]
        test_draws = draws[start_idx : start_idx + fold_test_size]
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


def build_model_namespace(args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    """train_model.build_model()에 전달할 인자 네임스페이스를 만든다."""
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    if model_name == "lightgbm":
        if args.lgbm_n_estimators is not None:
            n_estimators = args.lgbm_n_estimators
        if args.lgbm_max_depth is not None:
            max_depth = args.lgbm_max_depth

    return argparse.Namespace(
        model=model_name,
        seed=args.seed,
        gbdt_max_iter=args.gbdt_max_iter,
        gbdt_learning_rate=args.gbdt_learning_rate,
        gbdt_max_depth=args.gbdt_max_depth,
        n_estimators=n_estimators,
        learning_rate=args.learning_rate,
        max_depth=max_depth,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_max_iter=args.mlp_max_iter,
        mlp_learning_rate_init=args.mlp_learning_rate_init,
        mlp_tol=args.mlp_tol,
        mlp_validation_fraction=args.mlp_validation_fraction,
        mlp_n_iter_no_change=args.mlp_n_iter_no_change,
        mlp_early_stopping=args.mlp_early_stopping,
    )


def summarize_values(values: list[float]) -> dict:
    """값 목록의 평균/표준편차/최솟값/최댓값을 계산한다."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_model(metrics_list: list[dict], thresholds: list[int]) -> dict:
    """모델의 폴드별 메트릭을 집계한다."""
    average_hits = [float(item["average_hits"]) for item in metrics_list]
    summary = {
        "folds": len(metrics_list),
        "average_hits": summarize_values(average_hits),
        "hit_rates": {},
    }
    for threshold in thresholds:
        values = [float(item["hit_rates"][threshold]) for item in metrics_list]
        summary["hit_rates"][threshold] = summarize_values(values)
    for key in ("mrr", "mean_min_rank", "brier", "log_loss", "ece"):
        values = [float(item[key]) for item in metrics_list if key in item]
        if values:
            summary[key] = summarize_values(values)
    portfolio_metrics = [item["portfolio"] for item in metrics_list if "portfolio" in item]
    if portfolio_metrics:
        summary["portfolio"] = {
            "average_best_hits": summarize_values(
                [float(item["average_best_hits"]) for item in portfolio_metrics]
            ),
            "average_unique_numbers": summarize_values(
                [float(item["average_unique_numbers"]) for item in portfolio_metrics]
            ),
            "average_pairwise_overlap": summarize_values(
                [float(item["average_pairwise_overlap"]) for item in portfolio_metrics]
            ),
            "expected_payout": summarize_values(
                [float(item["expected_payout"]) for item in portfolio_metrics]
            ),
            "expected_profit": summarize_values(
                [float(item["expected_profit"]) for item in portfolio_metrics]
            ),
            "roi": summarize_values([float(item["roi"]) for item in portfolio_metrics]),
            "best_hit_rates": {
                threshold: summarize_values(
                    [float(item["best_hit_rates"][threshold]) for item in portfolio_metrics]
                )
                for threshold in thresholds
            },
            "tier_hit_rates": {
                tier: summarize_values(
                    [float(item["tier_hit_rates"][tier]) for item in portfolio_metrics]
                )
                for tier in range(1, 6)
            },
        }
    return summary


def score_metrics(metrics: dict, thresholds: list[int]) -> float:
    """모델 우열 판정을 위한 종합 점수를 계산한다."""
    score = float(metrics.get("average_hits", 0.0))
    hit_rates = metrics.get("hit_rates", {})

    weight_by_threshold = {1: 0.03, 2: 0.07, 3: 0.20, 4: 0.05, 5: 0.02}
    for threshold in thresholds:
        weight = weight_by_threshold.get(threshold, 0.01)
        value = hit_rates.get(threshold)
        if value is None:
            value = hit_rates.get(str(threshold), 0.0)
        score += float(value) * weight

    score += float(metrics.get("mrr", 0.0)) * 0.15
    mean_min_rank = float(metrics.get("mean_min_rank", 0.0))
    if mean_min_rank > 0:
        score += (1.0 / mean_min_rank) * 0.40

    score -= float(metrics.get("brier", 0.0)) * 0.10
    score -= float(metrics.get("log_loss", 0.0)) * 0.05
    score -= float(metrics.get("ece", 0.0)) * 0.10
    portfolio = metrics.get("portfolio", {})
    if portfolio:
        score += float(portfolio.get("average_best_hits", 0.0)) * 0.25
        score += float(portfolio.get("roi", 0.0)) * 0.02
    return float(score)


def score_summary(summary: dict, thresholds: list[int]) -> float:
    """집계 요약값(평균)으로 종합 점수를 계산한다."""
    metrics = {
        "average_hits": summary["average_hits"]["mean"],
        "hit_rates": {
            threshold: summary["hit_rates"][threshold]["mean"] for threshold in thresholds
        },
    }
    for key in ("mrr", "mean_min_rank", "brier", "log_loss", "ece"):
        if key in summary:
            metrics[key] = summary[key]["mean"]
    portfolio = summary.get("portfolio")
    if portfolio:
        metrics["portfolio"] = {
            "average_best_hits": portfolio["average_best_hits"]["mean"],
            "expected_profit": portfolio["expected_profit"]["mean"],
            "roi": portfolio["roi"]["mean"],
        }
    return score_metrics(metrics, thresholds)


def pick_best_model(fold_metrics: dict, thresholds: list[int]) -> str:
    """한 폴드에서 종합 점수가 가장 높은 모델명을 반환한다."""
    if not fold_metrics:
        return ""
    ranked = sorted(
        fold_metrics.items(),
        key=lambda item: score_metrics(item[1], thresholds),
        reverse=True,
    )
    return ranked[0][0]


def render_markdown(
    args: argparse.Namespace,
    thresholds: list[int],
    folds: list[dict],
    fold_results: list[dict],
    model_summaries: dict,
    unavailable_models: dict,
) -> str:
    """롤링 검증 마크다운 리포트를 생성한다."""
    lines = [
        "# 롤링 백테스트 리포트",
        "",
        "## 설정",
        f"- 데이터: `{args.in_parquet}`",
        f"- 모델: {', '.join(model_summaries.keys()) if model_summaries else '(없음)'}",
        f"- 최소 학습 회차: {args.min_train_draws}",
        f"- 폴드 테스트 크기: {args.fold_test_size}",
        f"- 폴드 이동 간격: {args.fold_step_size}",
        f"- 사용 폴드 수: {len(folds)}",
        f"- ECE bin 수: {args.calibration_bins}",
        f"- 앙상블 평가: {'사용' if args.include_ensemble else '미사용'}",
        "",
    ]

    if unavailable_models:
        lines.extend(["## 미사용 모델", ""])
        for model, reason in unavailable_models.items():
            lines.append(f"- {model}: {reason}")
        lines.append("")

    if model_summaries:
        header = ["모델", "평균 일치(평균±표준편차)"] + [
            f"Hit@{t} 평균±표준편차" for t in thresholds
        ] + [
            "MRR(평균±표준편차)",
            "Brier(평균±표준편차)",
            "ECE(평균±표준편차)",
            "포트폴리오 최대일치",
            "포트폴리오 예상수익",
            "포트폴리오 ROI",
            "종합점수",
        ]
        lines.extend(["## 모델 요약", ""])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        sorted_items = sorted(
            model_summaries.items(),
            key=lambda item: score_summary(item[1], thresholds),
            reverse=True,
        )
        for model_name, summary in sorted_items:
            avg = summary["average_hits"]
            row = [model_name, f"{avg['mean']:.4f} ± {avg['std']:.4f}"]
            for threshold in thresholds:
                hit = summary["hit_rates"][threshold]
                row.append(f"{hit['mean']:.4f} ± {hit['std']:.4f}")
            mrr = summary.get("mrr", {"mean": 0.0, "std": 0.0})
            brier = summary.get("brier", {"mean": 0.0, "std": 0.0})
            ece = summary.get("ece", {"mean": 0.0, "std": 0.0})
            portfolio = summary.get("portfolio", {})
            row.append(f"{mrr['mean']:.4f} ± {mrr['std']:.4f}")
            row.append(f"{brier['mean']:.6f} ± {brier['std']:.6f}")
            row.append(f"{ece['mean']:.6f} ± {ece['std']:.6f}")
            if portfolio:
                row.append(
                    f"{portfolio['average_best_hits']['mean']:.4f} ± "
                    f"{portfolio['average_best_hits']['std']:.4f}"
                )
                row.append(
                    f"{portfolio['expected_profit']['mean']:.2f} ± "
                    f"{portfolio['expected_profit']['std']:.2f}"
                )
                row.append(
                    f"{portfolio['roi']['mean']:.6f} ± "
                    f"{portfolio['roi']['std']:.6f}"
                )
            else:
                row.extend(["N/A", "N/A", "N/A"])
            row.append(f"{score_summary(summary, thresholds):.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    winner_counts: dict[str, int] = defaultdict(int)
    for fold in fold_results:
        model_name = fold.get("best_model", "")
        if model_name:
            winner_counts[model_name] += 1
    if winner_counts:
        lines.extend(["## 폴드 승자 집계", ""])
        lines.append("| 모델 | 승자 폴드 수 |")
        lines.append("| --- | --- |")
        for model_name, count in sorted(
            winner_counts.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"| {model_name} | {count} |")
        lines.append("")

    lines.extend(["## 폴드 구간", ""])
    lines.append("| Fold | 학습 구간 | 테스트 구간 |")
    lines.append("| --- | --- | --- |")
    for fold in folds:
        lines.append(
            f"| {fold['fold']} | {fold['train_span'][0]}..{fold['train_span'][1]} | "
            f"{fold['test_span'][0]}..{fold['test_span'][1]} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.in_parquet):
        print(f"Missing input file: {args.in_parquet}")
        return 1
    if args.portfolio_size > 0 and not os.path.exists(args.in_processed):
        print(f"Missing processed draw file: {args.in_processed}")
        return 1

    models = parse_models(args.models)
    if not models:
        print("검증할 모델이 없습니다.")
        return 1

    thresholds = parse_thresholds(args.hit_thresholds)
    df = pd.read_parquet(args.in_parquet).fillna(0)
    feature_cols = [col for col in df.columns if col not in ("label", "draw_no")]
    draw_contexts = lp.load_draw_contexts(args.in_processed)
    portfolio_config = tm.build_portfolio_config(args)
    prize_config = tm.build_prize_config(args)

    draws = sorted(int(value) for value in df["draw_no"].unique())
    folds = build_folds(
        draws,
        min_train_draws=args.min_train_draws,
        fold_test_size=args.fold_test_size,
        fold_step_size=args.fold_step_size,
        max_folds=args.max_folds,
    )
    if not folds:
        print("롤링 검증 폴드를 만들 수 없습니다. min-train-draws/fold-test-size를 조정하세요.")
        return 1

    unavailable_models: dict[str, str] = {}
    active_models = models[:]
    per_model_metrics: dict[str, list[dict]] = defaultdict(list)
    fold_results: list[dict] = []

    for fold in folds:
        train_df = df[df["draw_no"].isin(fold["train_draws"])].copy()
        test_df = df[df["draw_no"].isin(fold["test_draws"])].copy()

        fold_payload = {
            "fold": fold["fold"],
            "train_span": fold["train_span"],
            "test_span": fold["test_span"],
            "metrics": {},
        }
        fold_probas: dict[str, np.ndarray] = {}

        for model_name in active_models[:]:
            model_args = build_model_namespace(args, model_name)
            try:
                model = tm.build_model(model_args)
            except tm.OptionalDependencyError as exc:
                unavailable_models[model_name] = str(exc)
                active_models.remove(model_name)
                continue

            model.fit(train_df[feature_cols], train_df["label"])
            proba = model.predict_proba(test_df[feature_cols])[:, 1]
            fold_probas[model_name] = np.asarray(proba, dtype=float)
            metrics = tm.evaluate_from_proba(
                fold_probas[model_name],
                test_df,
                thresholds,
                calibration_bins=args.calibration_bins,
                draw_contexts=draw_contexts,
                portfolio_config=portfolio_config,
                prize_config=prize_config,
            )
            per_model_metrics[model_name].append(metrics)
            fold_payload["metrics"][model_name] = metrics

        if args.include_ensemble and len(fold_probas) >= 2:
            model_names = sorted(fold_probas.keys())
            stacked = np.vstack([fold_probas[name] for name in model_names])
            ensemble_proba = np.mean(stacked, axis=0)
            ensemble_name = "ensemble_avg"
            ensemble_metrics = tm.evaluate_from_proba(
                ensemble_proba,
                test_df,
                thresholds,
                calibration_bins=args.calibration_bins,
                draw_contexts=draw_contexts,
                portfolio_config=portfolio_config,
                prize_config=prize_config,
            )
            per_model_metrics[ensemble_name].append(ensemble_metrics)
            fold_payload["metrics"][ensemble_name] = ensemble_metrics

        fold_payload["best_model"] = pick_best_model(fold_payload["metrics"], thresholds)
        fold_results.append(fold_payload)

    if not per_model_metrics:
        print("사용 가능한 모델이 없습니다. 의존성 설치 상태를 확인하세요.")
        return 1

    model_summaries = {
        model_name: summarize_model(metrics_list, thresholds)
        for model_name, metrics_list in per_model_metrics.items()
    }

    payload = {
        "config": {
            "data_path": args.in_parquet,
            "models_requested": models,
            "models_used": list(model_summaries.keys()),
            "hit_thresholds": thresholds,
            "min_train_draws": args.min_train_draws,
            "fold_test_size": args.fold_test_size,
            "fold_step_size": args.fold_step_size,
            "max_folds": args.max_folds,
            "feature_count": len(feature_cols),
            "calibration_bins": args.calibration_bins,
            "include_ensemble": args.include_ensemble,
            "portfolio_config": portfolio_config,
            "prize_config": {
                "ticket_price": prize_config["ticket_price"],
                "tier1_estimate": prize_config["tier1_estimate"],
                "tier_payouts": prize_config["tier_payouts"],
            },
        },
        "unavailable_models": unavailable_models,
        "folds": fold_results,
        "summary": model_summaries,
    }

    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(tm.to_json_compatible(payload), handle, ensure_ascii=False, indent=2)

    md_text = render_markdown(
        args,
        thresholds,
        folds,
        fold_results,
        model_summaries,
        unavailable_models,
    )
    ensure_parent_dir(args.out_md)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write(md_text)

    print(f"롤링 검증 JSON 저장: {args.out_json}")
    print(f"롤링 검증 리포트 저장: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
