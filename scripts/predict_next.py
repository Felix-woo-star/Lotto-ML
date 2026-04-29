#!/usr/bin/env python3
"""학습된 모델로 다음 회차 추천 번호를 생성한다."""
import argparse
import json
import os
import pickle

import pandas as pd
import lotto_portfolio as lp


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Predict next lotto numbers.")
    parser.add_argument(
        "--features-parquet",
        default="data/features/lotto_features.parquet",
        help="피처 Parquet 경로.",
    )
    parser.add_argument(
        "--model-path",
        default="models/logreg.pkl",
        help="학습된 모델 경로.",
    )
    parser.add_argument(
        "--model-paths",
        default=None,
        help="여러 모델 경로(쉼표 구분).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="추천할 번호 개수.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=256,
        help="생성할 후보 조합 수.",
    )
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=12,
        help="최종 추천 포트폴리오 티켓 수.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=18,
        help="후보 생성 시 샘플링에 사용할 상위 번호 풀 크기.",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.9,
        help="후보 생성용 확률 샘플링 temperature.",
    )
    parser.add_argument(
        "--overlap-penalty",
        type=float,
        default=0.18,
        help="포트폴리오 선택 시 중복 번호 패널티 강도.",
    )
    parser.add_argument(
        "--unique-bonus",
        type=float,
        default=0.035,
        help="포트폴리오 선택 시 새로운 번호 커버리지 보너스.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="후보 생성 난수 시드.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="예측 결과 JSON 저장 경로.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="예측 결과 CSV 저장 경로.",
    )
    parser.add_argument(
        "--out-portfolio-csv",
        default=None,
        help="포트폴리오 티켓 CSV 저장 경로.",
    )
    return parser.parse_args()


def to_json_compatible(value):
    """numpy/pandas 스칼라를 JSON 직렬화 가능한 기본 타입으로 변환한다."""
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.features_parquet):
        print(f"피처 파일이 없습니다: {args.features_parquet}")
        return 1

    model_paths = []
    if args.model_paths:
        model_paths.extend(
            [path.strip() for path in args.model_paths.split(",") if path.strip()]
        )
    if args.model_path:
        model_paths.append(args.model_path)
    model_paths = list(dict.fromkeys(model_paths))
    if not model_paths:
        print("모델 경로가 제공되지 않았습니다.")
        return 1
    for path in model_paths:
        if not os.path.exists(path):
            print(f"모델 파일이 없습니다: {path}")
            return 1

    df = pd.read_parquet(args.features_parquet)
    df = df.fillna(0)
    latest_draw = int(df["draw_no"].max())
    next_draw = latest_draw + 1
    latest_slice = df[df["draw_no"] == latest_draw].copy()

    feature_cols = [col for col in df.columns if col not in ("label", "draw_no")]
    probas = None
    for path in model_paths:
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        if not hasattr(model, "predict_proba"):
            print(f"predict_proba를 지원하지 않는 모델입니다: {path}")
            return 1
        model_proba = model.predict_proba(latest_slice[feature_cols])[:, 1]
        if probas is None:
            probas = model_proba
        else:
            probas += model_proba

    if probas is None:
        print("확률을 계산할 수 없습니다.")
        return 1
    probas = probas / len(model_paths)
    latest_slice["proba"] = probas
    ranked = latest_slice.sort_values(["proba", "number"], ascending=[False, True])
    top_rows = ranked.head(args.top_k)
    top_numbers = [int(value) for value in top_rows["number"].tolist()]

    probability_by_number = {
        int(row["number"]): float(row["proba"])
        for _, row in ranked[["number", "proba"]].iterrows()
    }
    portfolio_bundle = lp.build_portfolio(
        probability_by_number,
        num_candidates=args.num_candidates,
        portfolio_size=args.portfolio_size,
        top_pool_size=args.candidate_pool_size,
        temperature=args.sampling_temperature,
        overlap_penalty=args.overlap_penalty,
        unique_bonus=args.unique_bonus,
        seed=args.seed + next_draw,
    )
    portfolio = portfolio_bundle["portfolio"]
    portfolio_summary = portfolio_bundle["summary"]

    print(f"최신 회차: {latest_draw}")
    print(f"예측 대상 회차: {next_draw}")
    print(f"상위 {args.top_k}개 번호: {top_numbers}")
    print(f"생성된 후보 조합 수: {len(portfolio_bundle['candidates'])}")
    print(f"포트폴리오 크기: {len(portfolio)}")
    print(
        f"포트폴리오 고유 번호 수: {portfolio_summary['unique_number_count']} "
        f"(평균 중복도 {portfolio_summary['average_pairwise_overlap']:.4f})"
    )
    if len(model_paths) > 1:
        print(f"앙상블 모델 수: {len(model_paths)}")

    if args.out_json or args.out_csv or args.out_portfolio_csv:
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["is_top_k"] = ranked["rank"] <= args.top_k
        ranked["latest_draw"] = latest_draw
        ranked["predicted_draw"] = next_draw
        ranked["model_count"] = len(model_paths)

    portfolio_rows = []
    for candidate in portfolio:
        row = {
            "latest_draw": latest_draw,
            "predicted_draw": next_draw,
            "ticket_rank": int(candidate["portfolio_rank"]),
            "numbers": ",".join(str(number) for number in candidate["numbers"]),
            "score": float(candidate["selection_score"]),
            "sum_probability": float(candidate["sum_probability"]),
            "new_unique_numbers": int(candidate["new_unique_numbers"]),
            "max_overlap_with_previous": int(candidate["max_overlap_with_previous"]),
            "source": candidate["source"],
        }
        for idx, number in enumerate(candidate["numbers"], start=1):
            row[f"n{idx}"] = int(number)
        portfolio_rows.append(row)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "latest_draw": latest_draw,
            "predicted_draw": next_draw,
            "top_k": args.top_k,
            "model_paths": model_paths,
            "ensemble_method": "average",
            "recommendations": [
                {"number": int(row["number"]), "probability": float(row["proba"])}
                for _, row in top_rows.iterrows()
            ],
            "candidate_count": len(portfolio_bundle["candidates"]),
            "portfolio_config": {
                "num_candidates": args.num_candidates,
                "portfolio_size": args.portfolio_size,
                "candidate_pool_size": args.candidate_pool_size,
                "sampling_temperature": args.sampling_temperature,
                "overlap_penalty": args.overlap_penalty,
                "unique_bonus": args.unique_bonus,
                "seed": args.seed,
            },
            "portfolio_summary": portfolio_summary,
            "portfolio": [
                {
                    "ticket_rank": int(candidate["portfolio_rank"]),
                    "numbers": [int(number) for number in candidate["numbers"]],
                    "selection_score": float(candidate["selection_score"]),
                    "sum_probability": float(candidate["sum_probability"]),
                    "new_unique_numbers": int(candidate["new_unique_numbers"]),
                    "max_overlap_with_previous": int(
                        candidate["max_overlap_with_previous"]
                    ),
                    "source": candidate["source"],
                }
                for candidate in portfolio
            ],
            "ranked": [
                {
                    "number": int(row["number"]),
                    "probability": float(row["proba"]),
                    "rank": int(row["rank"]),
                    "is_top_k": bool(row["is_top_k"]),
                }
                for _, row in ranked.iterrows()
            ],
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(to_json_compatible(payload), handle, ensure_ascii=False, indent=2)
        print(f"JSON 예측 결과를 저장했습니다: {args.out_json}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        ranked[
            [
                "latest_draw",
                "predicted_draw",
                "number",
                "proba",
                "rank",
                "is_top_k",
                "model_count",
            ]
        ].to_csv(args.out_csv, index=False)
        print(f"CSV 예측 결과를 저장했습니다: {args.out_csv}")

    if args.out_portfolio_csv:
        os.makedirs(os.path.dirname(args.out_portfolio_csv) or ".", exist_ok=True)
        pd.DataFrame(portfolio_rows).to_csv(args.out_portfolio_csv, index=False)
        print(f"포트폴리오 CSV를 저장했습니다: {args.out_portfolio_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
