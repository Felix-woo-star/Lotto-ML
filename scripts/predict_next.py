#!/usr/bin/env python3
"""학습된 모델로 다음 회차 추천 번호를 생성한다."""
import argparse
import json
import os
import pickle

import pandas as pd


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
        "--out-json",
        default=None,
        help="예측 결과 JSON 저장 경로.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="예측 결과 CSV 저장 경로.",
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
        print(f"Missing features file: {args.features_parquet}")
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
        print("No model paths provided.")
        return 1
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Missing model file: {path}")
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
            print(f"Model does not support predict_proba: {path}")
            return 1
        model_proba = model.predict_proba(latest_slice[feature_cols])[:, 1]
        if probas is None:
            probas = model_proba
        else:
            probas += model_proba

    if probas is None:
        print("Could not calculate probabilities.")
        return 1
    probas = probas / len(model_paths)
    latest_slice["proba"] = probas
    ranked = latest_slice.sort_values(["proba", "number"], ascending=[False, True])
    top_rows = ranked.head(args.top_k)
    top_numbers = [int(value) for value in top_rows["number"].tolist()]

    print(f"Latest draw: {latest_draw}")
    print(f"Predicted draw: {next_draw}")
    print(f"Top-{args.top_k} numbers: {top_numbers}")
    if len(model_paths) > 1:
        print(f"Ensemble models: {len(model_paths)}")

    if args.out_json or args.out_csv:
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["is_top_k"] = ranked["rank"] <= args.top_k
        ranked["latest_draw"] = latest_draw
        ranked["predicted_draw"] = next_draw
        ranked["model_count"] = len(model_paths)

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
        print(f"Saved JSON predictions to {args.out_json}")

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
        print(f"Saved CSV predictions to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
