#!/usr/bin/env python3
"""피처 데이터로 모델을 학습하고 번호 추천 성능을 평가한다."""

import argparse
import importlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class OptionalDependencyError(RuntimeError):
    """선택 의존성 패키지가 없을 때 발생하는 오류."""


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Train and evaluate lotto model.")
    parser.add_argument(
        "--in-parquet",
        default="data/features/lotto_features.parquet",
        help="피처 Parquet 경로.",
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
        "--model",
        default="logreg",
        choices=[
            "logreg",
            "gbdt",
            "randomforest",
            "extratrees",
            "mlp",
            "lightgbm",
            "xgboost",
            "catboost",
        ],
        help="모델 종류.",
    )

    parser.add_argument(
        "--gbdt-max-iter",
        type=int,
        default=200,
        help="GBDT 반복 횟수.",
    )
    parser.add_argument(
        "--gbdt-learning-rate",
        type=float,
        default=0.1,
        help="GBDT 학습률.",
    )
    parser.add_argument(
        "--gbdt-max-depth",
        type=int,
        default=3,
        help="GBDT 최대 깊이.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="트리 기반 모델의 트리 개수.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="부스팅 계열 모델의 학습률.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="트리 기반 모델의 최대 깊이(0 이하면 제한 없음).",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="LightGBM 리프 개수.",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=20,
        help="LightGBM leaf 최소 샘플 수.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="XGBoost subsample.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.8,
        help="XGBoost colsample_bytree.",
    )

    parser.add_argument(
        "--mlp-hidden-layers",
        default="64,32",
        help="MLP hidden layer 크기(쉼표 구분). 예: 64,32",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=1000,
        help="MLP 최대 반복 횟수.",
    )
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=0.001,
        help="MLP 초기 학습률.",
    )
    parser.add_argument(
        "--mlp-tol",
        type=float,
        default=1e-3,
        help="MLP 수렴 허용오차(tol).",
    )
    parser.add_argument(
        "--mlp-validation-fraction",
        type=float,
        default=0.1,
        help="MLP early stopping 검증 데이터 비율.",
    )
    parser.add_argument(
        "--mlp-n-iter-no-change",
        type=int,
        default=25,
        help="MLP early stopping 기준 개선 없음 반복 횟수.",
    )
    parser.add_argument(
        "--mlp-early-stopping",
        dest="mlp_early_stopping",
        action="store_true",
        default=True,
        help="MLP early stopping 사용(기본값: 사용).",
    )
    parser.add_argument(
        "--no-mlp-early-stopping",
        dest="mlp_early_stopping",
        action="store_false",
        help="MLP early stopping 비활성화.",
    )

    parser.add_argument(
        "--out-model",
        default=None,
        help="학습 모델 저장 경로(pickle).",
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


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전 상위 디렉터리를 만든다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def parse_hidden_layers(raw: str) -> tuple[int, ...]:
    """MLP hidden layer 문자열을 튜플로 변환한다."""
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    return tuple(values) if values else (64, 32)


def optional_import(module_name: str, package_name: str, model_name: str):
    """선택 의존성 모듈을 임포트한다."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        compatibility_note = ""
        troubleshooting = ""
        if package_name == "catboost" and sys.version_info >= (3, 14):
            compatibility_note = (
                " 현재 Python 3.14에서는 CatBoost wheel 호환이 제한될 수 있습니다. "
                "Python 3.13 환경 사용 또는 catboost 모델 제외를 권장합니다."
            )
        if package_name == "lightgbm" and isinstance(exc, OSError):
            if "libomp.dylib" in str(exc):
                troubleshooting = (
                    " macOS에서는 OpenMP 런타임이 필요합니다. "
                    "`brew install libomp` 설치 후 다시 시도하세요."
                )
        raise OptionalDependencyError(
            f"모델 '{model_name}'에는 패키지 '{package_name}'가 필요합니다. "
            f"`uv sync --extra advanced-models` 후 다시 실행하세요.{compatibility_note}"
            f"{troubleshooting} 원인: {exc}"
        ) from exc


def split_by_draw(
    df: pd.DataFrame, train_end: int | None, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """회차 기준 시간 순 분할을 수행한다."""
    draws = sorted(int(value) for value in df["draw_no"].unique())
    if train_end is not None:
        train_draws = [value for value in draws if value <= train_end]
        test_draws = [value for value in draws if value > train_end]
    else:
        if test_size <= 0:
            train_draws, test_draws = draws, []
        else:
            split_index = max(len(draws) - test_size, 0)
            train_draws, test_draws = draws[:split_index], draws[split_index:]
    train_df = df[df["draw_no"].isin(train_draws)].copy()
    test_df = df[df["draw_no"].isin(test_draws)].copy()
    return train_df, test_df, train_draws, test_draws


def build_model(args: argparse.Namespace):
    """모델 인자에 맞는 분류기를 생성한다."""
    max_depth = args.max_depth if args.max_depth > 0 else None

    if args.model == "gbdt":
        clf = HistGradientBoostingClassifier(
            max_iter=args.gbdt_max_iter,
            learning_rate=args.gbdt_learning_rate,
            max_depth=args.gbdt_max_depth,
            random_state=args.seed,
        )
        return Pipeline([("clf", clf)])

    if args.model == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if args.model == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])

    if args.model == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])

    if args.model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=parse_hidden_layers(args.mlp_hidden_layers),
            max_iter=args.mlp_max_iter,
            learning_rate="adaptive",
            learning_rate_init=args.mlp_learning_rate_init,
            tol=args.mlp_tol,
            early_stopping=args.mlp_early_stopping,
            validation_fraction=args.mlp_validation_fraction,
            n_iter_no_change=args.mlp_n_iter_no_change,
            random_state=args.seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if args.model == "lightgbm":
        lgb = optional_import("lightgbm", "lightgbm", args.model)
        clf = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=max_depth if max_depth is not None else -1,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            objective="binary",
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return Pipeline([("clf", clf)])

    if args.model == "xgboost":
        xgb = optional_import("xgboost", "xgboost", args.model)
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=max_depth if max_depth is not None else 6,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])

    if args.model == "catboost":
        cb = optional_import("catboost", "catboost", args.model)
        clf = cb.CatBoostClassifier(
            iterations=args.n_estimators,
            depth=max_depth if max_depth is not None else 6,
            learning_rate=args.learning_rate,
            loss_function="Logloss",
            random_seed=args.seed,
            verbose=False,
        )
        return Pipeline([("clf", clf)])

    raise ValueError(f"지원하지 않는 모델입니다: {args.model}")


def evaluate(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    thresholds: list[int],
    calibration_bins: int = 10,
) -> dict:
    """테스트 구간에서 Top-6 추천 정확도를 계산한다."""
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    return evaluate_from_proba(
        proba,
        test_df,
        thresholds,
        calibration_bins=calibration_bins,
    )


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    """Equal-width bins 기반 ECE를 계산한다."""
    if labels.size == 0:
        return 0.0

    bins = max(2, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = float(labels.size)
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bucket_labels = labels[mask]
        bucket_probs = probabilities[mask]
        accuracy = float(np.mean(bucket_labels))
        confidence = float(np.mean(bucket_probs))
        weight = float(np.sum(mask)) / total
        ece += abs(accuracy - confidence) * weight
    return ece


def evaluate_from_proba(
    proba: np.ndarray,
    test_df: pd.DataFrame,
    thresholds: list[int],
    calibration_bins: int = 10,
) -> dict:
    """확률값 배열을 받아 Top-6 추천 및 캘리브레이션 지표를 계산한다."""
    scored = test_df.copy()
    scored["proba"] = proba

    hits = []
    reciprocal_ranks = []
    min_ranks = []
    for _, group in scored.groupby("draw_no"):
        ranked = (
            group.sort_values(["proba", "number"], ascending=[False, True])
            .reset_index(drop=True)
            .copy()
        )
        ranked["rank"] = ranked.index + 1
        top_numbers = ranked.head(6)["number"].tolist()
        actual = set(group.loc[group["label"] == 1, "number"])
        hits.append(len(actual.intersection(top_numbers)))

        actual_ranks = ranked.loc[ranked["label"] == 1, "rank"].tolist()
        if actual_ranks:
            min_rank = float(min(actual_ranks))
            min_ranks.append(min_rank)
            reciprocal_ranks.append(1.0 / min_rank)

    average_hits = sum(hits) / len(hits) if hits else 0.0
    hit_rates = {
        threshold: sum(1 for value in hits if value >= threshold) / len(hits)
        if hits
        else 0.0
        for threshold in thresholds
    }
    labels = scored["label"].to_numpy(dtype=float)
    probabilities = np.asarray(proba, dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    brier = float(np.mean((probabilities - labels) ** 2)) if labels.size else 0.0
    log_loss = (
        float(
            -np.mean(
                labels * np.log(probabilities)
                + (1.0 - labels) * np.log(1.0 - probabilities)
            )
        )
        if labels.size
        else 0.0
    )
    ece = expected_calibration_error(labels, probabilities, calibration_bins)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_min_rank = float(np.mean(min_ranks)) if min_ranks else 0.0

    return {
        "average_hits": average_hits,
        "hit_rates": hit_rates,
        "draws": len(hits),
        "mrr": mrr,
        "mean_min_rank": mean_min_rank,
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "calibration_bins": int(max(2, calibration_bins)),
    }


def write_json(path: str, payload: dict) -> None:
    """JSON 파일로 결과를 저장한다."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_json_compatible(payload), handle, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """CSV 파일로 결과를 저장한다."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False, columns=fieldnames)


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


def build_model_params(args: argparse.Namespace) -> dict:
    """선택된 모델의 하이퍼파라미터를 딕셔너리로 반환한다."""
    if args.model == "gbdt":
        return {
            "gbdt_max_iter": args.gbdt_max_iter,
            "gbdt_learning_rate": args.gbdt_learning_rate,
            "gbdt_max_depth": args.gbdt_max_depth,
        }
    if args.model in {"randomforest", "extratrees"}:
        return {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        }
    if args.model == "mlp":
        return {
            "mlp_hidden_layers": args.mlp_hidden_layers,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_learning_rate_init": args.mlp_learning_rate_init,
            "mlp_tol": args.mlp_tol,
            "mlp_validation_fraction": args.mlp_validation_fraction,
            "mlp_n_iter_no_change": args.mlp_n_iter_no_change,
            "mlp_early_stopping": args.mlp_early_stopping,
        }
    if args.model == "lightgbm":
        return {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
        }
    if args.model == "xgboost":
        return {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample_bytree": args.xgb_colsample_bytree,
        }
    if args.model == "catboost":
        return {
            "iterations": args.n_estimators,
            "learning_rate": args.learning_rate,
            "depth": args.max_depth,
        }
    return {}


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    if not os.path.exists(args.in_parquet):
        print(f"Missing input file: {args.in_parquet}")
        return 1

    df = pd.read_parquet(args.in_parquet).fillna(0)
    train_df, test_df, train_draws, test_draws = split_by_draw(
        df, args.train_end, args.test_size
    )
    if train_df.empty or test_df.empty:
        print("훈련/테스트 데이터가 부족합니다.")
        return 1

    thresholds = [
        int(value.strip()) for value in args.hit_thresholds.split(",") if value.strip()
    ]

    feature_cols = [col for col in df.columns if col not in ("label", "draw_no")]

    try:
        model = build_model(args)
    except OptionalDependencyError as exc:
        print(str(exc))
        return 2

    model.fit(train_df[feature_cols], train_df["label"])
    metrics = evaluate(
        model,
        test_df,
        feature_cols,
        thresholds,
        calibration_bins=args.calibration_bins,
    )

    print("Model metrics")
    print(f"- average_hits: {metrics['average_hits']:.4f}")
    for threshold in thresholds:
        print(f"- hit_rate_{threshold}: {metrics['hit_rates'][threshold]:.4f}")
    print(f"- mrr: {metrics['mrr']:.4f}")
    print(f"- mean_min_rank: {metrics['mean_min_rank']:.4f}")
    print(f"- brier: {metrics['brier']:.6f}")
    print(f"- log_loss: {metrics['log_loss']:.6f}")
    print(f"- ece: {metrics['ece']:.6f}")

    if args.out_model:
        ensure_parent_dir(args.out_model)
        with open(args.out_model, "wb") as handle:
            pickle.dump(model, handle)
        print(f"Saved model to {args.out_model}")

    if args.out_json or args.out_csv:
        config = {
            "data_path": args.in_parquet,
            "train_draws": len(train_draws),
            "test_draws": len(test_draws),
            "train_span": (train_draws[0], train_draws[-1]),
            "test_span": (test_draws[0], test_draws[-1]),
            "train_end": args.train_end if args.train_end is not None else "",
            "test_size_param": args.test_size,
            "eval_protocol": "single_top6",
            "hit_thresholds": ",".join(str(value) for value in thresholds),
            "calibration_bins": args.calibration_bins,
            "feature_count": len(feature_cols),
            "model": args.model,
            "model_params": build_model_params(args),
        }

        payload = {"config": config, "metrics": metrics}
        if args.out_json:
            write_json(args.out_json, payload)
            print(f"Saved JSON metrics to {args.out_json}")

        if args.out_csv:
            row = {"metric_type": "model", "average_hits": metrics["average_hits"]}
            for threshold in thresholds:
                row[f"hit_rate_{threshold}"] = metrics["hit_rates"][threshold]
            row["mrr"] = metrics["mrr"]
            row["mean_min_rank"] = metrics["mean_min_rank"]
            row["brier"] = metrics["brier"]
            row["log_loss"] = metrics["log_loss"]
            row["ece"] = metrics["ece"]
            row.update(config)
            fieldnames = list(row.keys())
            write_csv(args.out_csv, [row], fieldnames)
            print(f"Saved CSV metrics to {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
