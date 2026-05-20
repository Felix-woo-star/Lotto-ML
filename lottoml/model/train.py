"""LightGBM 단일 ranker 학습 + Platt calibration."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ..data.types import Draw
from ..features.build import build_features

FEATURE_COLS = (
    "freq_20", "freq_50", "freq_100",
    "last_seen_gap", "bonus_freq_50",
    "bucket", "is_odd", "mod_10",
)


def train_ranker(
    draws: Iterable[Draw],
    *,
    out_model: Path,
    out_meta: Path,
    holdout: int = 50,
    decay: float = 0.997,
    seed: int = 42,
) -> dict:
    """피처 빌드 → LightGBM 학습 → Platt 캘리브레이션 → 저장 + 메트릭 반환."""
    df = build_features(list(draws))
    if df.empty:
        raise ValueError("회차 데이터가 부족합니다 (>=101 필요)")

    draws_sorted = sorted({int(v) for v in df["draw_no"].unique()})
    if len(draws_sorted) <= holdout:
        raise ValueError("학습 데이터가 holdout보다 작습니다")
    cutoff = draws_sorted[-holdout - 1]

    train_mask = df["draw_no"] <= cutoff
    test_mask = df["draw_no"] > cutoff
    # DataFrame을 유지해 feature name이 학습·예측 양쪽에 보존되도록 한다
    # (LightGBM의 "X does not have valid feature names" 경고 방지)
    X_train = df.loc[train_mask, list(FEATURE_COLS)]
    y_train = df.loc[train_mask, "label"].to_numpy()
    X_test = df.loc[test_mask, list(FEATURE_COLS)]

    weights = np.power(decay, cutoff - df.loc[train_mask, "draw_no"].to_numpy())

    base = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        min_data_in_leaf=50,
        learning_rate=0.03,
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective="binary",
        random_state=seed,
        verbosity=-1,
    )
    base.fit(X_train, y_train, sample_weight=weights)

    calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrator.fit(X_train, y_train, sample_weight=weights)

    probs = calibrator.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(df.loc[test_mask].assign(p=probs))

    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as handle:
        pickle.dump(calibrator, handle)
    meta = {
        "train_draws": [int(draws_sorted[0]), int(cutoff)],
        "holdout_draws": [int(cutoff + 1), int(draws_sorted[-1])],
        "feature_cols": list(FEATURE_COLS),
        "decay": decay,
        "metrics": metrics,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _compute_metrics(scored) -> dict:
    y = scored["label"].to_numpy(dtype=float)
    p = np.clip(scored["p"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(mask):
            continue
        ece += abs(np.mean(y[mask]) - np.mean(p[mask])) * (np.sum(mask) / len(y))

    hits = []
    for _, group in scored.groupby("draw_no"):
        top6 = set(group.sort_values("p", ascending=False).head(6)["number"])
        actual = set(group.loc[group["label"] == 1, "number"])
        hits.append(1 if top6 & actual else 0)
    top6_hit = float(np.mean(hits)) if hits else 0.0

    return {"brier": brier, "ece": ece, "top6_hit": top6_hit}
