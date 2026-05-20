"""학습된 ranker로 다음 회차 45개 확률 산출."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from ..data.types import Draw
from ..features.build import build_features
from .train import FEATURE_COLS


def predict_probabilities(draws: Iterable[Draw], *, model_path: Path) -> dict[int, float]:
    """가장 최근 회차 기준 피처로 다음 회차 번호별 확률 산출."""
    df = build_features(list(draws))
    if df.empty:
        raise ValueError("회차 데이터가 부족합니다")

    latest = int(df["draw_no"].max())
    latest_slice = df[df["draw_no"] == latest].sort_values("number")
    # DataFrame을 유지해 학습 때와 동일한 feature name으로 전달한다
    X = latest_slice[list(FEATURE_COLS)]

    with model_path.open("rb") as handle:
        calibrator = pickle.load(handle)
    probs = calibrator.predict_proba(X)[:, 1]

    return {
        int(number): float(prob)
        for number, prob in zip(latest_slice["number"], probs)
    }
