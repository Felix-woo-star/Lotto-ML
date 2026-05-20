import datetime as dt

import pandas as pd

from lottoml.data.types import Draw
from lottoml.features.build import build_features


def _make_draw(no: int, numbers: tuple[int, ...], bonus: int = 1) -> Draw:
    return Draw(
        draw_no=no,
        draw_date=dt.date(2002, 12, 7),
        numbers=numbers,  # type: ignore[arg-type]
        bonus=bonus,
        total_sales=0, first_prize=0, first_winners=0,
        second_prize=0, second_winners=0,
        third_prize=0, third_winners=0,
    )


def test_build_features_returns_dataframe_with_expected_columns() -> None:
    draws = [
        _make_draw(no=i, numbers=tuple(((i + k) % 45) + 1 for k in range(6)))
        for i in range(1, 201)
    ]
    df = build_features(draws)
    expected_cols = {
        "draw_no", "number", "label",
        "freq_20", "freq_50", "freq_100",
        "last_seen_gap", "bonus_freq_50",
        "bucket", "is_odd", "mod_10",
    }
    assert expected_cols.issubset(df.columns)
    # 회차 100부터 피처 생성 (window 100 필요)
    assert df["draw_no"].min() == 101
    assert df["draw_no"].max() == 200
    # 각 회차마다 45행
    assert (df.groupby("draw_no").size() == 45).all()
    # 라벨 합은 회차당 6
    assert (df.groupby("draw_no")["label"].sum() == 6).all()


def test_build_features_bucket_and_parity() -> None:
    draws = [
        _make_draw(no=i, numbers=tuple(((i + k) % 45) + 1 for k in range(6)))
        for i in range(1, 201)
    ]
    df = build_features(draws)
    sample = df[df["draw_no"] == 150].set_index("number")
    # 번호 15는 bucket 1 (1-15)
    assert sample.loc[15, "bucket"] == 1
    # 번호 16은 bucket 2 (16-30)
    assert sample.loc[16, "bucket"] == 2
    # 번호 31은 bucket 3 (31-45)
    assert sample.loc[31, "bucket"] == 3
    # 홀짝
    assert sample.loc[7, "is_odd"] == 1
    assert sample.loc[8, "is_odd"] == 0
    # 끝자리
    assert sample.loc[23, "mod_10"] == 3
