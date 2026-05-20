import datetime as dt
import json

import pytest

from lottoml.data.fetch import (
    backfill_range,
    discover_latest_draw,
    fetch_draw_urllib,
    FetchError,
)
from lottoml.data.types import Draw


SAMPLE_API_RESPONSE = {
    "resultCode": None,
    "resultMessage": None,
    "data": {
        "list": [
            {
                "ltEpsd": 1,
                "ltRflYmd": "20021207",
                "tm1WnNo": 10, "tm2WnNo": 23, "tm3WnNo": 29,
                "tm4WnNo": 33, "tm5WnNo": 37, "tm6WnNo": 40,
                "bnsWnNo": 16,
                "rlvtEpsdSumNtslAmt": 3681782000,
                "rnk1WnAmt": 0, "rnk1WnNope": 0,
                "rnk2WnAmt": 143934100, "rnk2WnNope": 1,
                "rnk3WnAmt": 5140500, "rnk3WnNope": 28,
            }
        ]
    },
}


def _make_fake_urlopen(payload_bytes: bytes):
    def fake_urlopen(request, timeout=None):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return payload_bytes
        return _Resp()
    return fake_urlopen


def test_fetch_draw_urllib_parses_response(monkeypatch) -> None:
    payload = json.dumps(SAMPLE_API_RESPONSE).encode()
    monkeypatch.setattr("urllib.request.urlopen", _make_fake_urlopen(payload))

    draw = fetch_draw_urllib(1)
    assert draw.draw_no == 1
    assert draw.numbers == (10, 23, 29, 33, 37, 40)
    assert draw.bonus == 16
    assert draw.draw_date == dt.date(2002, 12, 7)
    assert draw.second_prize == 143934100
    assert draw.third_prize == 5140500


def test_fetch_draw_urllib_raises_on_fail(monkeypatch) -> None:
    payload = json.dumps({"data": {"list": []}}).encode()
    monkeypatch.setattr("urllib.request.urlopen", _make_fake_urlopen(payload))

    with pytest.raises(FetchError):
        fetch_draw_urllib(99999)


def test_fetch_draw_urllib_raises_on_non_json(monkeypatch) -> None:
    """폐기된 엔드포인트가 HTML로 리다이렉트할 때 FetchError가 발생해야 함."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _make_fake_urlopen(b"<html>moved</html>")
    )
    with pytest.raises(FetchError):
        fetch_draw_urllib(1)


def test_backfill_range_calls_each_draw(monkeypatch) -> None:
    calls: list[int] = []

    def fake_fetch(draw_no: int, *, timeout: float = 10.0) -> Draw:
        calls.append(draw_no)
        return Draw(
            draw_no=draw_no,
            draw_date=dt.date(2026, 1, 1),
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            total_sales=0, first_prize=0, first_winners=0,
            second_prize=0, second_winners=0,
            third_prize=0, third_winners=0,
        )

    monkeypatch.setattr("lottoml.data.fetch.fetch_draw_urllib", fake_fetch)
    draws = backfill_range(start=5, end=8, sleep_seconds=0.0)
    assert calls == [5, 6, 7, 8]
    assert [d.draw_no for d in draws] == [5, 6, 7, 8]


def test_backfill_range_skips_failed_draws(monkeypatch) -> None:
    def fake_fetch(draw_no: int, *, timeout: float = 10.0) -> Draw:
        if draw_no == 6:
            raise FetchError("nope")
        return Draw(
            draw_no=draw_no,
            draw_date=dt.date(2026, 1, 1),
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            total_sales=0, first_prize=0, first_winners=0,
            second_prize=0, second_winners=0,
            third_prize=0, third_winners=0,
        )

    monkeypatch.setattr("lottoml.data.fetch.fetch_draw_urllib", fake_fetch)
    draws = backfill_range(start=5, end=7, sleep_seconds=0.0)
    assert [d.draw_no for d in draws] == [5, 7]


def test_discover_latest_draw_finds_highest_responding(monkeypatch) -> None:
    """fetch_draw_urllib이 draw_no <= 250까지만 성공하면 250을 반환."""

    def fake_fetch(draw_no: int, *, timeout: float = 10.0) -> Draw:
        if draw_no > 250:
            raise FetchError(f"회차 {draw_no} 응답 실패")
        return Draw(
            draw_no=draw_no,
            draw_date=dt.date(2026, 1, 1),
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            total_sales=0, first_prize=0, first_winners=0,
            second_prize=0, second_winners=0,
            third_prize=0, third_winners=0,
        )

    monkeypatch.setattr("lottoml.data.fetch.fetch_draw_urllib", fake_fetch)
    assert discover_latest_draw(probe_step=4) == 250


def test_discover_latest_draw_raises_when_draw_one_fails(monkeypatch) -> None:
    """draw 1 자체가 실패하면 FetchError를 raise한다."""

    def fake_fetch(draw_no: int, *, timeout: float = 10.0) -> Draw:
        raise FetchError("API down")

    monkeypatch.setattr("lottoml.data.fetch.fetch_draw_urllib", fake_fetch)
    with pytest.raises(FetchError):
        discover_latest_draw()
