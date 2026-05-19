import datetime as dt
import json

import pytest

from lottoml.data.fetch import fetch_draw_urllib, FetchError, backfill_range
from lottoml.data.types import Draw


SAMPLE_API_RESPONSE = {
    "totSellamnt": 3681782000,
    "returnValue": "success",
    "drwNoDate": "2002-12-07",
    "firstWinamnt": 0,
    "drwtNo6": 40,
    "drwtNo4": 33,
    "firstPrzwnerCo": 0,
    "drwtNo5": 37,
    "bnusNo": 16,
    "firstAccumamnt": 863604600,
    "drwNo": 1,
    "drwtNo2": 23,
    "drwtNo3": 29,
    "drwtNo1": 10,
}


def test_fetch_draw_urllib_parses_response(monkeypatch) -> None:
    payload = json.dumps(SAMPLE_API_RESPONSE).encode()

    def fake_urlopen(url, timeout):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return payload
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    draw = fetch_draw_urllib(1)
    assert draw.draw_no == 1
    assert draw.numbers == (10, 23, 29, 33, 37, 40)
    assert draw.bonus == 16
    assert draw.draw_date == dt.date(2002, 12, 7)


def test_fetch_draw_urllib_raises_on_fail(monkeypatch) -> None:
    payload = json.dumps({"returnValue": "fail"}).encode()

    def fake_urlopen(url, timeout):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return payload
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(FetchError):
        fetch_draw_urllib(99999)


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
