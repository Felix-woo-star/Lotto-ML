"""동행복권 회차 데이터 수집."""
from __future__ import annotations

import datetime as dt
import json
import urllib.request

from .types import Draw

API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
DEFAULT_TIMEOUT = 10.0


class FetchError(RuntimeError):
    """동행복권 응답이 비정상이거나 회차가 존재하지 않을 때."""


def fetch_draw_urllib(draw_no: int, *, timeout: float = DEFAULT_TIMEOUT) -> Draw:
    """urllib 기반 단일 회차 조회. 2등/3등 금액은 응답에 없어 0으로 채움."""
    url = API_URL.format(draw_no=draw_no)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("returnValue") != "success":
        raise FetchError(f"회차 {draw_no} 응답 실패: {payload.get('returnValue')}")

    return Draw(
        draw_no=int(payload["drwNo"]),
        draw_date=dt.date.fromisoformat(payload["drwNoDate"]),
        numbers=(
            int(payload["drwtNo1"]), int(payload["drwtNo2"]),
            int(payload["drwtNo3"]), int(payload["drwtNo4"]),
            int(payload["drwtNo5"]), int(payload["drwtNo6"]),
        ),
        bonus=int(payload["bnusNo"]),
        total_sales=int(payload.get("totSellamnt", 0)),
        first_prize=int(payload.get("firstWinamnt", 0)),
        first_winners=int(payload.get("firstPrzwnerCo", 0)),
        second_prize=0,
        second_winners=0,
        third_prize=0,
        third_winners=0,
    )
