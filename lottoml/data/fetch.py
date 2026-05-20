"""동행복권 회차 데이터 수집."""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from typing import Callable

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


def backfill_range(
    start: int,
    end: int,
    *,
    sleep_seconds: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Draw]:
    """[start, end] 범위 회차를 순회. 실패한 회차는 건너뛴다."""
    draws: list[Draw] = []
    for draw_no in range(start, end + 1):
        try:
            draws.append(fetch_draw_urllib(draw_no, timeout=timeout))
        except FetchError:
            continue
        except Exception:  # noqa: BLE001  - 네트워크 일시 오류도 동일하게 건너뜀
            continue
        if on_progress is not None:
            on_progress(draw_no, end)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return draws


def discover_latest_draw(*, timeout: float = DEFAULT_TIMEOUT, probe_step: int = 64) -> int:
    """현재 시점에서 응답하는 가장 큰 회차 번호를 이분 탐색으로 찾는다.

    Raises:
        FetchError: 1회차마저 응답하지 않을 때 (API 전체 장애).
    """
    # 먼저 1회차를 확인해 API 자체가 살아있는지 본다
    try:
        fetch_draw_urllib(1, timeout=timeout)
    except FetchError as exc:
        raise FetchError("draw 1 unreachable; API may be down") from exc

    low = 1
    high = max(1, low + probe_step)
    while True:
        try:
            fetch_draw_urllib(high, timeout=timeout)
            low = high
            high = high * 2
        except FetchError:
            break
    while low + 1 < high:
        mid = (low + high) // 2
        try:
            fetch_draw_urllib(mid, timeout=timeout)
            low = mid
        except FetchError:
            high = mid
    return low
