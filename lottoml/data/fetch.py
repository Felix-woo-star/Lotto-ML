"""동행복권 회차 데이터 수집."""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from typing import Callable

from .types import Draw

# 신규 lt645 API. 구 common.do?method=getLottoNumber 엔드포인트는 2025년경
# 폐기되어 HTML 리다이렉트로 응답한다.
LT645_API_URL = (
    "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
    "?srchDir=center&srchLtEpsd={draw_no}"
)
DEFAULT_TIMEOUT = 10.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class FetchError(RuntimeError):
    """동행복권 응답이 비정상이거나 회차가 존재하지 않을 때."""


def fetch_draw_urllib(draw_no: int, *, timeout: float = DEFAULT_TIMEOUT) -> Draw:
    """urllib + lt645 API 기반 단일 회차 조회.

    lt645 API는 요청한 회차를 포함한 최근 여러 회차를 리스트로 반환하므로
    `ltEpsd == draw_no` 항목을 찾아 사용한다.
    """
    url = LT645_API_URL.format(draw_no=draw_no)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": f"https://www.dhlottery.co.kr/lt645/result?ltEpsd={draw_no}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchError(f"회차 {draw_no} 비정상 응답 (JSON 아님)") from exc

    rows = (payload or {}).get("data", {}).get("list") or []
    row = next((r for r in rows if int(r.get("ltEpsd", 0)) == draw_no), None)
    if row is None:
        raise FetchError(f"회차 {draw_no} 응답에 해당 회차가 없음")

    return Draw(
        draw_no=int(row["ltEpsd"]),
        draw_date=_parse_lt645_date(str(row["ltRflYmd"])),
        numbers=(
            int(row["tm1WnNo"]), int(row["tm2WnNo"]),
            int(row["tm3WnNo"]), int(row["tm4WnNo"]),
            int(row["tm5WnNo"]), int(row["tm6WnNo"]),
        ),
        bonus=int(row["bnsWnNo"]),
        total_sales=int(row.get("rlvtEpsdSumNtslAmt", 0) or 0),
        first_prize=int(row.get("rnk1WnAmt", 0) or 0),
        first_winners=int(row.get("rnk1WnNope", 0) or 0),
        second_prize=int(row.get("rnk2WnAmt", 0) or 0),
        second_winners=int(row.get("rnk2WnNope", 0) or 0),
        third_prize=int(row.get("rnk3WnAmt", 0) or 0),
        third_winners=int(row.get("rnk3WnNope", 0) or 0),
    )


def _parse_lt645_date(ymd: str) -> dt.date:
    """'20260516' → date(2026, 5, 16)."""
    if len(ymd) != 8 or not ymd.isdigit():
        raise FetchError(f"잘못된 날짜 형식: {ymd!r}")
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


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
