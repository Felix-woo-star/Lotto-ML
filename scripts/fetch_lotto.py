#!/usr/bin/env python3
"""동행복권 로또 당첨 번호를 수집한다.

기본 수집 엔진은 Playwright(실브라우저)이며, 필요 시 urllib 엔진을 사용할 수 있다.
"""

import argparse
import csv
import gzip
import importlib
import json
import os
import re
import time
import zlib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URLS = [
    "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}",
    "https://dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}",
    "http://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}",
    "http://dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}",
]
DRAW_PAGE_URL = "https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo={draw_no}"
LT645_API_URL = (
    "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
    "?srchDir=center&srchLtEpsd={draw_no}"
)
HOME_URL = "https://www.dhlottery.co.kr/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FIELDS = [
    "draw_no",
    "draw_date",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "bonus",
    "total_sell_amount",
    "first_prize_amount",
    "first_prize_winner_count",
    "first_accum_amount",
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Fetch Lotto winning numbers.")
    parser.add_argument("--start", type=int, default=1, help="시작 회차 번호.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="종료 회차 번호. 생략 시 최신 회차까지 수집.",
    )
    parser.add_argument(
        "--out-raw",
        default="data/raw/lotto_draws.csv",
        help="출력 CSV 경로.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="요청 간 대기 시간(초).",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="요청 타임아웃(초).")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="실패 시 재시도 횟수.",
    )
    parser.add_argument(
        "--engine",
        choices=["playwright", "urllib"],
        default="playwright",
        help="수집 엔진 선택(기본: playwright).",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Playwright 헤드리스 모드 사용 여부(기본: true).",
    )
    return parser.parse_args()


def resolve_project_path(path: str) -> str:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환한다."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def read_existing_draws(path: str) -> tuple[set[int], int]:
    """기존 CSV에서 회차 목록과 최대 회차를 읽어온다."""
    existing = set()
    max_no = 0
    if not os.path.exists(path):
        return existing, max_no

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return existing, max_no

        for row in reader:
            value = row.get("draw_no")
            if not value:
                continue
            try:
                draw_no = int(value)
            except ValueError:
                continue
            existing.add(draw_no)
            if draw_no > max_no:
                max_no = draw_no
    return existing, max_no


def to_int(value: object) -> int:
    """값을 안전하게 int로 변환한다."""
    return int(value) if value is not None and value != "" else 0


def normalize_record(data: dict) -> dict:
    """수집된 데이터를 CSV 스키마에 맞게 정규화한다."""
    return {
        "draw_no": to_int(data.get("drwNo")),
        "draw_date": data.get("drwNoDate", ""),
        "n1": to_int(data.get("drwtNo1")),
        "n2": to_int(data.get("drwtNo2")),
        "n3": to_int(data.get("drwtNo3")),
        "n4": to_int(data.get("drwtNo4")),
        "n5": to_int(data.get("drwtNo5")),
        "n6": to_int(data.get("drwtNo6")),
        "bonus": to_int(data.get("bnusNo")),
        "total_sell_amount": to_int(data.get("totSellamnt")),
        "first_prize_amount": to_int(data.get("firstWinamnt")),
        "first_prize_winner_count": to_int(data.get("firstPrzwnerCo")),
        "first_accum_amount": to_int(data.get("firstAccumamnt")),
    }


def format_ymd(ymd: str) -> str:
    """YYYYMMDD 문자열을 YYYY-MM-DD로 변환한다."""
    text = str(ymd or "")
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return ""


def normalize_lt645_row(row: dict) -> dict:
    """신규 lt645 API 행 데이터를 공통 스키마로 변환한다."""
    return {
        "returnValue": "success",
        "drwNo": to_int(row.get("ltEpsd")),
        "drwNoDate": format_ymd(str(row.get("ltRflYmd", ""))),
        "drwtNo1": to_int(row.get("tm1WnNo")),
        "drwtNo2": to_int(row.get("tm2WnNo")),
        "drwtNo3": to_int(row.get("tm3WnNo")),
        "drwtNo4": to_int(row.get("tm4WnNo")),
        "drwtNo5": to_int(row.get("tm5WnNo")),
        "drwtNo6": to_int(row.get("tm6WnNo")),
        "bnusNo": to_int(row.get("bnsWnNo")),
        "totSellamnt": to_int(row.get("wholEpsdSumNtslAmt")),
        "firstWinamnt": to_int(row.get("rnk1WnAmt")),
        "firstPrzwnerCo": to_int(row.get("rnk1WnNope")),
        "firstAccumamnt": to_int(row.get("rnk1SumWnAmt")),
    }


def extract_lt645_draw(text: str, draw_no: int) -> dict | None:
    """신규 lt645 API 응답 텍스트에서 특정 회차 데이터를 추출한다."""
    payload = json.loads(text)
    rows = ((payload.get("data") or {}).get("list")) or []
    for row in rows:
        if to_int(row.get("ltEpsd")) == draw_no:
            return normalize_lt645_row(row)
    return None


def parse_payload(payload: str) -> dict:
    """응답 본문에서 JSON 객체를 추출해 파싱한다."""
    decoder = json.JSONDecoder()
    text = payload.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    objects: list[dict] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = max(end, start + 1)
        except json.JSONDecodeError:
            idx = start + 1

    for obj in objects:
        if "returnValue" in obj:
            return obj
    for obj in objects:
        if "drwNo" in obj and all(f"drwtNo{i}" in obj for i in range(1, 7)):
            return obj
    if objects:
        return objects[0]
    raise json.JSONDecodeError("Invalid payload", text, 0)


def is_soft_no_data_error(err: Exception) -> bool:
    """회차 미발표로 볼 수 있는 비치명 응답 오류인지 판별한다."""
    if isinstance(err, json.JSONDecodeError):
        return True
    message = str(err).lower()
    patterns = (
        "invalid payload",
        "expecting value",
        "extra data",
        "unexpected response format",
    )
    return any(pattern in message for pattern in patterns)


def decode_response_bytes(content: bytes, charset: str | None) -> str:
    """응답 바이트를 가능한 인코딩으로 디코딩한다."""
    candidates = []
    if charset:
        candidates.append(charset)
    candidates.extend(["utf-8", "cp949", "euc-kr"])

    tried: set[str] = set()
    for encoding in candidates:
        if encoding in tried:
            continue
        tried.add(encoding)
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def decode_http_text(content: bytes, headers) -> str:
    """HTTP 응답 바이트를 압축/인코딩을 고려해 텍스트로 변환한다."""
    encoding = str(headers.get("Content-Encoding", "")).lower()
    if "gzip" in encoding:
        try:
            content = gzip.decompress(content)
        except OSError:
            pass
    elif "deflate" in encoding:
        try:
            content = zlib.decompress(content)
        except zlib.error:
            pass

    charset = None
    get_charset = getattr(headers, "get_content_charset", None)
    if callable(get_charset):
        charset = get_charset()
    return decode_response_bytes(content, charset)


def parse_draw_page(html: str, requested_draw_no: int) -> dict | None:
    """결과 페이지 HTML에서 회차 정보를 추출한다."""
    js_draw_no = re.search(r"drwNo\s*[:=]\s*['\"]?(\d{1,5})['\"]?", html)
    js_date = re.search(
        r"drwNoDate\s*[:=]\s*['\"]?(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})['\"]?",
        html,
    )
    js_numbers = [
        re.search(rf"drwtNo{i}\s*[:=]\s*['\"]?(\d{{1,2}})['\"]?", html)
        for i in range(1, 7)
    ]
    js_bonus = re.search(r"bnusNo\s*[:=]\s*['\"]?(\d{1,2})['\"]?", html)
    if (
        js_draw_no
        and js_date
        and all(match is not None for match in js_numbers)
        and js_bonus
    ):
        page_draw_no = int(js_draw_no.group(1))
        if page_draw_no != requested_draw_no:
            return None
        year, month, day = js_date.groups()
        return {
            "returnValue": "success",
            "drwNo": page_draw_no,
            "drwNoDate": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
            "drwtNo1": int(js_numbers[0].group(1)),
            "drwtNo2": int(js_numbers[1].group(1)),
            "drwtNo3": int(js_numbers[2].group(1)),
            "drwtNo4": int(js_numbers[3].group(1)),
            "drwtNo5": int(js_numbers[4].group(1)),
            "drwtNo6": int(js_numbers[5].group(1)),
            "bnusNo": int(js_bonus.group(1)),
            "totSellamnt": 0,
            "firstWinamnt": 0,
            "firstPrzwnerCo": 0,
            "firstAccumamnt": 0,
        }

    draw_no_match = re.search(r"제\s*(\d{1,5})\s*회", html)
    page_draw_no = int(draw_no_match.group(1)) if draw_no_match else requested_draw_no
    if page_draw_no != requested_draw_no:
        return None

    date_match = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", html)
    draw_date = ""
    if date_match:
        year, month, day = date_match.groups()
        draw_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    number_candidates = [
        int(value) for value in re.findall(r"ball_645[^>]*>\s*(\d{1,2})\s*<", html)
    ]
    numbers: list[int] = []
    for value in number_candidates:
        if 1 <= value <= 45:
            numbers.append(value)
        if len(numbers) >= 7:
            break
    if len(numbers) < 7:
        return None

    return {
        "returnValue": "success",
        "drwNo": page_draw_no,
        "drwNoDate": draw_date,
        "drwtNo1": numbers[0],
        "drwtNo2": numbers[1],
        "drwtNo3": numbers[2],
        "drwtNo4": numbers[3],
        "drwtNo5": numbers[4],
        "drwtNo6": numbers[5],
        "bnusNo": numbers[6],
        "totSellamnt": 0,
        "firstWinamnt": 0,
        "firstPrzwnerCo": 0,
        "firstAccumamnt": 0,
    }


def fetch_draw_from_lt645_api(draw_no: int, timeout: float, max_retries: int) -> dict | None:
    """신규 lt645 JSON API로 회차 데이터를 조회한다."""
    url = LT645_API_URL.format(draw_no=draw_no)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Encoding": "gzip, deflate",
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Referer": f"https://www.dhlottery.co.kr/lt645/result?ltEpsd={draw_no}",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                text = decode_http_text(response.read(), response.headers)
            return extract_lt645_draw(text, draw_no)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as err:
            last_error = err
            time.sleep(0.5 * (attempt + 1))

    if last_error is not None:
        if is_soft_no_data_error(last_error):
            return None
        raise RuntimeError(f"Failed to fetch lt645 api draw {draw_no}: {last_error}")
    return None


def fetch_draw_via_urllib(draw_no: int, timeout: float, max_retries: int) -> dict | None:
    """urllib 엔진으로 회차 데이터를 조회한다."""
    data = fetch_draw_from_lt645_api(draw_no, timeout, max_retries)
    if data is not None:
        return data

    urls = [template.format(draw_no=draw_no) for template in API_URLS]
    last_error: Exception | None = None

    for attempt in range(max_retries):
        for url in urls:
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": DEFAULT_USER_AGENT,
                        "Accept": "application/json, text/plain, */*",
                        "Accept-Encoding": "gzip, deflate",
                        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Referer": "https://www.dhlottery.co.kr/gameResult.do?method=byWin",
                    },
                )
                with urlopen(request, timeout=timeout) as response:
                    payload = decode_http_text(response.read(), response.headers)
                data = parse_payload(payload)

                if not data:
                    continue
                return_value = str(data.get("returnValue", "")).lower()
                if return_value == "success":
                    response_draw_no = to_int(data.get("drwNo"))
                    if response_draw_no and response_draw_no != draw_no:
                        raise RuntimeError(
                            f"Unexpected draw number in response: "
                            f"requested={draw_no}, response={response_draw_no}"
                        )
                    return data
                if return_value in {"fail", ""}:
                    continue
                raise RuntimeError(
                    f"Unexpected response format for draw {draw_no}: "
                    f"returnValue={data.get('returnValue')!r}, keys={list(data.keys())[:8]}"
                )
            except (
                HTTPError,
                URLError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as err:
                last_error = err
        time.sleep(0.5 * (attempt + 1))

    if last_error is not None:
        if is_soft_no_data_error(last_error):
            return None
        raise RuntimeError(f"Failed to fetch draw {draw_no}: {last_error}")
    return None


class PlaywrightFetcher:
    """Playwright 브라우저 세션으로 회차 데이터를 조회한다."""

    def __init__(self, headless: bool, timeout: float):
        self.headless = headless
        self.timeout = timeout
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._timeout_error = None

    def __enter__(self):
        try:
            sync_api = importlib.import_module("playwright.sync_api")
        except ImportError as err:
            raise RuntimeError(
                "playwright 패키지가 없습니다. "
                "먼저 `uv add playwright` 및 "
                "`uv run playwright install chromium`를 실행하세요."
            ) from err

        sync_playwright = getattr(sync_api, "sync_playwright", None)
        playwright_timeout_error = getattr(sync_api, "TimeoutError", TimeoutError)
        if sync_playwright is None:
            raise RuntimeError("playwright.sync_api.sync_playwright를 찾을 수 없습니다.")

        self._timeout_error = playwright_timeout_error
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            locale="ko-KR",
            extra_http_headers={
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )
        self._page = self._context.new_page()
        self._page.goto(HOME_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    @property
    def timeout_ms(self) -> int:
        return int(max(self.timeout, 1.0) * 1000)

    def fetch_draw(self, draw_no: int, max_retries: int) -> dict | None:
        """Playwright로 회차 데이터를 조회한다."""
        url = DRAW_PAGE_URL.format(draw_no=draw_no)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                api_text = self._page.evaluate(
                    """
                    async (drawNo) => {
                      try {
                        const url = `/lt645/selectPstLt645InfoNew.do?srchDir=center&srchLtEpsd=${drawNo}`;
                        const resp = await fetch(url, { credentials: "include" });
                        return await resp.text();
                      } catch (e) {
                        return "";
                      }
                    }
                    """,
                    draw_no,
                )
                if api_text:
                    try:
                        data = extract_lt645_draw(api_text, draw_no)
                    except (json.JSONDecodeError, ValueError) as err:
                        last_error = err
                        data = None
                    if data is not None:
                        return data

                self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                self._page.wait_for_timeout(350)
                html = self._page.content()
                data = parse_draw_page(html, draw_no)
                if data is not None:
                    return data

                # 페이지 파싱 실패 시 브라우저 세션으로 API를 재호출한다.
                legacy_text = self._page.evaluate(
                    """
                    async (drawNo) => {
                      try {
                        const resp = await fetch(
                          `/common.do?method=getLottoNumber&drwNo=${drawNo}`,
                          { credentials: "include" }
                        );
                        return await resp.text();
                      } catch (e) {
                        return "";
                      }
                    }
                    """,
                    draw_no,
                )
                if legacy_text:
                    try:
                        payload = parse_payload(legacy_text)
                    except (json.JSONDecodeError, ValueError) as err:
                        last_error = err
                        payload = None
                    if payload is None:
                        return None
                    return_value = str(payload.get("returnValue", "")).lower()
                    if return_value == "success":
                        if to_int(payload.get("drwNo")) == draw_no:
                            return payload
                    elif return_value in {"fail", ""}:
                        return None
                return None
            except self._timeout_error as err:
                last_error = err
            except Exception as err:  # noqa: BLE001
                last_error = err
            time.sleep(0.5 * (attempt + 1))

        if last_error is not None:
            if is_soft_no_data_error(last_error):
                return None
            raise RuntimeError(f"Failed to fetch draw {draw_no} with playwright: {last_error}")
        return None


def build_fetch_fn(args: argparse.Namespace):
    """엔진 설정에 맞는 fetch 함수를 반환한다."""
    if args.engine == "urllib":
        return None, lambda draw_no: fetch_draw_via_urllib(
            draw_no, args.timeout, args.max_retries
        )

    fetcher = PlaywrightFetcher(headless=args.headless, timeout=args.timeout)
    context = fetcher.__enter__()
    return context, lambda draw_no: fetcher.fetch_draw(draw_no, args.max_retries)


def main() -> int:
    """CLI 실행 진입점."""
    args = parse_args()
    args.out_raw = resolve_project_path(args.out_raw)

    existing, max_existing = read_existing_draws(args.out_raw)
    start = args.start
    if existing and start <= max_existing:
        start = max_existing + 1
        print(f"Existing data up to draw {max_existing}; starting from draw {start}.")

    if args.end is not None and start > args.end:
        print("No new draws to fetch.")
        return 0

    os.makedirs(os.path.dirname(args.out_raw) or ".", exist_ok=True)
    file_exists = os.path.exists(args.out_raw)
    needs_header = not file_exists or os.path.getsize(args.out_raw) == 0

    fetch_context = None
    fetch_fn = None
    fetched = 0

    try:
        fetch_context, fetch_fn = build_fetch_fn(args)
        with open(args.out_raw, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            if needs_header:
                writer.writeheader()

            draw_no = start
            while True:
                if args.end is not None and draw_no > args.end:
                    break
                if draw_no in existing:
                    draw_no += 1
                    continue

                try:
                    data = fetch_fn(draw_no)
                except RuntimeError as err:
                    if is_soft_no_data_error(err):
                        data = None
                    else:
                        print(
                            f"Warning: fetch failed for draw {draw_no}: {err}"
                        )
                        print(
                            "Stopping safely without appending new rows. "
                            "Check network/WAF and try again."
                        )
                        break
                if data is None:
                    if args.end is None:
                        if draw_no > 1:
                            known = fetch_fn(draw_no - 1)
                            if known is None:
                                print(
                                    "Warning: data source validation failed. "
                                    f"draw {draw_no - 1} could not be verified."
                                )
                                print(
                                    "Stopping safely without appending new rows. "
                                    "Check network/WAF and try again."
                                )
                                break
                        print(f"No data for draw {draw_no}; assuming latest draw.")
                    else:
                        print(f"No data for draw {draw_no}; stopping.")
                    break

                writer.writerow(normalize_record(data))
                fetched += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
                draw_no += 1
    finally:
        if fetch_context is not None:
            fetch_context.__exit__(None, None, None)

    print(f"Fetched {fetched} new draws into {args.out_raw}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
