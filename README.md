# Lotto-ML v2

주 5,000원(5티켓) 예산 한국 로또(6/45) 추천 시스템. 4 covering + 1 dream 하이브리드.

> v1 문서는 [README_v1.md](README_v1.md)를 참고하세요.

## 설계 철학

- **"다음 번호 맞히기"가 아님.** 5장으로 최대한 많은 3-매치 조합을 커버하는 문제.
- 메인 엔진: ortools CP-SAT 기반 covering design ILP (4티켓)
- 보조 엔진: 단일 LightGBM 약한 prior (45개 번호 확률)
- 1티켓: ML Top-6 꿈 트랙

자세한 설계는 [docs/superpowers/specs/2026-05-19-lotto-v2-design.md](docs/superpowers/specs/2026-05-19-lotto-v2-design.md).

## 설치

```bash
uv sync --all-extras
uv run playwright install chromium
```

## 사용법

```bash
# 최초 1회
uv run lotto setup
# → v1 데이터 마이그레이션 + 모델 학습

# 매주 토요일
uv run lotto recommend
# → 직전 회차 채점 + 5조합 추천

# 가끔
uv run lotto report   # 누적 통계
uv run lotto retrain  # 모델 재학습 (10회차마다)
uv run lotto status   # 현재 상태
```

## 출력 위치

- `data/draws.csv` - 전체 회차
- `data/portfolio.csv` - 추천 + 사후 결과
- `data/models/ranker.pkl` - 학습된 모델
- `reports/weekly/<회차>.md` - 회차별 리포트
- `reports/history.md` - 누적 통계
- `reports/v2_backtest.md` - 회귀 백테스트 결과

## 기대 성능

| 지표 | v1 | 무작위 5장 | v2 목표 |
|------|----|-----------|---------|
| 회당 5등 적중률 | 1.5% | 10.7% | 10~13% |
| 환급률 | 4% | 14% | 13~16% |

여전히 마이너스(이론 상한 -50%). 핵심 가치는 "v1의 클러스터링 결함 제거".

## 개발

```bash
uv run pytest                              # 전체 테스트
uv run pytest tests/lottoml/selection -v   # selection 모듈만
```
