# Lotto-ML v2 설계 문서

- **날짜**: 2026-05-19
- **상태**: 설계 완료, 구현 대기
- **선행 문서**: 없음 (v1을 완전히 대체)

## 1. 개요 및 목적

### 1.1 한 줄 정의

주 5,000원(5티켓) 예산으로 한국 로또(6/45)의 5등 적중률을 이론 한계(~11%)에 근접하게 끌어올리는 수학·데이터 기반 추천 시스템.

### 1.2 핵심 철학

- "다음 번호 맞히기"가 아니라 "**5장으로 최대한 많은 3-매치 조합을 커버하기**" 문제로 재정의
- ML은 보조 역할(약한 사전 prior): 45개 번호 중 후보 풀 25~30개 선별
- 메인 엔진은 **수학적 커버링 디자인** (deterministic, 검증 가능)
- 1티켓은 별도 "ML Top-6 꿈" 트랙

### 1.3 측정 가능한 목표 (52주 × 5,000원 = 26만원/년)

| 지표 | v1 현재 | 무작위 5장 | v2 목표 |
|------|---------|-----------|---------|
| 회당 5등 적중률 (hit@3) | ~1.5% | 10.7% | 10~13% |
| 연간 5등 적중 횟수 | ~1회 | ~5.5회 | 5~7회 |
| 회당 4등 적중률 (hit@4) | ~0% | 0.68% | 0.8~1.2% |
| 연간 4등 적중 기대 | ~0회 | ~0.35회 | 0.4~0.6회 |
| 환급률 | ~4% | ~14% | 13~16% |

**중요 정직 노트**:
- v2의 진짜 가치는 "무작위 5장 대비 마법"이 아니라 **"v1의 클러스터링 결함을 수학적으로 제거하는 것"** (자세한 분석은 섹션 3.5 참조)
- 무작위 5장(독립)도 이미 v1보다 훨씬 낫다. v2는 "최소한 무작위만큼은 + 약간의 최적화"가 정직한 목표
- 이론 상한 환급률 50%에는 절대 도달 못함. 13~16%가 합리적 상한
- 표본(주) 수가 적은 초기 1년은 노이즈가 큼. 50주 이상 누적해야 통계적 의미 확보

### 1.4 비목표 (Non-Goals)

- 수익 보장 (수학적으로 불가능)
- 1등/2등 노림 (확률이 너무 낮아 알고리즘 효용 미미)
- 다중 사용자 지원 / 웹 서비스화
- 다른 복권 종류 (연금복권, 스피또 등)
- 베팅 사이즈 자동 조절 (Kelly criterion 등) - 향후 검토

## 2. 아키텍처 및 컴포넌트

### 2.1 패키지 구조

```
lottoml/
├── data/
│   ├── fetch.py        # 동행복권에서 최신 회차 수집 (Playwright + urllib 폴백)
│   └── storage.py      # CSV 기반 영속 계층
├── features/
│   └── build.py        # 약한 ML용 피처 테이블 (~8개)
├── model/
│   ├── train.py        # LightGBM 학습 + Platt calibration
│   └── predict.py      # 45개 번호 확률 출력
├── selection/
│   ├── covering.py     # 4티켓 보장형 covering design (ortools CP-SAT)
│   └── hybrid.py       # 4 covering + 1 dream 조립
├── reporting/
│   ├── weekly.py       # 주간 추천 + 직전 결과 markdown
│   └── history.py      # 누적 적중 통계 markdown
└── cli.py              # 진입점
```

### 2.2 데이터 디렉토리

```
data/
├── draws.csv               # 전체 회차 (번호6 + 보너스 + 등수별 당첨금)
├── portfolio.csv           # 매주 추천 5조합 + 사후 적중 결과 누적
└── models/
    ├── ranker.pkl          # LightGBM + calibrator
    └── ranker_meta.json    # 학습 메타 (회차 범위, 메트릭)
reports/
├── weekly/
│   └── <draw_no>.md        # 회차별 추천 + 결과
└── history.md              # 누적 통계
```

### 2.3 CLI 명령 (5개)

| 명령 | 기능 |
|------|------|
| `lotto setup` | 최초 1회: Playwright 설치, 전체 backfill, 초기 모델 학습 |
| `lotto recommend` | 메인 명령: fetch → 직전 추천 채점 → 다음 회차 추천 |
| `lotto report` | 누적 적중 통계 markdown 생성 |
| `lotto retrain` | 모델 재학습 (10회차마다 권장) |
| `lotto status` | 데이터/모델 상태 표시 |

### 2.4 컴포넌트 책임

| 모듈 | 책임 | 의존 |
|------|------|------|
| data | 동행복권에서 회차 가져와 CSV로 영속 | HTTP, Playwright |
| features | 회차 시계열을 번호별 행으로 펴냄 | data |
| model | 피처를 받아 45개 번호의 확률을 냄 | features |
| selection | 확률 벡터를 받아 최적 5티켓을 냄 | 없음 (순수 수학) |
| reporting | 이력 + 결과를 받아 markdown을 냄 | storage |
| cli | 오케스트레이션 | 각 모듈 |

`selection`은 ML과 무관한 순수 수학 모듈로 분리되어 단독 테스트가 가능하다.

### 2.5 데이터 저장 정책

- **CSV** 사용 (SQLite 아님). 이유: git diff로 변경 확인 가능, 투명성, 단일 사용자.
- `portfolio.csv` 컬럼:
  ```
  draw_no, ticket_idx, n1..n6, source, recommended_at,
  actual_n1..n6, actual_bonus, hits, tier, payout, result_recorded_at
  ```
- 추천 시점에는 `actual_*` 컬럼이 비어있고, 다음 `recommend` 호출 때 자동 채워진다.

### 2.6 의존성

핵심:
- `playwright` - 데이터 수집
- `pandas`, `pyarrow` - 데이터 처리
- `lightgbm` - 단일 ML 모델
- `ortools` - ILP 솔버 (covering design 핵심)
- `click` - CLI

테스트:
- `pytest`, `hypothesis` (covering design property tests)
- `pytest-snapshot` (markdown 출력)
- `pytest-httpx` 또는 `responses` (HTTP mock)

## 3. 핵심 알고리즘 (Selection Engine)

이 시스템의 진짜 가치가 나오는 부분.

### 3.1 입력

- ML 모델이 출력한 45개 번호별 확률 `p[1..45]`
- 설정: `pool_size N=30`, `dream_ticket=True`

### 3.2 단계별 흐름

**1단계: 후보 풀 선정**

```
pool ← p에서 확률 상위 N=30개 번호
```

근거:
- N=30이면 당첨 6개 중 ≥3개가 풀에 포함될 확률 약 91%
- 풀이 작을수록 ML 신호 강도 활용, 클수록 커버리지 안정성
- v1 모델의 약한 신호(near-uniform) 특성에 맞춘 절충점
- `--pool-size` 인자로 조정 가능

**2단계: 4티켓 보장형 선택 (Covering Design)**

문제 정의: 풀 P(|P|=30)에서 4티켓(각 6번호)을 선택해 **기대 hit@3 최대화**.

ILP 정식화 (ortools CP-SAT):

```
변수:
  x_t ∈ {0,1}  for each candidate ticket t
  y_s ∈ {0,1}  for each 3-subset s of P   (C(30,3) = 4060개)

목적:
  maximize Σ_s  w_s * y_s
  w_s = p[s_1] * p[s_2] * p[s_3]   (s가 당첨 후보일 사전 가중)

제약:
  Σ_t x_t = 4
  y_s ≤ Σ_t (x_t · [s ⊆ t])  for each s
  x_t, y_s binary
```

**탐색 공간 축소**:

`C(30,6) = 593,775` 전체 후보를 변수로 두면 ILP가 무거워진다. 대신 candidate ticket pool을 사전 생성한다:

- 가중 샘플링으로 후보 티켓 5,000개 생성 (ML proba 기반)
- ILP는 이 5,000개 중 4개 선택
- 결과: 변수 수 1만 미만, CP-SAT 30초 내 수렴 기대

**3단계: 5번째 꿈 티켓 (ML Top-6)**

```
dream_ticket ← p에서 확률 상위 6개 번호 (전체 45개 기준, 풀 무관)
```

만약 dream_ticket이 1단계 ticket과 정확히 일치하면 ML 확률 7번째 번호로 교체.

**4단계: 출력**

```
portfolio = [coverage_t1, coverage_t2, coverage_t3, coverage_t4, dream_ticket]
```

### 3.3 폴백 (ILP 실패 시)

CP-SAT가 30초 내 해를 못 내면 greedy로 대체:

1. 가장 높은 가중치 후보 티켓 선택
2. 다음은 기존 선택된 티켓들과 3-subset 겹침이 최소인 후보 선택
3. 4티켓 채울 때까지 반복

콘솔에 "covering fallback" 표시.

### 3.4 결정성

`seed = draw_no` 고정. 같은 회차에 두 번 호출 시 동일 결과.

### 3.5 기대 성능 (수학적 검증)

5티켓 strategy별 hit@3 비교 (P(per ticket) = C(6,3)·C(39,3)/C(45,6) = 2.24%):

- **완전 무작위 5티켓**: 1 - (1 - 0.0224)^5 = **10.7%**
- **v1 현재 시스템 (12 ticket 환산, 클러스터링)**: 2.2%
- **v2 (N=30 covering + ML prior + dream)**: 10~13% (추정 범위)

**솔직한 평가**:
v2의 이론적 hit@3 상한은 풀에 ≥3 매치가 들어올 확률 91.5%를 분모로 한 부분 커버리지 곱이며, 약 10~13% 범위에 떨어진다. 이는 무작위 5장(10.7%)과 큰 차이가 없다.

**진짜 가치**:
- v1의 2.2%는 무작위 5장의 10.7%보다도 **낮다** - 12티켓이 모두 같은 hot 번호를 공유해서 함께 빗나가는 클러스터링 때문
- v2의 핵심 가치는 *"앙상블 매직"이 아니라 **클러스터링 제거** + 약간의 ILP 최적화*
- v1 대비 약 5배 개선의 대부분은 "수학적으로 합리적인 분산"에서 나옴
- ILP는 hit@4 적중률에서 +0.3~0.5pp 추가 기여 (covering이 잘 펴주면 4매치가 한 티켓에 몰릴 확률↑)

**검증**: M5 단계에서 최근 50회차 백테스트로 실측 검증한다.

## 4. 데이터·피처·모델 (Weak ML Layer)

철학: **신호가 거의 없다는 것을 인정하고, 단순함으로 안전하게.**

### 4.1 데이터 스키마 (`data/draws.csv`)

```
draw_no, draw_date, n1, n2, n3, n4, n5, n6, bonus,
total_sales, first_prize, first_winners,
second_prize, second_winners, third_prize, third_winners
```

v1 스키마 그대로 (검증된 형식). v1의 raw/processed 분리 단계는 폐기 - 단일 파일로 통합.

### 4.2 피처 설계 - v1의 114개 → 8개

근거: v1의 ECE 0.0008은 "모델이 균등분포에 수렴"했다는 신호. 피처가 많을수록 노이즈 적합. 물리적으로 의미 있는 피처만 사용.

| 피처 | 의미 | 정당화 |
|------|------|--------|
| `freq_20` | 직전 20회차 출현 횟수 | 단기 hot/cold |
| `freq_50` | 직전 50회차 출현 횟수 | 중기 추세 |
| `freq_100` | 직전 100회차 출현 횟수 | 장기 평균 수렴 검증 |
| `last_seen_gap` | 마지막 출현 후 경과 회차 | recency |
| `bonus_freq_50` | 직전 50회차 보너스 출현 횟수 | 보너스 신호 |
| `bucket` | 번호 구간 (1=1-15, 2=16-30, 3=31-45) | 구조적 분포 |
| `is_odd` | 홀짝 | 구조적 분포 |
| `mod_10` | 끝자리 0~9 | 구조적 분포 |

**v1에서 제거**:
- 추세 비율 (`trend_ratio_*`) - 평균 회귀 노이즈
- 공출현 (`cooc_prev_*`) - 통계적 무의미
- 페어 누적 (`pair_with_prev_*`) - 동일 사유
- EWM 가중 - freq_N과 중복
- 거리/평균갭 (`distance_to_recent_mean_*`) - 노이즈
- 요일/월 - 항상 토요일 추첨이므로 무의미

### 4.3 모델: 단일 LightGBM

```python
LGBMClassifier(
    n_estimators=300,
    num_leaves=15,           # overfit 방지
    min_data_in_leaf=50,     # regularization
    learning_rate=0.03,
    reg_alpha=0.1,           # L1
    reg_lambda=0.5,          # L2
    objective="binary",
    random_state=42,
)
```

**단일 모델 선택 이유**:
- v1의 lightgbm/gbdt/xgboost가 사실상 동등 성능
- 앙상블이 ECE만 살짝 개선, hit@3에는 거의 영향 없음
- 디버깅·재학습·유지보수 비용 대폭 절감
- 핵심 가치는 selection layer에서 발생

**v1에서 제거하는 모델**: logreg, mlp, randomforest, extratrees, gbdt, xgboost, catboost, ensemble (모두 미사용)

### 4.4 학습 전략

- **시간 순 분할**: 마지막 50회차를 holdout으로 영구 분리
- **샘플 가중치**: `decay=0.997` (v1 0.998보다 약간 빠른 망각)
- **Calibration**: Platt scaling 우선 (단순함), 캘리브레이션 곡선이 비단조적이면 isotonic regression 폴백. v1은 raw output 사용했으나 v2는 보정하여 정직한 확률 제공
- **재학습 주기**: 10회차마다 (`lotto retrain` 수동 트리거)

### 4.5 검증 메트릭

| 메트릭 | 의미 | 목표 |
|--------|------|------|
| Brier | 확률 정확도 | < 0.13 |
| ECE | 캘리브레이션 오차 | < 0.01 |
| Top-6 hit rate | top-6에 진짜 번호 1개 이상 | > 0.55 |

모델 메트릭은 "모델이 망가지지 않았는지" 확인용. 시스템 가치는 selection layer에서 발생하므로 모델 메트릭에 과도하게 집착하지 않음.

### 4.6 v1과의 비교

| 항목 | v1 | v2 |
|------|-----|-----|
| 피처 수 | 114 | 8 |
| 모델 수 | 7 + ensemble | 1 |
| 학습 시간 | ~5분 | <30초 |
| Calibration | 없음 | Platt/isotonic |
| 코드 라인 (학습) | ~940줄 | ~150줄 예상 |

## 5. CLI / UX / 리포팅

### 5.1 `lotto recommend` - 메인 명령

콘솔 출력 예시:

```
[lotto] 최신 회차 확인 중... 1224회 새로 수집됨 (2026-05-17 추첨)

📊 지난 회차(1224) 결과
   당첨번호: 12 18 24 33 35 41 (보너스 7)

   1번 (covering): 5  12 18 23 35 41 → 4개 적중 ✨ 4등 (+50,000원)
   2번 (covering): 8  16 28 33 38 44 → 1개 적중
   3번 (covering): 11 19 24 30 37 42 → 2개 적중
   4번 (covering): 3  14 21 25 33 45 → 2개 적중
   5번 (dream)   : 7  15 22 28 31 40 → 0개 적중

   회차 손익: +45,000원 (지출 5,000 / 환급 50,000)

🎯 다음 회차(1225) 추천 - 2026-05-24 추첨
   1번 (covering): 5  11 17 26 32 39
   2번 (covering): 8  14 22 28 35 42
   3번 (covering): 3  19 25 31 37 44
   4번 (covering): 12 16 24 29 33 40
   5번 (dream)   : 7  18 23 30 36 41

   📝 reports/weekly/1225.md 저장됨
```

**동작 흐름**:

1. 마지막 fetch 회차와 현재 날짜 비교
2. 새 회차가 있으면 fetch (Playwright + urllib 폴백)
3. 미평가 추천이 있으면 자동 채점 (portfolio.csv 갱신)
4. 다음 회차 추천 생성 (ML predict → selection)
5. 콘솔 출력 + markdown 작성 + portfolio.csv append

### 5.2 `lotto report` - 누적 통계

```
[lotto] 누적 통계 (1224~1234회, 11주)

총 베팅: 55,000원  |  총 환급: 15,000원  |  순손익: -40,000원
환급률: 27.3%      |  이론 상한: 50%      |  v1 환급률: ~4%

등수별 적중
   1등: 0회   2등: 0회   3등: 0회   4등: 0회   5등: 3회

회당 평균 best_hits: 2.2개
hit@3 누적 적중률: 27.3% (3/11)  ← 목표 10~13% (표본 부족 노이즈)

ℹ️  표본 11주는 통계적으로 노이즈가 매우 큽니다. 50주 이상 누적 시 신뢰 가능.
   기대 환급률 13~16% 도달까지 약 6개월~1년 누적 필요.

📝 reports/history.md 갱신됨
```

### 5.3 `lotto setup` - 최초 1회

```
[lotto] 초기 설정 시작

✓ Python 의존성 확인
✓ Playwright Chromium 설치 (1.2GB, ~30초)
⏳ 1~1223회 backfill 수집 중... (4분 예상)
✓ 모델 학습 (LightGBM, decay=0.997)
   - Brier: 0.117  ECE: 0.008  Top6 hit: 0.62
✓ data/models/ranker.pkl 저장

준비 완료. `lotto recommend` 명령으로 시작하세요.
```

### 5.4 `lotto retrain` & `lotto status`

```
$ lotto status
데이터:     1~1234회 (마지막 fetch: 2026-05-17)
모델:       2026-04-12 학습, 1180회까지 사용 (54회차 경과)
다음 추첨:  2026-05-24 (토)
미평가 추천: 없음

ℹ️  모델 학습 후 50회차 경과 - 재학습 권장 (`lotto retrain`)
```

```
$ lotto retrain
[lotto] 모델 재학습 (1~1184회 학습 / 1185~1234회 holdout)
   - Brier: 0.117 (이전 0.117)
   - ECE:   0.009 (이전 0.008)
   - Top6:  0.61  (이전 0.62)

새 모델로 교체하시겠습니까? [Y/n]
```

### 5.5 Markdown 리포트 포맷

**`reports/weekly/1225.md`**:

```markdown
# 1225회 추천 (2026-05-24 추첨)

## 추천 포트폴리오

| 티켓 | 번호 | 출처 |
|------|------|------|
| 1 | 5 11 17 26 32 39 | covering |
| 2 | 8 14 22 28 35 42 | covering |
| 3 | 3 19 25 31 37 44 | covering |
| 4 | 12 16 24 29 33 40 | covering |
| 5 | 7 18 23 30 36 41 | dream |

## ML 확률 Top-10
| 순위 | 번호 | 확률 |
| 1 | 35 | 0.162 |
| ... |

## 후보 풀 (30번호)
3 5 7 8 11 12 14 16 17 18 19 22 23 24 25 26 28 29 30 31 32 33 35 36 37 39 40 41 42 44

## 결과 (추첨 후 자동 갱신)
- **당첨번호**: TBD
- **포트폴리오 best hits**: TBD
- **환급**: TBD원
```

**`reports/history.md`**: `lotto report` 출력 내용 + 회차별 시계열 표.

### 5.6 에러/엣지 처리

| 상황 | 동작 |
|------|------|
| fetch 실패 (네트워크) | 캐시 데이터로 추천 + 경고. recommend 진행 |
| 모델 없음 | "`lotto setup`을 먼저 실행하세요" 안내 후 종료 |
| 같은 회차 재호출 | 기존 추천 반환 (seed 고정) |
| 추첨 전 채점 시도 | 스킵, "아직 추첨 전" 표시 |
| ILP 30초 타임아웃 | greedy 폴백, 콘솔에 표시 |

### 5.7 v1과의 UX 차이

| 항목 | v1 | v2 |
|------|-----|-----|
| 명령 수 | 8 스크립트, 인자 30+개 | 5 명령, 인자 거의 없음 |
| 주간 운용 | fetch → prepare → features → train → predict → record (수동) | `lotto recommend` 1회 |
| 누적 통계 | 없음 (백테스트만) | 자동 history.md |
| 메시지 언어 | 한국어/영어 혼재 | 한국어 일관 |

## 6. 테스트 전략 + 개발 마일스톤

### 6.1 테스트 전략

**`selection/` - 가장 깊게 (시스템 핵심)**:
- Unit tests: covering 함수가 결정적·재현 가능
- Property-based (hypothesis):
  - 출력 티켓은 항상 6개 고유 번호 + 1..45 범위
  - 동일 입력에 대해 동일 출력 (seed 고정)
  - 작은 풀(N=8, 2티켓)에서 ILP 해와 brute-force 해 일치
- 성능: N=30, 4티켓 ILP 30초 내 완료

**`data/` - 통합 위주**:
- HTTP mock으로 fetch 응답 파싱 검증
- storage 라운드트립 일관성
- 회차 결손/중복 감지

**`model/` - smoke 위주**:
- 학습 완료, 확률이 [0,1], 45개 모두 산출
- Brier/ECE가 명시 임계 내 (회귀 방지)

**`reporting/` - snapshot 위주**:
- 고정 입력에 대해 markdown 출력이 정확히 일치

**`cli.py` - end-to-end smoke**:
- temp 디렉토리에서 setup → recommend → report 시퀀스 무에러
- 회차 결과 평가가 portfolio.csv에 반영

### 6.2 커버리지 목표

- `selection/`: 90%+
- `data/`, `reporting/`: 80%+
- `model/`: 60%+ (smoke 중심)
- 전체: 80%+

### 6.3 개발 마일스톤 (5단계)

**M1. 스캐폴딩 + 데이터 레이어**:
- 디렉토리 구조, `pyproject.toml`, 의존성
- `data/fetch.py` (Playwright + urllib 폴백)
- `data/storage.py` (draws.csv read/write)
- `lotto setup` 명령 (backfill까지)
- 완료 조건: `lotto setup`이 1~최신회차 backfill 성공

**M2. 피처 + 모델 레이어**:
- `features/build.py` (8개 피처)
- `model/train.py` (LightGBM + Platt calibration)
- `model/predict.py` (45개 확률)
- `lotto retrain` 명령
- 완료 조건: holdout Brier < 0.13

**M3. Selection 엔진**:
- `selection/covering.py` (ortools CP-SAT)
- `selection/hybrid.py` (4 covering + 1 dream)
- Greedy 폴백
- Property-based 테스트 풀세트
- 완료 조건: 시뮬레이션 hit@3 ≥ 10% (무작위 5장 baseline 10.7%에 근접)

**M4. CLI + 리포팅**:
- `lotto recommend` (fetch → 채점 → 추천)
- `lotto report` (history.md)
- `lotto status`
- 주간 markdown 자동 생성
- 완료 조건: fresh run에서 콘솔 + markdown + portfolio.csv 정상

**M5. 폴리시 + 실사용 검증**:
- 에러 처리, README (한국어)
- 회귀 백테스트: 최근 50회차에서 v1, 무작위 5장, v2 hit@3 3-way 비교
- 첫 실 회차 추천 수행
- 완료 조건:
  - 백테스트 hit@3가 v1(2.2%) 대비 4배 이상 + 무작위 5장(10.7%) 이상
  - 실제 다음 회차 추천 정상 산출

### 6.4 마이그레이션 전략 (v1 → v2)

- v1 코드는 git history에 그대로 보존
- v2는 `v2` 브랜치에서 작업 → 완성 후 main 병합
- v1의 `data/processed/lotto_draws.csv`를 v2 `data/draws.csv` 시드로 재사용
- v1 `models/`, `reports/`는 비교용으로 보존, 신규는 새 경로

### 6.5 비범위 (v2에 포함하지 않음)

- 웹 UI / 모바일 / API 서버
- 다중 사용자
- 알림(슬랙/이메일) - 향후 검토
- 다른 복권 종류
- 베팅 사이즈 자동 조절 (Kelly criterion)
- 보너스 번호 별도 모델 (효과 미미 판정)

## 7. 위험과 완화

| 위험 | 영향 | 완화 |
|------|------|------|
| ILP가 30초 내 수렴 안 함 | recommend 지연 | greedy 폴백 자동 발동 |
| 동행복권 응답 형식 변경 | fetch 실패 | urllib 폴백 + 캐시 사용 + 에러 메시지 |
| 표본(주) 부족으로 hit@3 추정 노이즈 | 잘못된 시스템 평가 | history.md에 신뢰 구간 경고 |
| 모델 자체에 신호가 없음 | 선택 효과 미미 | 시스템 가치를 selection에 집중. 모델은 보조 |
| 한국 로또 규칙 변경 | 시스템 가정 붕괴 | 발생 시 즉시 점검 (가능성 낮음) |

## 8. 향후 확장 (v3 이상 후보)

- 알림 통합 (슬랙/이메일/카카오)
- 베팅 사이즈 적응 (조건부 운용)
- 다른 복권 (연금복권 720+)
- 웹 대시보드 (Streamlit)
