# Lotto-ML

대한민국 로또 당첨 번호를 대상으로 한 머신러닝 실험 프로젝트입니다.

## 데이터 파이프라인

원천 데이터는 동행복권 공식 엔드포인트에서 수집하여 CSV로 저장합니다.
가공 데이터는 검증/정제 후 별도 CSV로 저장합니다.

## 환경 설치

기본 설치:

```bash
uv sync
```

고급 모델(LightGBM/XGBoost/CatBoost)까지 설치:

```bash
uv sync --extra advanced-models
```

참고:
- Python 3.14에서는 `catboost` 설치가 실패할 수 있습니다.
- 이 경우 `catboost` 모델은 자동으로 제외하고 나머지 모델만 학습/검증할 수 있습니다.

### 수집 엔진 준비

기본 수집 엔진은 Playwright입니다. 최초 1회 브라우저 설치가 필요합니다.

```bash
uv run playwright install chromium
```

### 원천 데이터 수집

```bash
python scripts/fetch_lotto.py
```

옵션:
- `--start`: 시작 회차 번호 (기본값: 1)
- `--end`: 종료 회차 번호 (기본값: 최신 회차까지)
- `--out-raw`: 출력 CSV 경로 (기본값: `data/raw/lotto_draws.csv`)
- `--sleep`: 요청 간 대기 시간(초) (기본값: 0.2)
- `--engine`: 수집 엔진 선택 (`playwright` 또는 `urllib`, 기본값: `playwright`)
- `--headless`: Playwright 헤드리스 모드 사용 여부 (기본값: `true`)

### 정제 데이터 생성

```bash
python scripts/prepare_dataset.py
```

옵션:
- `--in-raw`: 입력 CSV 경로 (기본값: `data/raw/lotto_draws.csv`)
- `--out-processed`: 출력 CSV 경로 (기본값: `data/processed/lotto_draws.csv`)

### 베이스라인 평가

```bash
python scripts/evaluate_baseline.py
```

옵션:
- `--test-size`: 테스트에 사용할 회차 수 (기본값: 100)
- `--eval-protocol`: 평가 방식 (`single_top6` 또는 `max_of_candidates`, 기본값: `single_top6`)
- `--num-candidates`: 회차당 추천 조합 개수 (`max_of_candidates`에서 사용, 기본값: 100)
- `--recent-window`: 최근 N회차만 사용해 빈도 가중치 계산
- `--hit-thresholds`: 적중 기준 (기본값: `1,2,3,4,5`)
- `--out-json`: 메트릭을 JSON으로 저장
- `--out-csv`: 메트릭을 CSV로 저장

### 피처 데이터셋 생성

```bash
python scripts/build_features.py
```

옵션:
- `--out-parquet`: 출력 Parquet 경로 (기본값: `data/features/lotto_features.parquet`)
- `--out-config`: 설정 JSON 경로 (기본값: `data/features/feature_config.json`)
- `--windows`: 빈도 윈도우 목록 (기본값: `5,10,20,50,100`)
- `--ewm`: 지수 감쇠율 목록 (기본값: `0.9,0.95`)
- `--cooc-windows`: 직전 회차 기준 공출현 윈도우 목록 (기본값: `20,50`)

생성 피처에는 빈도/최근성/추세/EWM 외에 `number_bucket`, `draw_month`, `draw_weekday`,
`cooc_prev_20`, `cooc_prev_50`가 포함됩니다.

추가 확장 피처:
- 번호 속성: `number_is_odd`, `number_is_high`, `number_mod_10`
- 직전 회차 컨텍스트: `is_prev_draw_number`, `is_prev_bonus_number`, `adjacent_prev_count`
- 보너스 히스토리: `bonus_freq_*`, `bonus_last_seen_gap`
- 분포/변화: `recent_odd_rate_*`, `recent_high_rate_*`, `recent_consecutive_rate_*`, `trend_ratio_*`, `freq_rate_diff_*`

### 모델 학습

```bash
python scripts/train_model.py
```

옵션:
- `--test-size`: 테스트에 사용할 회차 수 (기본값: 100)
- `--hit-thresholds`: 적중 기준 (기본값: `1,2,3,4,5`)
- `--calibration-bins`: ECE 계산 bin 수 (기본값: 10)
- `--out-model`: 모델 저장 경로 (pickle)
- `--out-json`: 메트릭을 JSON으로 저장
- `--out-csv`: 메트릭을 CSV로 저장
- `--mlp-max-iter`: MLP 최대 반복 수 (기본값: 1000)
- `--mlp-early-stopping` / `--no-mlp-early-stopping`: MLP early stopping on/off (기본값: on)
- `--mlp-tol`: MLP 수렴 허용오차 (기본값: `1e-3`)
- `--model`: 모델 타입
  - `logreg`, `gbdt`, `randomforest`, `extratrees`, `mlp`, `lightgbm`, `xgboost`, `catboost`

참고:
- `lightgbm`, `xgboost`, `catboost`는 추가 패키지 의존성이 필요합니다.
- 미설치 시 `train_model.py`가 설치 안내 메시지를 출력합니다.
- Python 3.14에서는 `catboost` 설치가 제한될 수 있습니다.

모델 리포트 고급 지표:
- `MRR`
- `mean_min_rank`
- `Brier`
- `LogLoss`
- `ECE`

### 다음 회차 예측

```bash
python scripts/predict_next.py --model-path models/logreg.pkl
```

앙상블 예측 + 결과 저장:

```bash
python scripts/predict_next.py --model-paths models/logreg.pkl,models/gbdt.pkl --out-json reports/predictions.json --out-csv reports/predictions.csv
```

### 메트릭 비교 리포트

```bash
python scripts/report_compare.py --baseline-json reports/baseline.json --model-json reports/model.json --out-md reports/compare.md
```

여러 모델 비교:

```bash
python scripts/report_compare.py --baseline-json reports/baseline.json --model-json reports/model.json,reports/model_gbdt.json --out-md reports/compare.md
```

### 롤링 백테스트(강화 검증)

```bash
python scripts/rolling_validate.py --models logreg,gbdt,randomforest,extratrees,mlp --out-md reports/rolling_validation.md
```

옵션:
- `--min-train-draws`: 첫 폴드 최소 학습 회차 수
- `--fold-test-size`: 폴드별 테스트 회차 수
- `--fold-step-size`: 폴드 간 이동 간격
- `--max-folds`: 최근 기준 최대 폴드 수
- `--calibration-bins`: ECE 계산 bin 수
- `--include-ensemble` / `--no-ensemble`: 폴드별 평균 앙상블 평가 on/off
- `--mlp-max-iter`, `--mlp-tol`, `--mlp-early-stopping`: 롤링 검증 시 MLP 수렴 설정
- `--out-json`: 롤링 검증 JSON 저장 경로
- `--out-md`: 롤링 검증 Markdown 저장 경로

### LightGBM 하이퍼파라미터 튜닝

```bash
python scripts/tune_lightgbm.py
```

튜닝 대상:
- `n_estimators`
- `num_leaves`
- `max_depth`
- `min_data_in_leaf`

출력:
- `reports/tuning_lightgbm.json`
- `reports/tuning_lightgbm.md`

### XGBoost 하이퍼파라미터 튜닝

```bash
python scripts/tune_xgboost.py
```

튜닝 대상:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

출력:
- `reports/tuning_xgboost.json`
- `reports/tuning_xgboost.md`

### 전체 파이프라인 한 번에 실행

```bash
./run_pipeline.sh
```

옵션 예시:

```bash
./run_pipeline.sh --skip-fetch --test-size 120
./run_pipeline.sh --fetch-start 1205 --train-end 1180
./run_pipeline.sh --fetch-engine urllib
./run_pipeline.sh --models logreg,gbdt,randomforest,extratrees,mlp
./run_pipeline.sh --lgbm-n-estimators 500 --lgbm-num-leaves 63 --lgbm-max-depth 8 --lgbm-min-data-in-leaf 10
./run_pipeline.sh --run-lgbm-tuning --run-xgb-tuning
./run_pipeline.sh --calibration-bins 15
```

파이프라인 출력:
- `reports/compare.md` (모델 비교 리포트)
- `reports/rolling_validation.md` (롤링 백테스트 리포트)
- `reports/predictions.json`, `reports/predictions.csv` (다음 회차 예측)
