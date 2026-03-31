#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SKIP_FETCH=0
FETCH_START=""
FETCH_END=""
TEST_SIZE=100
TRAIN_END=""
BASELINE_PROTOCOL="single_top6"
FETCH_ENGINE="playwright"
TRAIN_DECAY=0.998
# catboost는 Python 3.14 환경에서 설치 호환 이슈가 있어 기본 목록에서 제외한다.
MODEL_LIST="logreg,gbdt,randomforest,extratrees,mlp,lightgbm,xgboost"
NUM_CANDIDATES=256
PORTFOLIO_SIZE=12
CANDIDATE_POOL_SIZE=18
SAMPLING_TEMPERATURE=0.9
OVERLAP_PENALTY=0.18
UNIQUE_BONUS=0.035

ROLLING_MIN_TRAIN_DRAWS=600
ROLLING_FOLD_TEST_SIZE=40
ROLLING_FOLD_STEP_SIZE=20
ROLLING_MAX_FOLDS=8
CALIBRATION_BINS=10

LGBM_N_ESTIMATORS=500
LGBM_NUM_LEAVES=63
LGBM_MAX_DEPTH=8
LGBM_MIN_DATA_IN_LEAF=10

RUN_LGBM_TUNING=0
RUN_XGB_TUNING=0

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh [options]

Options:
  --skip-fetch                      원천 데이터 수집 단계를 건너뜁니다.
  --fetch-start <draw_no>           수집 시작 회차.
  --fetch-end <draw_no>             수집 종료 회차.
  --fetch-engine <engine>           수집 엔진: playwright|urllib (기본값: playwright).
  --test-size <n>                   평가 테스트 회차 수 (기본값: 100).
  --train-end <draw_no>             학습 종료 회차(지정 시 --test-size 무시).
  --train-decay <v>                 최근 회차 가중 학습 감쇠율(기본값: 0.998).
  --baseline-protocol <mode>        baseline 평가 방식: single_top6|max_of_candidates|portfolio
  --models <csv>                    학습/검증 모델 목록(쉼표 구분).
  --num-candidates <n>              회차당 생성할 후보 조합 수.
  --portfolio-size <n>              최종 포트폴리오 티켓 수.
  --candidate-pool-size <n>         후보 생성 시 샘플링에 사용할 상위 번호 풀 크기.
  --sampling-temperature <v>        후보 생성 temperature.
  --overlap-penalty <v>             포트폴리오 중복 패널티.
  --unique-bonus <v>                포트폴리오 고유번호 보너스.
  --rolling-min-train-draws <n>     롤링 검증 최소 학습 회차 수.
  --rolling-fold-test-size <n>      롤링 검증 폴드별 테스트 회차 수.
  --rolling-fold-step-size <n>      롤링 검증 폴드 간 이동 간격.
  --rolling-max-folds <n>           롤링 검증 최대 폴드 수(최근 기준).
  --calibration-bins <n>            ECE 계산용 bin 개수(기본값: 10).
  --lgbm-n-estimators <n>           LightGBM n_estimators (기본값: 500).
  --lgbm-num-leaves <n>             LightGBM num_leaves (기본값: 63).
  --lgbm-max-depth <n>              LightGBM max_depth (기본값: 8).
  --lgbm-min-data-in-leaf <n>       LightGBM min_data_in_leaf (기본값: 10).
  --run-lgbm-tuning                 LightGBM 튜닝 스크립트를 실행합니다.
  --run-xgb-tuning                  XGBoost 튜닝 스크립트를 실행합니다.
  -h, --help                        도움말 출력.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fetch)
      SKIP_FETCH=1
      shift
      ;;
    --fetch-start)
      FETCH_START="${2:-}"
      shift 2
      ;;
    --fetch-end)
      FETCH_END="${2:-}"
      shift 2
      ;;
    --fetch-engine)
      FETCH_ENGINE="${2:-}"
      shift 2
      ;;
    --test-size)
      TEST_SIZE="${2:-}"
      shift 2
      ;;
    --train-end)
      TRAIN_END="${2:-}"
      shift 2
      ;;
    --train-decay)
      TRAIN_DECAY="${2:-}"
      shift 2
      ;;
    --baseline-protocol)
      BASELINE_PROTOCOL="${2:-}"
      shift 2
      ;;
    --models)
      MODEL_LIST="${2:-}"
      shift 2
      ;;
    --num-candidates)
      NUM_CANDIDATES="${2:-}"
      shift 2
      ;;
    --portfolio-size)
      PORTFOLIO_SIZE="${2:-}"
      shift 2
      ;;
    --candidate-pool-size)
      CANDIDATE_POOL_SIZE="${2:-}"
      shift 2
      ;;
    --sampling-temperature)
      SAMPLING_TEMPERATURE="${2:-}"
      shift 2
      ;;
    --overlap-penalty)
      OVERLAP_PENALTY="${2:-}"
      shift 2
      ;;
    --unique-bonus)
      UNIQUE_BONUS="${2:-}"
      shift 2
      ;;
    --rolling-min-train-draws)
      ROLLING_MIN_TRAIN_DRAWS="${2:-}"
      shift 2
      ;;
    --rolling-fold-test-size)
      ROLLING_FOLD_TEST_SIZE="${2:-}"
      shift 2
      ;;
    --rolling-fold-step-size)
      ROLLING_FOLD_STEP_SIZE="${2:-}"
      shift 2
      ;;
    --rolling-max-folds)
      ROLLING_MAX_FOLDS="${2:-}"
      shift 2
      ;;
    --calibration-bins)
      CALIBRATION_BINS="${2:-}"
      shift 2
      ;;
    --lgbm-n-estimators)
      LGBM_N_ESTIMATORS="${2:-}"
      shift 2
      ;;
    --lgbm-num-leaves)
      LGBM_NUM_LEAVES="${2:-}"
      shift 2
      ;;
    --lgbm-max-depth)
      LGBM_MAX_DEPTH="${2:-}"
      shift 2
      ;;
    --lgbm-min-data-in-leaf)
      LGBM_MIN_DATA_IN_LEAF="${2:-}"
      shift 2
      ;;
    --run-lgbm-tuning)
      RUN_LGBM_TUNING=1
      shift
      ;;
    --run-xgb-tuning)
      RUN_XGB_TUNING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$BASELINE_PROTOCOL" != "single_top6" && "$BASELINE_PROTOCOL" != "max_of_candidates" && "$BASELINE_PROTOCOL" != "portfolio" ]]; then
  echo "Invalid --baseline-protocol: $BASELINE_PROTOCOL" >&2
  exit 1
fi

if [[ "$FETCH_ENGINE" != "playwright" && "$FETCH_ENGINE" != "urllib" ]]; then
  echo "Invalid --fetch-engine: $FETCH_ENGINE" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_RUN=(./.venv/bin/python)
else
  PYTHON_RUN=(python3)
fi

run_step() {
  local title="$1"
  shift
  echo
  echo "==> $title"
  "$@"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

trim_spaces() {
  echo "$1" | tr -d ' '
}

model_output_paths() {
  local model="$1"
  local json_path=""
  local csv_path=""
  local model_path="models/${model}.pkl"

  case "$model" in
    logreg)
      json_path="reports/model.json"
      csv_path="reports/model.csv"
      model_path="models/logreg.pkl"
      ;;
    gbdt)
      json_path="reports/model_gbdt.json"
      csv_path="reports/model_gbdt.csv"
      model_path="models/gbdt.pkl"
      ;;
    *)
      json_path="reports/model_${model}.json"
      csv_path="reports/model_${model}.csv"
      ;;
  esac

  echo "$json_path|$csv_path|$model_path"
}

mkdir -p data/raw data/processed data/features reports models

SPLIT_ARGS=()
if [[ -n "$TRAIN_END" ]]; then
  SPLIT_ARGS+=(--train-end "$TRAIN_END")
else
  SPLIT_ARGS+=(--test-size "$TEST_SIZE")
fi

if [[ "$SKIP_FETCH" -eq 0 ]]; then
  FETCH_ARGS=(scripts/fetch_lotto.py)
  if [[ -n "$FETCH_START" ]]; then
    FETCH_ARGS+=(--start "$FETCH_START")
  fi
  if [[ -n "$FETCH_END" ]]; then
    FETCH_ARGS+=(--end "$FETCH_END")
  fi
  FETCH_ARGS+=(--engine "$FETCH_ENGINE")
  run_step "수집 (fetch_lotto)" "${PYTHON_RUN[@]}" "${FETCH_ARGS[@]}"
else
  echo
  echo "==> 수집 단계 건너뜀 (--skip-fetch)"
fi

run_step "정제 (prepare_dataset)" "${PYTHON_RUN[@]}" scripts/prepare_dataset.py
run_step "피처 생성 (build_features)" "${PYTHON_RUN[@]}" scripts/build_features.py

if [[ "$RUN_LGBM_TUNING" -eq 1 ]]; then
  run_step "튜닝 (tune_lightgbm)" \
    "${PYTHON_RUN[@]}" scripts/tune_lightgbm.py
fi

if [[ "$RUN_XGB_TUNING" -eq 1 ]]; then
  run_step "튜닝 (tune_xgboost)" \
    "${PYTHON_RUN[@]}" scripts/tune_xgboost.py
fi

run_step "베이스라인 평가 (evaluate_baseline)" \
  "${PYTHON_RUN[@]}" scripts/evaluate_baseline.py \
  "${SPLIT_ARGS[@]}" \
  --eval-protocol "$BASELINE_PROTOCOL" \
  --num-candidates "$NUM_CANDIDATES" \
  --portfolio-size "$PORTFOLIO_SIZE" \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --overlap-penalty "$OVERLAP_PENALTY" \
  --unique-bonus "$UNIQUE_BONUS" \
  --out-json reports/baseline.json \
  --out-csv reports/baseline.csv

REQUESTED_MODELS_CSV="$(trim_spaces "$MODEL_LIST")"
IFS=',' read -r -a REQUESTED_MODELS <<< "$REQUESTED_MODELS_CSV"

SUCCESS_MODEL_JSONS=()
SUCCESS_MODEL_PATHS=()
SUCCESS_MODELS=()
FAILED_MODELS=()

for model in "${REQUESTED_MODELS[@]}"; do
  [[ -z "$model" ]] && continue

  IFS='|' read -r out_json out_csv out_model < <(model_output_paths "$model")

  echo
  echo "==> 모델 학습 ($model)"
  set +e
  if [[ "$model" == "lightgbm" ]]; then
    "${PYTHON_RUN[@]}" scripts/train_model.py \
      "${SPLIT_ARGS[@]}" \
      --n-estimators "$LGBM_N_ESTIMATORS" \
      --num-leaves "$LGBM_NUM_LEAVES" \
      --max-depth "$LGBM_MAX_DEPTH" \
      --min-data-in-leaf "$LGBM_MIN_DATA_IN_LEAF" \
      --calibration-bins "$CALIBRATION_BINS" \
      --train-decay "$TRAIN_DECAY" \
      --num-candidates "$NUM_CANDIDATES" \
      --portfolio-size "$PORTFOLIO_SIZE" \
      --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
      --sampling-temperature "$SAMPLING_TEMPERATURE" \
      --overlap-penalty "$OVERLAP_PENALTY" \
      --unique-bonus "$UNIQUE_BONUS" \
      --model "$model" \
      --out-json "$out_json" \
      --out-csv "$out_csv" \
      --out-model "$out_model"
  else
    "${PYTHON_RUN[@]}" scripts/train_model.py \
      "${SPLIT_ARGS[@]}" \
      --calibration-bins "$CALIBRATION_BINS" \
      --train-decay "$TRAIN_DECAY" \
      --num-candidates "$NUM_CANDIDATES" \
      --portfolio-size "$PORTFOLIO_SIZE" \
      --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
      --sampling-temperature "$SAMPLING_TEMPERATURE" \
      --overlap-penalty "$OVERLAP_PENALTY" \
      --unique-bonus "$UNIQUE_BONUS" \
      --model "$model" \
      --out-json "$out_json" \
      --out-csv "$out_csv" \
      --out-model "$out_model"
  fi
  status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    SUCCESS_MODEL_JSONS+=("$out_json")
    SUCCESS_MODEL_PATHS+=("$out_model")
    SUCCESS_MODELS+=("$model")
  else
    echo "경고: 모델 '$model' 학습 실패(코드: $status). 이 모델은 파이프라인에서 제외합니다."
    FAILED_MODELS+=("$model")
  fi
done

if [[ ${#SUCCESS_MODELS[@]} -eq 0 ]]; then
  echo "오류: 학습/검증에 성공한 모델이 없습니다." >&2
  exit 1
fi

MODEL_JSON_ARG="$(join_by_comma "${SUCCESS_MODEL_JSONS[@]}")"
MODEL_PATH_ARG="$(join_by_comma "${SUCCESS_MODEL_PATHS[@]}")"

run_step "리포트 생성 (report_compare)" \
  "${PYTHON_RUN[@]}" scripts/report_compare.py \
  --baseline-json reports/baseline.json \
  --model-json "$MODEL_JSON_ARG" \
  --out-md reports/compare.md

run_step "롤링 검증 (rolling_validate)" \
  "${PYTHON_RUN[@]}" scripts/rolling_validate.py \
  --models "$REQUESTED_MODELS_CSV" \
  --min-train-draws "$ROLLING_MIN_TRAIN_DRAWS" \
  --fold-test-size "$ROLLING_FOLD_TEST_SIZE" \
  --fold-step-size "$ROLLING_FOLD_STEP_SIZE" \
  --max-folds "$ROLLING_MAX_FOLDS" \
  --calibration-bins "$CALIBRATION_BINS" \
  --train-decay "$TRAIN_DECAY" \
  --num-candidates "$NUM_CANDIDATES" \
  --portfolio-size "$PORTFOLIO_SIZE" \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --overlap-penalty "$OVERLAP_PENALTY" \
  --unique-bonus "$UNIQUE_BONUS" \
  --lgbm-n-estimators "$LGBM_N_ESTIMATORS" \
  --num-leaves "$LGBM_NUM_LEAVES" \
  --lgbm-max-depth "$LGBM_MAX_DEPTH" \
  --min-data-in-leaf "$LGBM_MIN_DATA_IN_LEAF" \
  --out-json reports/rolling_validation.json \
  --out-md reports/rolling_validation.md

run_step "다음 회차 예측 (predict_next)" \
  "${PYTHON_RUN[@]}" scripts/predict_next.py \
  --model-paths "$MODEL_PATH_ARG" \
  --num-candidates "$NUM_CANDIDATES" \
  --portfolio-size "$PORTFOLIO_SIZE" \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --overlap-penalty "$OVERLAP_PENALTY" \
  --unique-bonus "$UNIQUE_BONUS" \
  --out-json reports/predictions.json \
  --out-csv reports/predictions.csv \
  --out-portfolio-csv reports/prediction_portfolio.csv

echo
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "제외된 모델: $(join_by_comma "${FAILED_MODELS[@]}")"
fi
echo "사용된 모델: $(join_by_comma "${SUCCESS_MODELS[@]}")"
echo "완료: reports/compare.md, reports/rolling_validation.md, reports/predictions.json, reports/predictions.csv, reports/prediction_portfolio.csv"
