# XGBoost 튜닝 리포트

## 설정
- 데이터: `data/features/lotto_features.parquet`
- 폴드 수: 1
- 탐색 조합 수: 1
- 탐색 파라미터: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

## 최적 파라미터
- n_estimators: 50
- max_depth: 4
- learning_rate: 0.1
- subsample: 1.0
- colsample_bytree: 1.0
- average_hits: 0.7500
- MRR: 0.2686
- Brier: 0.115915
- ECE: 0.000635
- 종합점수: 0.8462
- Hit@1: 0.5500
- Hit@2: 0.1500
- Hit@3: 0.0500
- Hit@4: 0.0000
- Hit@5: 0.0000

## 상위 결과

| 순위 | 종합점수 | average_hits | Hit@1 | Hit@2 | Hit@3 | Hit@4 | Hit@5 | MRR | Brier | ECE | n_estimators | max_depth | learning_rate | subsample | colsample_bytree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8462 | 0.7500 | 0.5500 | 0.1500 | 0.0500 | 0.0000 | 0.0000 | 0.2686 | 0.115915 | 0.000635 | 50 | 4 | 0.1 | 1.0 | 1.0 |
