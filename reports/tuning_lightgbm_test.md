# LightGBM 튜닝 리포트

## 설정
- 데이터: `data/features/lotto_features.parquet`
- 폴드 수: 1
- 탐색 조합 수: 1
- 탐색 파라미터: n_estimators, num_leaves, max_depth, min_data_in_leaf

## 최적 파라미터
- n_estimators: 50
- num_leaves: 31
- max_depth: 6
- min_data_in_leaf: 20
- learning_rate: 0.05
- average_hits: 0.9000
- MRR: 0.3533
- Brier: 0.115262
- ECE: 0.004608
- 종합점수: 1.0177
- Hit@1: 0.6500
- Hit@2: 0.2500
- Hit@3: 0.0000
- Hit@4: 0.0000
- Hit@5: 0.0000

## 상위 결과

| 순위 | 종합점수 | average_hits | Hit@1 | Hit@2 | Hit@3 | Hit@4 | Hit@5 | MRR | Brier | ECE | n_estimators | num_leaves | max_depth | min_data_in_leaf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0177 | 0.9000 | 0.6500 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.3533 | 0.115262 | 0.004608 | 50 | 31 | 6 | 20 |
