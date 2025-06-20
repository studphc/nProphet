# CHANGELOG

## [Unreleased]
- `PATTERN_WEIGHT`와 `ENSEMBLE_WEIGHT_MLP` 기본값을 `config`에 추가하여 초기 실행 시 `KeyError` 발생 문제 수정.
- 예측 구간 계산을 분기별 상대 오차율 기반으로 변경하고, 고정된 월별 시즌 가중치를 적용.
