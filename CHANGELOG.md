# CHANGELOG

## [Unreleased]
- `PATTERN_WEIGHT`와 `ENSEMBLE_WEIGHT_MLP` 기본값을 `config`에 추가하여 초기 실행 시 `KeyError` 발생 문제 수정.
- 패턴 가중치를 남은 영업일 비율로 동적으로 조정하도록 개선.
- 당월 예측값이 현재 확정 매출보다 작지 않도록 하한 보정 로직 추가.
- 모든 메서드에 한글 docstring을 작성.
- `DeliveryDate`와 `InvoiceDate` 차이를 활용해 월별 지연·전이율 피처를 추가.
