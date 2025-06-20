# CHANGELOG

## [Unreleased]
- `PATTERN_WEIGHT`와 `ENSEMBLE_WEIGHT_MLP` 기본값을 `config`에 추가하여 초기 실행 시 `KeyError` 발생 문제 수정.
- 패턴 가중치를 남은 영업일 비율로 동적으로 조정하도록 개선.
- 당월 예측값이 현재 확정 매출보다 작지 않도록 하한 보정 로직 추가.
- 모든 메서드에 한글 docstring을 작성.
- `SIMULATION_DAY_OF_MONTH`에 `auto` 옵션을 추가해 실행 시점 날짜 자동 적용.
- AutoGluon Time Series 기반 단순 예측 스크립트 `autogloun.py` 추가.
- 해당 스크립트를 위한 단위 테스트 `test_autogloun.py` 작성.
- `holidays` 미사용 오류 수정 및 의존성 업데이트.
