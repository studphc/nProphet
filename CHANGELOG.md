# CHANGELOG

## [Unreleased]
- `PATTERN_WEIGHT`와 `ENSEMBLE_WEIGHT_MLP` 기본값을 `config`에 추가하여 초기 실행 시 `KeyError` 발생 문제 수정.
- 패턴 가중치를 남은 영업일 비율로 동적으로 조정하도록 개선.
- 당월 예측값이 현재 확정 매출보다 작지 않도록 하한 보정 로직 추가.
- 모든 메서드에 한글 docstring을 작성.
- `SIMULATION_DAY_OF_MONTH`에 `auto` 옵션을 추가해 실행 시점 날짜 자동 적용.
- GPU 사용 시 LightGBM을 GPU 모드로 학습하도록 옵션 추가.
- Optuna 하이퍼파라미터 탐색을 CPU 코어 수만큼 병렬화.
- LightGBM 경고 메시지를 억제해 콘솔 로그를 깔끔하게 유지.
  (STDERR 뿐 아니라 내부 C 출력도 차단)
- suppress_stdout 구현을 개선해 종료 시 'lost sys.stderr' 오류가 발생하지 않도록 수정.
