# nProphet
"""
자율 최적화 및 하이브리드 앙상블 매출 예측 엔진 (최종 완성본)

[기능 요약]
이 시스템은 과거 매출 데이터를 기반으로 미래 매출을 예측하는 자동화된 파이프라인입니다.
주요 기능은 다음과 같습니다.

1.  데이터 자동 분석 및 처리:
    - DuckDB 데이터베이스에서 데이터를 고속으로 로딩하고 전처리합니다.
    - `holidays` 라이브러리를 통해 대한민국 법정공휴일을 자동 반영하여 정확한 영업일 수를 계산합니다.
    - '미완결 월' 및 '0-매출 월'을 통계에서 제외하여 계절성 분석의 안정성을 확보합니다.

2.  지능형 피처 엔지니어링:
    - 시계열의 추세, 주기성(계절성), 과거 정보(Lag), 이동평균 등 예측에 유용한 피처를 자동으로 생성합니다.

3.  자율 최적화 모델링 (Self-Optimizing):
    - Optuna를 이용해 AI 모델의 하이퍼파라미터와 최종 예측의 블렌딩 가중치를 동시에 자동 튜닝합니다.
    - 시계열 교차검증(TimeSeriesSplit) 및 Pruner를 적용하여 안정적이고 효율적인 최적화를 수행합니다.
    - LightGBM 학습 시 Early Stopping을 적용하여 효율을 높이고, 최종 모델은 전체 데이터로 재학습(Re-fit)합니다.

4.  고성능 하이브리드 예측:
    - 딥러닝(MLP) 모델과 트리 기반(LightGBM) 모델의 장점을 결합한 앙상블 예측을 수행합니다.
    - Conformal Calibration을 통해 통계적으로 보정된 신뢰구간(95% PICP 목표)을 생성합니다.
    - 'AI 예측'과 '미래 확정 매출(선행 발주)'을 비교하여 더 신뢰도 높은 값을 선택하는 하이브리드 로직을 적용합니다.

5.  상세 리포트 및 시각화:
    - 백테스팅 기반의 객관적인 모델 성능 평가 지표(SMAPE, MAE, PICP 등)를 계산합니다.
    - 예측 결과를 상세한 분석 내용과 함께 콘솔에 출력하고, 별도의 텍스트 파일과 잔차 히스토그램으로 저장합니다.
    - 과거 실적부터 미래 예측, 신뢰구간까지 한눈에 볼 수 있는 종합 그래프를 생성하고 이미지 파일로 저장합니다.
    - 모델 저장 시 타임스탬프를 활용하여 버전을 관리합니다.

6.  예측 보정 로직:
    - 남은 영업일 비율에 따라 패턴 기반 가중치를 자동 조정합니다.
    - 최종 예측치는 현재까지 확정된 매출액보다 작지 않도록 보정합니다.

## 기본 설정
`nProphet.py` 실행을 위한 주요 설정 값은 스크립트 하단의 `config` 딕셔너리에 정의되어 있습니다. 
특히 `PATTERN_WEIGHT`와 `ENSEMBLE_WEIGHT_MLP`는 초기 가중치로 사용되며,
하이퍼파라미터 최적화 과정에서 자동으로 조정됩니다. 기본값은 두 항목 모두 `0.5`입니다.
또한 `SIMULATION_DAY_OF_MONTH`에 정수를 지정하면 해당 일자 기준으로 백테스트를 수행하며,
`auto`(또는 `today`)로 설정하면 실행 당일 날짜를 사용합니다.
