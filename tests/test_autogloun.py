import pandas as pd
from autogloun import AutoGluonForecaster


def test_compute_simulation_date_auto():
    fc = AutoGluonForecaster({"SIMULATION_DAY_OF_MONTH": "auto", "SEED": 1})
    fc.today = pd.Timestamp("2023-08-15")
    base = pd.Timestamp("2023-07-01")
    assert fc._compute_simulation_date(base) == base.replace(day=15)


def test_compute_simulation_date_fixed():
    fc = AutoGluonForecaster({"SIMULATION_DAY_OF_MONTH": 10, "SEED": 1})
    base = pd.Timestamp("2023-07-01")
    assert fc._compute_simulation_date(base) == base.replace(day=10)


def test_blend_projection_guarantees_lower_bound():
    fc = AutoGluonForecaster({"PATTERN_WEIGHT": 0.5, "SEED": 1})
    result, weight = fc._blend_current_month_projection(
        pattern_based_projection=80,
        ai_prediction=70,
        actual_so_far=90,
        working_days_so_far=15,
        total_working_days=20,
    )
    assert result >= 90
    assert 0 <= weight <= 1


def test_blend_projection_dynamic_weight():
    fc = AutoGluonForecaster({"PATTERN_WEIGHT": 0.8, "SEED": 1})
    result, weight = fc._blend_current_month_projection(
        pattern_based_projection=100,
        ai_prediction=80,
        actual_so_far=50,
        working_days_so_far=5,
        total_working_days=20,
    )
    expected_weight = 0.8 * (1 - 5 / 20)
    expected = (100 * expected_weight) + (80 * (1 - expected_weight))
    expected = max(expected, 50)
    assert abs(result - expected) < 1e-6
    assert abs(weight - expected_weight) < 1e-6
