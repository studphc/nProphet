import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from nProphet import NProphetForecaster


def test_compute_simulation_date_auto():
    fc = NProphetForecaster({"SIMULATION_DAY_OF_MONTH": "auto", "SEED": 1})
    fc.today = pd.Timestamp("2023-08-15")
    base = pd.Timestamp("2023-07-01")
    assert fc._compute_simulation_date(base) == base.replace(day=15)


def test_compute_simulation_date_fixed():
    fc = NProphetForecaster({"SIMULATION_DAY_OF_MONTH": 10, "SEED": 1})
    base = pd.Timestamp("2023-07-01")
    assert fc._compute_simulation_date(base) == base.replace(day=10)


def test_compute_simulation_date_overflow():
    fc = NProphetForecaster({"SIMULATION_DAY_OF_MONTH": 31, "SEED": 1})
    base = pd.Timestamp("2023-06-01")
    assert fc._compute_simulation_date(base) == base.replace(day=30)
