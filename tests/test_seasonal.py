import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nProphet import NProphetForecaster


def test_calculate_seasonal_factors_accepts_string_dates():
    fc = NProphetForecaster({"SEED": 1, "DB_PATH": "dummy"})
    fc.first_of_month = pd.Timestamp("2022-03-01")
    df = pd.DataFrame({"ds": ["2022-01-01", "2022-02-01"], "y": [10, 20]})

    fc.calculate_seasonal_factors(df)
    assert isinstance(fc.seasonal_factors, dict)

