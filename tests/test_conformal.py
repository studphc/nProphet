import pandas as pd
from nProphet import NProphetForecaster


def test_conformal_delta_global():
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "y_true": [100, 120, 130],
            "yhat": [90, 110, 135],
        }
    )
    fc = NProphetForecaster({"CONFORMAL_ALPHA": 0.1, "SEED": 1})
    fc._calculate_conformal_delta(df)
    assert fc.conformal_delta == 10


def test_conformal_delta_monthwise():
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
            ),
            "y_true": [100, 110, 200, 210],
            "yhat": [90, 100, 190, 215],
        }
    )
    fc = NProphetForecaster({"CONFORMAL_MODE": "monthwise", "CONFORMAL_ALPHA": 0.2, "SEED": 1})
    fc._calculate_conformal_delta(df)
    jan = pd.Timestamp("2023-01-01")
    feb = pd.Timestamp("2023-02-01")
    assert isinstance(fc.conformal_delta, dict)
    assert fc.conformal_delta[jan] == 10
    assert fc.conformal_delta[feb] == 9
