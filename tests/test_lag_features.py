import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nProphet import NProphetForecaster


def test_prepare_lag_conversion_features():
    fc = NProphetForecaster({"SEED": 1})
    fc.today = pd.Timestamp("2022-11-15")
    fc.first_of_month = fc.today.replace(day=1)
    fc.first_of_next_month = fc.first_of_month + pd.DateOffset(months=1)
    fc.year_end = fc.today.replace(month=12, day=31)

    fc.stock_base_df = pd.DataFrame(
        {
            "DeliveryDate": pd.to_datetime(["2022-09-10", "2022-09-15", "2022-10-01"]),
            "InvoiceDate": pd.to_datetime(["2022-09-12", "2022-09-20", "2022-10-05"]),
            "Quantity": [10, 20, 30],
        }
    )

    fc.prepare_lag_and_conversion_metrics()
    result = fc.monthly_lag_features
    assert {"ds", "mean_lag", "p90_lag", "max_lag", "conv_ma3"}.issubset(result.columns)
    assert len(result) == 2
