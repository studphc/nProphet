import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from nProphet import NProphetForecaster

def test_get_known_invoices_current_month():
    fc = NProphetForecaster({"SEED": 1})
    fc.today = pd.Timestamp("2023-09-10")
    fc.first_of_month = fc.today.replace(day=1)
    fc.first_of_next_month = fc.first_of_month + pd.DateOffset(months=1)
    fc.stock_base_df = pd.DataFrame(
        {
            "InvoiceDate": [
                pd.Timestamp("2023-09-05"),
                pd.Timestamp("2023-09-15"),
                pd.Timestamp("2023-09-20"),
            ],
            "NetAmount": [100, 200, 300],
        }
    )
    total, last_date = fc._get_known_invoices_current_month("InvoiceDate")
    assert total == 500
    assert last_date == pd.Timestamp("2023-09-20")



