"""autogloun.py
AutoGluon Time Series를 활용한 매출 예측 모델.
"""

import os
import sys
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import duckdb
import holidays
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def set_seed(seed: int) -> None:
    """재현성을 위해 시드를 고정한다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AutoGluonForecaster:
    """AutoGluon 기반 매출 예측기."""

    def __init__(self, config: dict) -> None:
        """클래스 초기화.

        Args:
            config (dict): 실행 설정 값 모음.
        """
        self.config = config
        set_seed(self.config["SEED"])
        self.db_conn = self._connect_db()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            self.today,
            self.first_of_month,
            self.first_of_next_month,
            self.year_end,
            self.month_start_str,
        ) = self._initialize_dates()
        self.predictor = None
        self.calendar_full = None
        self.wd_lookup = None
        self.stock_base_df = None

    @staticmethod
    @contextmanager
    def suppress_stdout():
        """콘솔 출력을 일시적으로 숨긴다."""
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def _initialize_dates(self):
        """날짜 관련 기본 값을 계산한다."""
        today = pd.Timestamp.now().normalize()
        first_of_month = today.replace(day=1)
        first_of_next_month = first_of_month + pd.DateOffset(months=1)
        year_end = today.replace(month=12, day=31)
        return (
            today,
            first_of_month,
            first_of_next_month,
            year_end,
            first_of_month.strftime("%Y-%m-%d"),
        )

    def _compute_simulation_date(self, base_date: pd.Timestamp) -> pd.Timestamp:
        """시뮬레이션 날짜 계산."""
        sim_day_setting = self.config.get("SIMULATION_DAY_OF_MONTH", 20)
        if isinstance(sim_day_setting, str) and sim_day_setting.lower() in ("auto", "today"):
            day_value = self.today.day
        else:
            try:
                day_value = int(sim_day_setting)
            except (ValueError, TypeError):
                day_value = 20
        day_value = max(1, min(day_value, base_date.days_in_month))
        return base_date.replace(day=day_value)

    def _connect_db(self):
        """DuckDB 연결을 생성한다."""
        db_path = self.config.get("DB_PATH")
        if not db_path or not os.path.exists(db_path):
            print(f"Error: DuckDB file not found at '{db_path}'.")
            return None
        try:
            db = duckdb.connect(db_path, read_only=True)
            threads = os.cpu_count() or 1
            db.execute(f"PRAGMA threads={threads}")
            print(f"Using {threads} DuckDB threads")
            return db
        except Exception as e:
            print(f"Failed to connect to DuckDB: {e}")
            return None

    def _load_data_from_db(self, date_column: str) -> bool:
        """데이터베이스에서 학습 데이터를 로드한다."""
        if not self.db_conn:
            return False
        try:
            query = f"SELECT * FROM stock_base WHERE {date_column} IS NOT NULL"
            df = self.db_conn.execute(query).fetch_arrow_table().to_pandas()
            for col in ["DeliveryDate", "InvoiceDate"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            if "NetAmount" in df.columns:
                df["NetAmount"] = pd.to_numeric(
                    df["NetAmount"].astype(str).str.replace(",", ""), errors="coerce"
                ).fillna(0)
            df.dropna(subset=[date_column], inplace=True)
            start_date_filter = self.config.get("TRAINING_START_DATE")
            if start_date_filter:
                df = df[df[date_column] >= pd.to_datetime(start_date_filter)].copy()
            self.stock_base_df = df
            return True
        except Exception as e:
            print(f"Failed to load data from DuckDB: {e}")
            return False

    def get_complete_calendar(self) -> None:
        """과거부터 연말까지의 캘린더 데이터를 생성한다."""
        start_date = self.today - pd.DateOffset(years=5)
        if self.stock_base_df is not None and not self.stock_base_df.empty:
            min_date_in_data = self.stock_base_df["DeliveryDate"].min()
            if pd.notna(min_date_in_data):
                start_date = min_date_in_data
        full_date_range = pd.date_range(start=start_date, end=self.year_end, freq="D")
        calendar = pd.DataFrame(index=full_date_range)
        calendar["IsHoliday"] = calendar.index.dayofweek.isin([5, 6]).astype(int)
        kr_holidays = holidays.KR(years=calendar.index.year.unique())
        for d in kr_holidays:
            if d in calendar.index:
                calendar.loc[d, "IsHoliday"] = 1
        calendar["ds"] = pd.to_datetime(calendar.index).to_period("M").to_timestamp()
        self.wd_lookup = (
            calendar[calendar["IsHoliday"] == 0]
            .groupby("ds")
            .size()
            .rename("working_days")
        )
        self.calendar_full = calendar.reset_index().rename(columns={"index": "Date"})

    def calculate_daily_weights(self, date_column: str) -> None:
        """영업일별 매출 가중치를 계산한다."""
        past_data = self.stock_base_df[
            self.stock_base_df[date_column] < self.first_of_month
        ].copy()
        if past_data.empty:
            self.daily_weights_del = [1.0 / 23] * 23
            return
        daily_sales = (
            past_data.groupby(past_data[date_column].dt.date)
            .agg(daily_y=("NetAmount", "sum"))
            .reset_index()
        )
        daily_sales.rename(columns={date_column: "Date"}, inplace=True)
        daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
        merged = pd.merge(daily_sales, self.calendar_full, on="Date", how="left")
        ws = merged[merged["IsHoliday"] == 0].copy()
        if ws.empty:
            self.daily_weights_del = [1.0 / 23] * 23
            return
        ws["ds"] = ws["Date"].dt.to_period("M").dt.to_timestamp()
        ws["working_day_of_month"] = ws.groupby("ds").cumcount()
        monthly_totals = ws.groupby("ds")["daily_y"].sum().rename("monthly_y")
        ws = ws.join(monthly_totals, on="ds")
        ws = ws[ws["monthly_y"] > 0]
        ws["weight"] = ws["daily_y"] / ws["monthly_y"]
        avg_weights = ws.groupby("working_day_of_month")["weight"].mean()
        if avg_weights.empty or avg_weights.sum() == 0:
            self.daily_weights_del = [1.0 / 23] * 23
            return
        final_weights = (avg_weights / avg_weights.sum()).tolist()
        if len(final_weights) < 25:
            final_weights.extend([0] * (25 - len(final_weights)))
        self.daily_weights_del = final_weights

    def calculate_seasonal_factors(self, df_hist: pd.DataFrame) -> None:
        """월별 계절성 보정 계수를 계산한다."""
        cutoff = self.first_of_month - pd.offsets.MonthBegin(1)
        sf_base = df_hist[(df_hist["ds"] < cutoff) & (df_hist["y"] > 0)].copy()
        if sf_base.empty:
            self.seasonal_factors = {m: 1.0 for m in range(1, 13)}
            return
        sf_base["month"] = sf_base["ds"].dt.month
        monthly_avg = sf_base.groupby("month")["y"].mean()
        overall_avg = sf_base["y"].mean()
        self.seasonal_factors = (monthly_avg / overall_avg).to_dict()

    def create_features(self, df: pd.DataFrame):
        """모델 학습에 필요한 피처를 생성한다."""
        df = df.merge(self.wd_lookup.reset_index(name="working_days"), on="ds", how="left")
        df["working_days"] = df["working_days"].replace(0, 1).fillna(int(self.wd_lookup.median()))
        df["y_daily_avg"] = df["y"] / df["working_days"]
        df["t"] = (df["ds"] - df["ds"].min()).dt.days // 30
        df["month_sin"], df["month_cos"] = (
            np.sin(2 * np.pi * df["ds"].dt.month / 12),
            np.cos(2 * np.pi * df["ds"].dt.month / 12),
        )
        df["y_norm"] = df["y_daily_avg"] / self.config["Y_SCALE"]
        df = df.fillna(0)
        feature_names = ["t", "working_days", "month_sin", "month_cos"]
        return df, feature_names

    def _blend_current_month_projection(
        self,
        pattern_based_projection: float,
        ai_prediction: float,
        actual_so_far: float,
        working_days_so_far: int,
        total_working_days: int,
    ) -> tuple[float, float]:
        """패턴 예측과 AI 예측을 동적으로 블렌딩한다."""
        remaining_ratio = max(0.0, 1 - working_days_so_far / total_working_days)
        w_base = self.config["PATTERN_WEIGHT"]
        w_dynamic = w_base * remaining_ratio
        blended = (pattern_based_projection * w_dynamic) + (ai_prediction * (1 - w_dynamic))
        return max(blended, actual_so_far), w_dynamic

    def train_autogluon_model(self, df_hist: pd.DataFrame) -> TimeSeriesDataFrame:
        """AutoGluon TimeSeriesPredictor 학습."""
        ts_df = df_hist.rename(columns={"ds": "timestamp", "y": "target"}).copy()
        ts_df["item_id"] = "sales"
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        with self.suppress_stdout():
            self.predictor = TimeSeriesPredictor(
                prediction_length=1, target="target", freq="MS"
            ).fit(train_data=ts_data)
        return ts_data

    def _forecast_next_month(self, ts_data: TimeSeriesDataFrame) -> float:
        """다음 달 예측치를 반환한다."""
        forecast = self.predictor.predict(ts_data)
        pred = forecast.loc["sales"].iloc[-1]
        working_days = self.wd_lookup.get(self.first_of_month, int(self.wd_lookup.median()))
        return float(pred) * self.config["Y_SCALE"] * working_days

    def run(self, date_column: str = "DeliveryDate") -> None:
        """예측 파이프라인 전체를 실행한다."""
        if not self._load_data_from_db(date_column):
            return
        self.get_complete_calendar()
        df_hist = self.stock_base_df[
            (self.stock_base_df[date_column].notna())
            & (self.stock_base_df[date_column] < self.first_of_month)
        ].copy()
        df_hist["ds"] = df_hist[date_column].dt.to_period("M").dt.to_timestamp()
        df_hist = df_hist.groupby("ds").agg(y=("NetAmount", "sum")).reset_index()
        if len(df_hist) < 6:
            print("Not enough historical data.")
            return
        self.calculate_daily_weights(date_column)
        self.calculate_seasonal_factors(df_hist)
        ts_data = self.train_autogluon_model(df_hist.copy())
        model_pred = self._forecast_next_month(ts_data)
        print(f"다음 달 예측 매출: {model_pred:,.0f} KRW")


if __name__ == "__main__":
    config = {
        "DB_PATH": "./data/import/base.duckdb",
        "TRAINING_START_DATE": "2021-01-01",
        "SEED": 42,
        "Y_SCALE": 1e7,
        "PATTERN_WEIGHT": 0.5,
        "SIMULATION_DAY_OF_MONTH": 20,
    }
    forecaster = AutoGluonForecaster(config)
    forecaster.run(date_column="DeliveryDate")
