# nprophet_final_v2.py
"""
자율 최적화 및 하이브리드 앙상블 매출 예측 엔진 (최종 완성본)

[v47 최종 수정]
- AttributeError: '_load_data' 누락 오류 최종 수정 (함수 이름 통일)
- 데이터 소스를 DuckDB로 완전히 고정하고, 불필요한 CSV 로직을 제거하여 코드 안정성 및 명확성 확보
"""

import os
import sys
from contextlib import contextmanager
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
import holidays
import optuna
import joblib
import lightgbm as lgbm
from tqdm import tqdm
import random
import logging
import duckdb


def set_seed(seed):
    """재현성을 위해 시드를 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


class NProphetForecaster:
    """매출 예측 전 과정을 담당하는 핵심 클래스."""

    def __init__(self, config):
        """클래스 초기화.

        Args:
            config (dict): 실행 설정 값 모음.
        """

        self.config = config
        set_seed(self.config["SEED"])
        self.db_conn = self._connect_db()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        (
            self.today,
            self.first_of_month,
            self.first_of_next_month,
            self.year_end,
            self.month_start_str,
        ) = self._initialize_dates()

        self.stock_base_df = None
        self.mlp_model, self.lgbm_model, self.scaler = None, None, None
        self.wd_lookup, self.calendar_full, self.daily_weights_del = None, None, None
        self.best_params, self.seasonal_factors, self.lgbm_residual_mae = {}, {}, 0.0
        self.conformal_delta = 0.0
        self.lgbm_feature_names_ = None

    @staticmethod
    @contextmanager
    def suppress_stdout():
        """콘솔과 에러 출력을 일시적으로 숨긴다."""

        with open(os.devnull, "w") as devnull_out, open(os.devnull, "w") as devnull_err:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = devnull_out
            sys.stderr = devnull_err
            os.dup2(devnull_out.fileno(), 1)
            os.dup2(devnull_err.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
                sys.stdout = old_stdout
                sys.stderr = old_stderr

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
        """시뮬레이션 날짜를 계산한다.

        Args:
            base_date (pd.Timestamp): 기준이 되는 날짜.

        Returns:
            pd.Timestamp: 계산된 시뮬레이션 날짜.
        """

        sim_day_setting = self.config.get("SIMULATION_DAY_OF_MONTH", 20)
        if isinstance(sim_day_setting, str) and sim_day_setting.lower() in (
            "auto",
            "today",
        ):
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

    def _load_data_from_db(self, date_column):
        """데이터베이스에서 학습 데이터를 로드한다."""

        if not self.db_conn:
            return False
        print("Loading data from DuckDB...")
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
                print(f"Filtering data to start from {start_date_filter}...")
                df = df[df[date_column] >= pd.to_datetime(start_date_filter)].copy()
            self.stock_base_df = df
            print("Data loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load data from DuckDB: {e}")
            return False

    def get_complete_calendar(self):
        """과거부터 연말까지의 캘린더 데이터를 생성한다."""

        print("Generating complete calendar...")
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
        print("Calendar generation complete.")

    def calculate_daily_weights(self, date_column):
        """영업일별 매출 가중치를 계산한다."""

        print("Calculating daily sales weights...")
        past_data = self.stock_base_df[
            self.stock_base_df[date_column] < self.first_of_month
        ].copy()
        if past_data.empty:
            print("Warning: No hist data for daily weights.")
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
            print("Warning: No working day sales data found.")
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
            print("Warning: Could not learn daily weights pattern.")
            self.daily_weights_del = [1.0 / 23] * 23
            return
        final_weights = (avg_weights / avg_weights.sum()).tolist()
        if len(final_weights) < 25:
            final_weights.extend([0] * (25 - len(final_weights)))
        self.daily_weights_del = final_weights

    def calculate_seasonal_factors(self, df_hist):
        """월별 계절성 보정 계수를 계산한다."""

        cutoff = self.first_of_month - pd.offsets.MonthBegin(1)
        sf_base = df_hist[(df_hist["ds"] < cutoff) & (df_hist["y"] > 0)].copy()
        print(f"Calculating seasonal factors (up to {cutoff.strftime('%Y-%m')})...")
        if sf_base.empty:
            print("Not enough valid data for factors.")
            self.seasonal_factors = {m: 1.0 for m in range(1, 13)}
            return
        sf_base["month"] = sf_base["ds"].dt.month
        monthly_avg = sf_base.groupby("month")["y"].mean()
        overall_avg = sf_base["y"].mean()
        self.seasonal_factors = (monthly_avg / overall_avg).to_dict()
        print(
            "Seasonal Factors:",
            {k: round(v, 2) for k, v in self.seasonal_factors.items()},
        )

    def create_features(self, df):
        """모델 학습에 필요한 피처를 생성한다."""

        df = df.merge(
            self.wd_lookup.reset_index(name="working_days"), on="ds", how="left"
        )
        df["working_days"] = (
            df["working_days"].replace(0, 1).fillna(int(self.wd_lookup.median()))
        )
        df["y_daily_avg"] = df["y"] / df["working_days"]
        df["t"] = (df["ds"] - df["ds"].min()).dt.days // 30
        df["month_sin"], df["month_cos"] = (
            np.sin(2 * np.pi * df["ds"].dt.month / 12),
            np.cos(2 * np.pi * df["ds"].dt.month / 12),
        )
        df["y_norm"] = df["y_daily_avg"] / self.config["Y_SCALE"]
        df["y_norm_lag1"], df["y_norm_lag3"], df["y_norm_lag12"] = (
            df["y_norm"].shift(1),
            df["y_norm"].shift(3),
            df["y_norm"].shift(12),
        )
        df["y_norm_ma3"], df["y_norm_std3"] = (
            df["y_norm"].rolling(window=3).mean().shift(1),
            df["y_norm"].rolling(window=3).std().shift(1),
        )
        df = df.fillna(0)
        feature_names = [
            "t",
            "working_days",
            "month_sin",
            "month_cos",
            "y_norm_lag1",
            "y_norm_lag3",
            "y_norm_lag12",
            "y_norm_ma3",
            "y_norm_std3",
        ]
        for col in feature_names:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df, feature_names

    def _build_mlp_model(self, n_features, params):
        """MLP 모델 구조를 생성한다."""

        return nn.Sequential(
            nn.Linear(n_features, params["n_units_l1"]),
            nn.ReLU(),
            nn.Dropout(params["dropout_l1"]),
            nn.Linear(params["n_units_l1"], params["n_units_l2"]),
            nn.ReLU(),
            nn.Dropout(params.get("dropout_l2", 0.1)),
            nn.Linear(params["n_units_l2"], 1),
        ).to(self.device)

    def optimize_hyperparams_and_weights(
        self, df_full_features, feature_names, date_column
    ):
        """Optuna를 활용해 하이퍼파라미터와 가중치를 탐색한다.

        CPU 코어 수만큼 병렬로 탐색 작업을 수행한다.
        """

        print("\n" + "=" * 50)
        print("--- Optimizing Hyperparameters & Weights via Backtesting ---")
        print("=" * 50)

        def objective(trial):
            n_units_l1 = trial.suggest_int("n_units_l1", 32, 128)
            params = {
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "dropout_l1": trial.suggest_float("dropout_l1", 0.1, 0.5),
                "n_units_l1": n_units_l1,
                "n_units_l2": trial.suggest_int("n_units_l2", 16, n_units_l1),
            }
            w_pattern = trial.suggest_float(
                "w_pattern", *self.config["PATTERN_WEIGHT_RANGE"]
            )
            w_mlp = trial.suggest_float("w_mlp", *self.config["ENSEMBLE_WEIGHT_RANGE"])
            errors = []
            for i in range(1, self.config["BACKTEST_MONTHS"] + 1):
                cutoff_date = self.first_of_month - pd.DateOffset(months=i)
                train_df, test_df = (
                    df_full_features[df_full_features["ds"] < cutoff_date],
                    df_full_features[df_full_features["ds"] == cutoff_date],
                )
                if len(train_df) < self.config["CV_SPLITS"] * 2 or test_df.empty:
                    continue
                temp_scaler = StandardScaler()
                X_train_scaled = temp_scaler.fit_transform(train_df[feature_names])
                y_train_tensor = (
                    torch.tensor(train_df["y_norm"].values, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )
                temp_mlp_model = self._build_mlp_model(len(feature_names), params)
                optimizer = torch.optim.Adam(
                    temp_mlp_model.parameters(), lr=params["lr"]
                )
                loss_fn = nn.MSELoss()
                for _ in range(self.config["BACKTEST_TUNE_EPOCHS"]):
                    temp_mlp_model.train()
                    loss = loss_fn(
                        temp_mlp_model(
                            torch.tensor(X_train_scaled, dtype=torch.float32).to(
                                self.device
                            )
                        ),
                        y_train_tensor,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                temp_lgbm_model = lgbm.LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=500,
                    seed=self.config["SEED"],
                )
                with self.suppress_stdout():
                    temp_lgbm_model.fit(
                        train_df[feature_names],
                        train_df["y_norm"],
                        eval_set=[(train_df[feature_names], train_df["y_norm"])],
                        callbacks=[lgbm.early_stopping(50, verbose=False)],
                    )
                X_test_scaled = temp_scaler.transform(test_df[feature_names])
                temp_mlp_model.eval()
                with torch.no_grad():
                    mlp_pred = temp_mlp_model(
                        torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
                    ).item()
                lgbm_pred = temp_lgbm_model.predict(test_df[feature_names])[0]
                ai_pred_norm = mlp_pred * w_mlp + lgbm_pred * (1 - w_mlp)
                ai_pred = (
                    ai_pred_norm
                    * self.config["Y_SCALE"]
                    * test_df["working_days"].iloc[0]
                )
                sim_date = self._compute_simulation_date(cutoff_date)
                actual_so_far_df = self.stock_base_df[
                    (self.stock_base_df[date_column].notna())
                    & (self.stock_base_df[date_column] >= cutoff_date)
                    & (self.stock_base_df[date_column] <= sim_date)
                ]
                actual_so_far = actual_so_far_df["NetAmount"].sum()
                working_days_so_far = len(
                    self.calendar_full[
                        (self.calendar_full["ds"] == cutoff_date)
                        & (self.calendar_full["Date"] <= sim_date)
                        & (self.calendar_full["IsHoliday"] == 0)
                    ]
                )
                max_len = len(self.daily_weights_del)
                cum_weight = sum(
                    self.daily_weights_del[: min(working_days_so_far, max_len)]
                )
                pattern_proj = (
                    (actual_so_far / cum_weight) if cum_weight > 0 else actual_so_far
                )
                final_pred = pattern_proj * w_pattern + ai_pred * (1 - w_pattern)
                y_true = test_df["y"].iloc[0]
                error = (
                    2
                    * np.abs(final_pred - y_true)
                    / (np.abs(y_true) + np.abs(final_pred) + 1e-8)
                )
                errors.append(error)
                del temp_mlp_model, temp_lgbm_model, temp_scaler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_error = np.mean(errors) * 100 if errors else float("inf")
            trial.report(avg_error, len(errors))
            if trial.should_prune():
                raise optuna.TrialPruned()
            return avg_error

        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        n_jobs = os.cpu_count() or 1
        study.optimize(
            objective,
            n_trials=self.config["N_TRIALS"],
            show_progress_bar=True,
            n_jobs=n_jobs,
        )
        self.best_params = study.best_trial.params
        # 초기 설정이 없을 수 있으므로 기본값을 사용해 안전하게 갱신
        self.config["PATTERN_WEIGHT"] = self.best_params.get(
            "w_pattern", self.config.get("PATTERN_WEIGHT", 0.5)
        )
        self.config["ENSEMBLE_WEIGHT_MLP"] = self.best_params.get(
            "w_mlp", self.config.get("ENSEMBLE_WEIGHT_MLP", 0.5)
        )
        print(f"\nBest hyperparameters and weights found: {self.best_params}")

    def refit_on_full_data(self, df_full_features, feature_names):
        """전체 데이터를 사용해 MLP 모델을 재학습한다."""

        print("Re-fitting final MLP model...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(df_full_features[feature_names])
        X_train, y_train = (
            torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device),
            torch.tensor(df_full_features["y_norm"].values, dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
        )
        self.mlp_model = self._build_mlp_model(len(feature_names), self.best_params)
        optimizer = torch.optim.Adam(
            self.mlp_model.parameters(), lr=self.best_params["lr"]
        )
        loss_fn = nn.MSELoss()
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.config["REFIT_LRP_PATIENCE"],
            factor=0.5,
            verbose=False,
        )
        for ep in tqdm(
            range(1, self.config["REFIT_EPOCHS"] + 1), desc="Re-fitting MLP"
        ):
            self.mlp_model.train()
            loss = loss_fn(self.mlp_model(X_train), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print(f"Learning rate converged. Stopping early at epoch {ep}.")
                break
        print("Final MLP model re-fitting completed.")

    def train_lightgbm_model(self, df_full_features, feature_names):
        """LightGBM 모델을 학습한다.

        GPU 사용 가능 시 GPU 모드로 학습한다.
        """

        print("Training LightGBM model...")
        X_train, y_train = df_full_features[feature_names], df_full_features["y_norm"]
        lgbm_params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": -1,
            "seed": self.config["SEED"],
        }
        if torch.cuda.is_available():
            lgbm_params.update({"device_type": "gpu", "gpu_device_id": 0})
        self.lgbm_model = lgbm.LGBMRegressor(**lgbm_params)
        with self.suppress_stdout():
            self.lgbm_model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[lgbm.early_stopping(100, verbose=False)],
            )
        train_preds = self.lgbm_model.predict(X_train)
        self.lgbm_residual_mae = np.mean(np.abs(y_train - train_preds))
        self.lgbm_feature_names_ = self.lgbm_model.booster_.feature_name()
        print(
            f"LightGBM model training completed. Residual MAE: {self.lgbm_residual_mae:.4f}"
        )

    def _get_monthly_prediction_distribution(self, input_df):
        """월 단위 예측 분포를 생성한다."""

        feature_names = self.lgbm_feature_names_
        input_scaled = self.scaler.transform(input_df[feature_names])
        self.mlp_model.train()
        with torch.no_grad():
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(
                self.device
            )
            batch_input = input_tensor.repeat(self.config["MC_ITERATIONS"], 1)
            mlp_mc_preds_norm = self.mlp_model(batch_input).squeeze().cpu().numpy()
        lgbm_pred_norm = self.lgbm_model.predict(input_df[feature_names])[0]
        lgbm_preds_dist = np.random.laplace(
            loc=lgbm_pred_norm,
            scale=self.lgbm_residual_mae,
            size=self.config["MC_ITERATIONS"],
        )
        w = self.config["ENSEMBLE_WEIGHT_MLP"]
        return (mlp_mc_preds_norm * w) + (lgbm_preds_dist * (1 - w))

    def _get_future_commitments(self, date_column):
        """미래에 확정된 매출 데이터를 조회한다."""

        print("Querying future commitments...")
        future_data = self.stock_base_df[
            self.stock_base_df[date_column] >= self.first_of_next_month
        ].copy()
        if future_data.empty:
            return pd.Series(dtype="float64")
        future_data["ds"] = future_data[date_column].dt.to_period("M").dt.to_timestamp()
        return future_data.groupby("ds")["NetAmount"].sum()

    def _generate_future_forecast(self, df_full_features, future_commitments):
        """미래 월의 하이브리드 예측 값을 생성한다."""

        print("Generating hybrid future forecast...")
        future_months_list = []
        last_known_features = df_full_features.iloc[-1]
        feature_names = self.lgbm_feature_names_
        date_range = pd.date_range(
            start=self.first_of_next_month, end=self.year_end, freq="MS"
        )
        for m_date in date_range:
            future_month_wd = self.wd_lookup.get(m_date, int(self.wd_lookup.median()))
            future_input_data = {
                "t": (m_date - df_full_features["ds"].min()).days // 30,
                "working_days": future_month_wd,
                "month_sin": np.sin(2 * np.pi * m_date.month / 12),
                "month_cos": np.cos(2 * np.pi * m_date.month / 12),
            }
            for fname in feature_names:
                if "lag12" in fname:
                    last_year_date = m_date - pd.DateOffset(years=1)
                    last_year_value = df_full_features[
                        df_full_features["ds"] == last_year_date
                    ]["y_norm"].values
                    if len(last_year_value) > 0:
                        future_input_data[fname] = last_year_value[0]
                    else:
                        future_input_data[fname] = last_known_features[fname]
                elif "lag" in fname or "ma" in fname or "std" in fname:
                    future_input_data[fname] = last_known_features[fname]
            future_input_df = pd.DataFrame([future_input_data])[feature_names]
            ai_pred_dist = self._get_monthly_prediction_distribution(future_input_df)
            ai_pred_scaled_mean = (
                np.mean(ai_pred_dist) * self.config["Y_SCALE"] * future_month_wd
            )
            adjustment_factor = self.seasonal_factors.get(m_date.month, 1.0)
            adjusted_ai_mean = ai_pred_scaled_mean * adjustment_factor
            known_commitment = future_commitments.get(m_date, 0.0)
            final_mean = max(adjusted_ai_mean, known_commitment)
            forecast_source = (
                "AI 주도" if final_mean == adjusted_ai_mean else "확정매출 기반"
            )
            final_lower, final_upper = (
                final_mean - self.conformal_delta,
                final_mean + self.conformal_delta,
            )
            future_months_list.append(
                {
                    "ds": m_date,
                    "yhat": final_mean,
                    "yhat_lower": final_lower,
                    "yhat_upper": final_upper,
                    "ai_forecast": adjusted_ai_mean,
                    "commitment": known_commitment,
                    "source": forecast_source,
                }
            )
        return pd.DataFrame(future_months_list)

    def _plot_residuals_histogram(self, backtest_eval_df):
        """백테스트 잔차 분포를 히스토그램으로 저장한다."""

        if backtest_eval_df.empty:
            return
        print("Generating residuals histogram...")
        residuals = backtest_eval_df["y_true"] - backtest_eval_df["yhat"]
        path = self.config["RESIDUAL_PLOT_SAVE_PATH"]
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
        plt.axvline(
            residuals.mean(),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"평균 오차: {residuals.mean():,.0f}",
        )
        plt.title("백테스트 예측 오차(잔차) 분포", fontsize=16)
        plt.xlabel("오차 (실제값 - 예측값)")
        plt.ylabel("빈도")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        formatter = mticker.FuncFormatter(lambda x, p: f"{x / 1e8:.0f}억")
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.savefig(path, dpi=150)
        print(f"Residuals histogram saved to {path}")
        plt.close()

    def _plot_forecast_results(self, df_hist, current_month_pred, future_preds):
        """예측 결과를 그래프로 저장한다."""

        print("Generating forecast visualization...")
        path = self.config["PLOT_SAVE_PATH"]
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.figure(figsize=(15, 8))
        plt.plot(df_hist["ds"], df_hist["y"], "b-", label="과거 실적")
        current_ds = self.first_of_month
        error = [
            [current_month_pred["mean"] - current_month_pred["lower"]],
            [current_month_pred["upper"] - current_month_pred["mean"]],
        ]
        plt.errorbar(
            current_ds,
            current_month_pred["mean"],
            yerr=error,
            fmt="o",
            color="green",
            capsize=5,
            label="현재 월 예측",
        )
        if not future_preds.empty:
            plt.plot(
                future_preds["ds"],
                future_preds["yhat"],
                "r--",
                label="미래 예측 (하이브리드)",
            )
            plt.fill_between(
                future_preds["ds"],
                future_preds["yhat_lower"],
                future_preds["yhat_upper"],
                color="red",
                alpha=0.15,
                label="95% 신뢰 구간 (보정됨)",
            )
        plt.title("종합 매출 예측 결과", fontsize=18)
        plt.xlabel("날짜", fontsize=12)
        plt.ylabel("매출액", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        formatter = mticker.FuncFormatter(lambda x, p: f"{x / 1e8:.0f}억")
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"Forecast plot saved to {path}")
        plt.close()

    def run_backtesting_for_reporting(self, df_full_features, feature_names):
        """최종 보고를 위한 백테스트를 수행한다."""

        print("\n" + "=" * 50)
        print("--- Running Final Backtesting for Reporting ---")
        print("=" * 50)
        backtest_months, results = self.config["BACKTEST_MONTHS"], []
        w_mlp = self.config["ENSEMBLE_WEIGHT_MLP"]
        for i in tqdm(range(1, backtest_months + 1), desc="Final Backtesting"):
            cutoff_date = self.first_of_month - pd.DateOffset(months=i)
            train_df, test_df = (
                df_full_features[df_full_features["ds"] < cutoff_date],
                df_full_features[df_full_features["ds"] == cutoff_date],
            )
            if len(train_df) < 2 or test_df.empty:
                continue
            temp_scaler = StandardScaler()
            X_train_scaled, X_test_scaled = (
                temp_scaler.fit_transform(train_df[feature_names]),
                temp_scaler.transform(test_df[feature_names]),
            )
            y_train_tensor = (
                torch.tensor(train_df["y_norm"].values, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )
            temp_mlp_model = self._build_mlp_model(len(feature_names), self.best_params)
            temp_optimizer = torch.optim.Adam(
                temp_mlp_model.parameters(), lr=self.best_params["lr"]
            )
            loss_fn = nn.MSELoss()
            for _ in range(self.config["BACKTEST_TUNE_EPOCHS"]):
                temp_mlp_model.train()
                loss = loss_fn(
                    temp_mlp_model(
                        torch.tensor(X_train_scaled, dtype=torch.float32).to(
                            self.device
                        )
                    ),
                    y_train_tensor,
                )
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
            temp_lgbm_model = lgbm.LGBMRegressor(
                objective="regression_l1", n_estimators=500, seed=self.config["SEED"]
            )
            with self.suppress_stdout():
                temp_lgbm_model.fit(
                    train_df[feature_names],
                    train_df["y_norm"],
                    eval_set=[(train_df[feature_names], train_df["y_norm"])],
                    callbacks=[lgbm.early_stopping(50, verbose=False)],
                )
            temp_mlp_model.eval()
            with torch.no_grad():
                mlp_pred = temp_mlp_model(
                    torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
                ).item()
            lgbm_pred = temp_lgbm_model.predict(test_df[feature_names])[0]
            ensemble_pred_norm = mlp_pred * w_mlp + lgbm_pred * (1 - w_mlp)
            y_true = test_df["y"].iloc[0]
            yhat = (
                ensemble_pred_norm
                * self.config["Y_SCALE"]
                * test_df["working_days"].iloc[0]
            )
            results.append({"ds": cutoff_date, "y_true": y_true, "yhat": yhat})
            del temp_mlp_model, temp_lgbm_model, temp_scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not results:
            return pd.DataFrame(), None
        eval_df = pd.DataFrame(results)
        mae = np.mean(np.abs(eval_df["y_true"] - eval_df["yhat"]))
        smape_denominator = np.abs(eval_df["y_true"]) + np.abs(eval_df["yhat"])
        smape = (
            np.mean(
                2
                * np.abs(eval_df["yhat"] - eval_df["y_true"])
                / np.where(smape_denominator == 0, 1, smape_denominator)
            )
            * 100
        )
        return eval_df, {"SMAPE": smape, "MAE": mae}

    def _calculate_conformal_delta(self, backtest_eval_df):
        """Conformal 방법으로 예측 구간을 보정한다."""

        print("Calibrating prediction interval with Conformal method...")
        if backtest_eval_df.empty:
            print("Warning: Backtest data is empty.")
            self.conformal_delta = 0.0
            return
        target_alpha = self.config["CONFORMAL_ALPHA"]
        abs_residuals = np.abs(backtest_eval_df["y_true"] - backtest_eval_df["yhat"])
        delta_q = np.quantile(abs_residuals, 1 - target_alpha)
        self.conformal_delta = delta_q
        print(
            f"Conformal delta (q_hat) for {100 * (1 - target_alpha):.1f}% CI calculated: {delta_q:,.0f} KRW"
        )

    def _save_report_to_file(self, report_data):
        """예측 결과 리포트를 텍스트로 저장한다."""

        path = self.config["REPORT_SAVE_PATH"]
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        print(f"Saving forecast report to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("--- 최종 매출 예측 및 의사결정 지원 리포트 ---\n")
            f.write(f"--- (예측 생성일: {self.today.strftime('%Y-%m-%d')}) ---\n")
            f.write("=" * 50 + "\n\n")
            if report_data.get("evaluation_results"):
                eval_res = report_data["evaluation_results"]
                f.write("### AI 모델 성능 평가 (지난 12개월 백테스트 결과)\n")
                f.write(
                    f"- SMAPE (평균 오차율)       : {eval_res.get('SMAPE', 0):.2f} %\n"
                )
                f.write(
                    f"- MAE (평균 오차액)         : {eval_res.get('MAE', 0):,.0f} KRW\n"
                )
                f.write(
                    f"- PICP (신뢰구간 포함률) : {eval_res.get('PICP', 0):.2f} % (Conformal 보정 후)\n"
                )
                f.write(
                    f"- 보정된 구간 너비 (상하) : ± {report_data.get('conformal_delta', 0):,.0f} KRW\n\n"
                )
            f.write(f"### 현재 월 진행상황 ({self.today.strftime('%Y-%m-%d')} 기준)\n")
            if report_data.get("optimal_weights"):
                f.write(
                    f"- 확정된 최적 가중치 (패턴/앙상블): {report_data['optimal_weights']['w_pattern_dynamic']:.2f} / {report_data['optimal_weights']['w_mlp']:.2f}\n"
                )
            for key, value in report_data["current_month_progress"].items():
                f.write(f"- {key}: {value:,.0f} KRW\n")
            f.write("\n")
            f.write("### 최종 당월 예측 (블렌딩)\n")
            f.write(
                f"- 예측치: {report_data['current_month_final']['mean']:,.0f} KRW\n"
            )
            f.write(
                f"- 95% 신뢰구간: {report_data['current_month_final']['lower']:,.0f} ~ {report_data['current_month_final']['upper']:,.0f} KRW\n\n"
            )
            f.write("### 미래 월별 상세 예측 (하이브리드 분석)\n")
            if not report_data["future_breakdown"].empty:
                for _, row in report_data["future_breakdown"].iterrows():
                    f.write(
                        f"- **{row['ds'].strftime('%Y-%m')} 예측** (Source: {row['source']})\n"
                    )
                    f.write(f"  - AI 기본 예측: {row['ai_forecast']:>15,.0f} KRW\n")
                    f.write(f"  - 확정 선행 발주: {row['commitment']:>15,.0f} KRW\n")
                    f.write(f"  - 최종 보정 예측: {row['yhat']:>15,.0f} KRW\n")
            else:
                f.write("- 예측할 미래 월이 없습니다.\n")
            f.write("\n")
            f.write("### 최종 연간 예측 (종합)\n")
            f.write(f"- 예측치: {report_data['annual_final']['mean']:,.0f} KRW\n")
            f.write(
                f"- 95% 신뢰구간: {report_data['annual_final']['lower']:,.0f} ~ {report_data['annual_final']['upper']:,.0f} KRW\n"
            )
            f.write("\n" + "=" * 50)
        print("Report saving complete.")

    def save_model(self):
        """훈련된 모델과 스케일러를 파일로 저장한다."""

        if self.mlp_model and self.scaler and self.lgbm_model:
            path = self.config["MODEL_SAVE_PATH"]
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            versioned_path = f"{base}_{timestamp}{ext if ext else '.pkl'}"
            print(f"Saving models and scaler to {versioned_path}...")
            lgbm_txt_path = f"{base}_{timestamp}.txt"
            with self.suppress_stdout():
                self.lgbm_model.booster_.save_model(lgbm_txt_path)
            print(f"LGBM model saved to {lgbm_txt_path}")
            joblib.dump(
                {
                    "mlp_model_state_dict": self.mlp_model.to("cpu").state_dict(),
                    "lgbm_model_path": lgbm_txt_path,
                    "scaler": self.scaler,
                    "best_params": self.best_params,
                    "seasonal_factors": self.seasonal_factors,
                    "lgbm_residual_mae": self.lgbm_residual_mae,
                    "conformal_delta": self.conformal_delta,
                    "lgbm_feature_names": self.lgbm_feature_names_,
                },
                versioned_path,
            )
            print("Other objects saved complete.")
        else:
            print("Models not trained yet.")

    def load_model(self, path=None):
        """저장된 모델을 로드한다."""

        if path is None:
            model_dir = os.path.dirname(self.config["MODEL_SAVE_PATH"])
            if not os.path.exists(model_dir):
                print(f"Model directory not found: {model_dir}.")
                return False
            files = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".pkl")
            ]
            if not files:
                print(f"No .pkl model files found in {model_dir}.")
                return False
            path = max(files, key=os.path.getctime)
        if not os.path.exists(path):
            print(f"Model file not found at {path}.")
            return False
        print(f"Loading latest model and scaler from {path}...")
        try:
            checkpoint = joblib.load(path)
            self.scaler, self.best_params = (
                checkpoint["scaler"],
                checkpoint["best_params"],
            )
            self.seasonal_factors, self.lgbm_residual_mae = (
                checkpoint.get("seasonal_factors", {}),
                checkpoint.get("lgbm_residual_mae", 0.01),
            )
            self.conformal_delta = checkpoint.get("conformal_delta", 0.0)
            lgbm_path = checkpoint["lgbm_model_path"]
            self.lgbm_model = lgbm.Booster(model_file=lgbm_path)
            self.lgbm_feature_names_ = checkpoint.get(
                "lgbm_feature_names", self.lgbm_model.feature_name()
            )
            n_features = self.scaler.n_features_in_
            self.mlp_model = self._build_mlp_model(n_features, self.best_params)
            self.mlp_model.load_state_dict(checkpoint["mlp_model_state_dict"])
            self.mlp_model.to(self.device)
            print("Load complete.")
            return True
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False

    def _blend_current_month_projection(
        self,
        pattern_based_projection: float,
        ai_prediction: float,
        actual_so_far: float,
        working_days_so_far: int,
        total_working_days: int,
    ) -> tuple[float, float]:
        """패턴 예측과 AI 예측을 동적으로 블렌딩한다.

        Args:
            pattern_based_projection (float): 패턴 기반 월간 예상 매출.
            ai_prediction (float): AI 모델 예측치.
            actual_so_far (float): 현재까지 확정된 매출.
            working_days_so_far (int): 경과한 영업일 수.
            total_working_days (int): 전체 영업일 수.

        Returns:
            tuple[float, float]: 하한 보정이 적용된 예측치와
            동적으로 계산된 패턴 가중치.
        """

        remaining_ratio = max(0.0, 1 - working_days_so_far / total_working_days)
        w_base = self.config["PATTERN_WEIGHT"]
        w_dynamic = w_base * remaining_ratio
        blended = (pattern_based_projection * w_dynamic) + (
            ai_prediction * (1 - w_dynamic)
        )
        return max(blended, actual_so_far), w_dynamic

    def run(self, date_column="DeliveryDate"):
        """예측 파이프라인 전체를 실행한다.

        Args:
            date_column (str): 기준 날짜 컬럼명.
        """
        if not self._load_data_from_db(date_column):
            return

        self.get_complete_calendar()

        df_hist = self.stock_base_df[
            (self.stock_base_df[date_column].notna())
            & (self.stock_base_df[date_column] < self.first_of_month)
        ].copy()
        df_hist["ds"] = df_hist[date_column].dt.to_period("M").dt.to_timestamp()
        df_hist = df_hist.groupby("ds").agg(y=("NetAmount", "sum")).reset_index()

        if len(df_hist) < self.config["CV_SPLITS"] * 2:
            print(
                f"Not enough historical data. Need > {self.config['CV_SPLITS'] * 2} months."
            )
            return

        self.calculate_daily_weights(date_column)
        self.calculate_seasonal_factors(df_hist)
        df_full_features, feature_names = self.create_features(df_hist.copy())

        self.optimize_hyperparams_and_weights(
            df_full_features, feature_names, date_column
        )

        self.refit_on_full_data(df_full_features, feature_names)
        self.train_lightgbm_model(df_full_features, feature_names)

        backtest_results_df, evaluation_results = self.run_backtesting_for_reporting(
            df_full_features, feature_names
        )

        if backtest_results_df is not None:
            self._calculate_conformal_delta(backtest_results_df)
            self._plot_residuals_histogram(backtest_results_df)
            backtest_results_df["yhat_lower_conf"] = (
                backtest_results_df["yhat"] - self.conformal_delta
            )
            backtest_results_df["yhat_upper_conf"] = (
                backtest_results_df["yhat"] + self.conformal_delta
            )
            evaluation_results["PICP"] = (
                np.mean(
                    (
                        backtest_results_df["y_true"]
                        >= backtest_results_df["yhat_lower_conf"]
                    )
                    & (
                        backtest_results_df["y_true"]
                        <= backtest_results_df["yhat_upper_conf"]
                    )
                )
                * 100
            )

        model_pred_dist = self._get_monthly_prediction_distribution(
            df_full_features.tail(1)
        )
        current_month_wd = self.wd_lookup.get(
            self.first_of_month, int(self.wd_lookup.median())
        )
        model_based_monthly_pred_mean = (
            np.mean(model_pred_dist) * self.config["Y_SCALE"] * current_month_wd
        )

        actual_del_todate_df = self.stock_base_df[
            self.stock_base_df[date_column].dt.to_period("M").dt.to_timestamp()
            == self.first_of_month
        ]
        actual_del_todate = actual_del_todate_df[
            actual_del_todate_df[date_column] <= self.today
        ]["NetAmount"].sum()

        working_days_so_far = len(
            self.calendar_full[
                (self.calendar_full["ds"] == self.first_of_month)
                & (self.calendar_full["Date"] <= self.today)
                & (self.calendar_full["IsHoliday"] == 0)
            ]
        )
        max_len = len(self.daily_weights_del)
        cumulative_weight_so_far = sum(
            self.daily_weights_del[: min(working_days_so_far, max_len)]
        )
        pattern_based_projection = (
            (actual_del_todate / cumulative_weight_so_far)
            if cumulative_weight_so_far > 0
            else actual_del_todate
        )

        final_monthly_mean, w_pattern_dynamic = self._blend_current_month_projection(
            pattern_based_projection,
            model_based_monthly_pred_mean,
            actual_del_todate,
            working_days_so_far,
            current_month_wd,
        )
        final_monthly_lower = final_monthly_mean - self.conformal_delta
        final_monthly_upper = final_monthly_mean + self.conformal_delta

        future_commitments = self._get_future_commitments(date_column)
        future_preds_df = self._generate_future_forecast(
            df_full_features, future_commitments
        )

        ytd_actual = df_hist[df_hist["ds"].dt.year == self.today.year]["y"].sum()
        annual_pred_mean = (
            ytd_actual + final_monthly_mean + future_preds_df["yhat"].sum()
        )
        annual_pred_lower = (
            ytd_actual + final_monthly_lower + future_preds_df["yhat_lower"].sum()
        )
        annual_pred_upper = (
            ytd_actual + final_monthly_upper + future_preds_df["yhat_upper"].sum()
        )

        report_data = {
            "evaluation_results": {
                **evaluation_results,
                "Conformal_Delta": self.conformal_delta,
            }
            if evaluation_results
            else None,
            "optimal_weights": {
                "w_pattern_dynamic": w_pattern_dynamic,
                "w_mlp": self.config["ENSEMBLE_WEIGHT_MLP"],
            },
            "current_month_progress": {
                "Actual Sales So Far": actual_del_todate,
                "Pattern-based Projection": pattern_based_projection,
                "AI Model Ensemble Forecast": model_based_monthly_pred_mean,
            },
            "current_month_final": {
                "mean": final_monthly_mean,
                "lower": final_monthly_lower,
                "upper": final_monthly_upper,
            },
            "future_breakdown": future_preds_df,
            "annual_final": {
                "mean": annual_pred_mean,
                "lower": annual_pred_lower,
                "upper": annual_pred_upper,
            },
        }

        self._save_report_to_file(report_data)
        self._plot_forecast_results(
            df_hist, report_data["current_month_final"], future_preds_df
        )
        self.save_model()


if __name__ == "__main__":
    config = {
        "DB_PATH": "./data/import/base.duckdb",
        "MODEL_SAVE_PATH": "./data/models/nprophet_model",
        "PLOT_SAVE_PATH": "./data/reports/forecast_visualization.png",
        "REPORT_SAVE_PATH": "./data/reports/forecast_report.txt",
        "RESIDUAL_PLOT_SAVE_PATH": "./data/reports/residuals_histogram.png",
        "TRAINING_START_DATE": "2021-01-01",
        "SEED": 42,
        "Y_SCALE": 1e7,
        # 앙상블 가중치 기본값을 명시적으로 설정
        "PATTERN_WEIGHT": 0.5,
        "ENSEMBLE_WEIGHT_MLP": 0.5,
        "ENSEMBLE_WEIGHT_RANGE": (0.4, 0.8),
        "PATTERN_WEIGHT_RANGE": (0.3, 0.9),
        "REFIT_EPOCHS": 500,
        "REFIT_LRP_PATIENCE": 20,
        "TUNE_EPOCHS": 100,
        "N_TRIALS": 70,
        "CV_SPLITS": 3,
        "MC_ITERATIONS": 100,
        "BACKTEST_MONTHS": 12,
        "BACKTEST_TUNE_EPOCHS": 50,
        "CONFORMAL_ALPHA": 0.05,
        "SIMULATION_DAY_OF_MONTH": 20,
    }

    forecaster = NProphetForecaster(config)
    forecaster.run(date_column="DeliveryDate")
