import pandas as pd
import os
import optuna
import torch
import lightgbm as lgbm
from nProphet import NProphetForecaster


def _minimal_forecaster(monkeypatch, cuda_available):
    monkeypatch.setattr(NProphetForecaster, "_connect_db", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    return NProphetForecaster({
        "SEED": 1,
        "DB_PATH": "dummy",
        "PATTERN_WEIGHT_RANGE": (0.3, 0.9),
        "ENSEMBLE_WEIGHT_RANGE": (0.4, 0.8),
        "N_TRIALS": 1,
    })


def test_lightgbm_uses_gpu_param(monkeypatch):
    class DummyBooster:
        def feature_name(self):
            return ["f1"]

    class DummyLGBM:
        def __init__(self, **params):
            self.params = params
            self.booster_ = DummyBooster()

        def fit(self, *args, **kwargs):
            return self

        def predict(self, X):
            return [0 for _ in range(len(X))]

        def get_params(self, deep=True):
            return self.params

    monkeypatch.setattr(lgbm, "LGBMRegressor", DummyLGBM)
    fc = _minimal_forecaster(monkeypatch, True)
    df = pd.DataFrame(
        {
            "y_norm": [0.1, 0.2],
            "t": [0, 1],
            "working_days": [20, 20],
            "month_sin": [0.0, 0.0],
            "month_cos": [1.0, 1.0],
            "y_norm_lag1": [0.0, 0.1],
            "y_norm_lag3": [0.0, 0.0],
            "y_norm_lag12": [0.0, 0.0],
            "y_norm_ma3": [0.0, 0.1],
            "y_norm_std3": [0.0, 0.05],
        }
    )
    features = [
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
    fc.train_lightgbm_model(df, features)
    params = fc.lgbm_model.get_params()
    assert params.get("device_type") == "gpu"
    assert params.get("verbosity") == -1


def test_optuna_passes_n_jobs(monkeypatch):
    captured = {}

    class DummyStudy:
        def __init__(self):
            self.best_trial = type("T", (), {"params": {}})()

        def optimize(self, objective, n_trials, show_progress_bar, n_jobs=None):
            captured["n_jobs"] = n_jobs

    monkeypatch.setattr(optuna, "create_study", lambda **_: DummyStudy())
    fc = _minimal_forecaster(monkeypatch, False)
    fc.optimize_hyperparams_and_weights(
        pd.DataFrame(), [], "DateColumn"
    )
    assert captured["n_jobs"] == (os.cpu_count() or 1)

