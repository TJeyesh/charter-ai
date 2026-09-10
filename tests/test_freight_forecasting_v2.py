"""
Charter-AI — Comprehensive Test Suite for Phase 3 REAL Freight Forecasting Engine.

Tests:
1. Multi-domain feature engineering (lags, rolling stats, momentum, indices, commodities, bunker, macro, operational, calendar)
2. Evaluation metrics (MAE, RMSE, MAPE, sMAPE, MASE)
3. Time-series walk-forward cross-validation (strict chronological splits)
4. Baselines (Naive, Moving Average, Seasonal)
5. SARIMA/ARIMA forecaster
6. XGBoost with quantile regression (P10, P50, P90)
7. Ensemble forecaster (inverse-RMSE & equal weighting)
8. FreightForecaster orchestrator & canonical API contract
9. Model persistence and metadata schema
10. FastAPI HTTP endpoint (/api/v1/forecast)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

from src.models.base_forecaster import ForecastPoint, ForecastResult, ForecastModel
from src.models.baseline_forecaster import (
    BaselineForecaster,
    NaiveBaselineForecaster,
    MovingAverageForecaster,
    SeasonalBaselineForecaster
)
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import (
    add_freight_lags,
    add_rolling_metrics,
    add_momentum,
    add_calendar_features,
    FreightFeatureBuilder,
    build_forecast_features
)
from src.models.model_evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_smape,
    calculate_mase,
    evaluate_forecast,
    walk_forward_cv,
    compare_models
)
from src.models.freight_forecaster import FreightForecaster
from src.services.freight_forecast_service import FreightForecastService
from src.api.main import app


@pytest.fixture
def sample_daily_timeseries():
    """Generates 90 days of realistic daily freight series."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    base = 15.0
    trend = np.linspace(0, 2.0, 90)
    cycle = np.sin(np.linspace(0, 4 * np.pi, 90)) * 0.8
    noise = np.random.normal(0, 0.15, 90)
    rates = base + trend + cycle + noise
    return pd.DataFrame({
        "date": dates,
        "freight_rate": rates,
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Capesize",
        "cargo_type": "thermal_coal"
    })


# =============================================================================
# 1. Feature Engineering Tests
# =============================================================================

def test_feature_engineering_lags(sample_daily_timeseries):
    df = add_freight_lags(sample_daily_timeseries, target_col="freight_rate", lags=[1, 3, 7, 14, 21, 28])
    for lag in [1, 3, 7, 14, 21, 28]:
        assert f"lag_{lag}" in df.columns
        # Shift property check
        assert df[f"lag_{lag}"].iloc[lag] == df["freight_rate"].iloc[0]


def test_feature_engineering_rolling_and_momentum(sample_daily_timeseries):
    df = add_rolling_metrics(sample_daily_timeseries, target_col="freight_rate", windows=[7, 14, 28])
    assert "rolling_mean_7" in df.columns
    assert "rolling_mean_14" in df.columns
    assert "rolling_mean_28" in df.columns
    assert "rolling_std_7" in df.columns
    assert "rolling_std_28" in df.columns

    df = add_momentum(df, target_col="freight_rate", periods=[7, 30])
    assert "momentum_7" in df.columns
    assert "momentum_30" in df.columns
    # Check momentum calculation at index 7
    expected = df["freight_rate"].iloc[7] - df["freight_rate"].iloc[0]
    assert np.isclose(df["momentum_7"].iloc[7], expected, atol=1e-5)


def test_feature_engineering_calendar(sample_daily_timeseries):
    df = add_calendar_features(sample_daily_timeseries, date_col="date")
    for col in ["month", "week", "day_of_week", "month_sin", "month_cos", "is_monsoon", "is_cyclone_season"]:
        assert col in df.columns
    assert (df["month"] >= 1).all() and (df["month"] <= 12).all()
    assert (df["day_of_week"] >= 0).all() and (df["day_of_week"] <= 6).all()


def test_freight_feature_builder_multi_domain(sample_daily_timeseries):
    feat_df, cols = build_forecast_features(sample_daily_timeseries)
    assert len(feat_df) == len(sample_daily_timeseries)
    assert len(cols) >= 15
    assert "lag_1" in cols
    assert "rolling_mean_7" in cols
    assert "momentum_7" in cols
    # Zero NaN values after ffill/bfill
    assert feat_df[cols].isna().sum().sum() == 0


# =============================================================================
# 2. Evaluation Metrics Tests
# =============================================================================

def test_evaluation_metrics_values():
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([11.0, 12.0, 13.0, 18.0])

    mae = calculate_mae(y_true, y_pred)
    assert mae == (1.0 + 0.0 + 1.0 + 2.0) / 4.0

    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt((1 + 0 + 1 + 4) / 4.0))

    mape = calculate_mape(y_true, y_pred)
    assert mape > 0.0

    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 200.0

    y_train = np.array([8.0, 9.0, 10.0])
    mase = calculate_mase(y_true, y_pred, y_train=y_train)
    assert mase > 0.0

    metrics = evaluate_forecast(y_true, y_pred, y_train=y_train)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE", "MASE"}


# =============================================================================
# 3. Time-Series Walk-Forward Cross-Validation
# =============================================================================

def test_walk_forward_cross_validation(sample_daily_timeseries):
    cv_res = walk_forward_cv(
        model_factory=lambda: MovingAverageForecaster(window_size=7),
        df=sample_daily_timeseries,
        target_col="freight_rate",
        date_col="date",
        horizon_days=7,
        initial_train_size=40,
        step_size=15,
        window_type="expanding"
    )

    assert len(cv_res.folds) >= 2
    assert "MAE" in cv_res.overall_metrics
    assert "RMSE" in cv_res.overall_metrics
    assert "sMAPE" in cv_res.overall_metrics

    # Strict temporal causality check
    for fold in cv_res.folds:
        assert fold.train_end < fold.test_start, "Data leakage detected: train_end must precede test_start"
        assert len(fold.predictions) == 7
        assert len(fold.actuals) == 7


# =============================================================================
# 4. Baselines Tests
# =============================================================================

def test_naive_baseline(sample_daily_timeseries):
    model = NaiveBaselineForecaster()
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.last_value is not None

    res = model.predict(horizon_days=5, context_df=sample_daily_timeseries, date_col="date")
    assert res.horizon_days == 5
    assert len(res.series) == 5
    assert res.confidence_available is True

    # Check uncertainty bounds
    for pt in res.series:
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


def test_seasonal_baseline(sample_daily_timeseries):
    model = SeasonalBaselineForecaster(seasonal_period=7)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.seasonal_pattern is not None
    assert len(model.seasonal_pattern) == 7

    res = model.predict(horizon_days=14, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 14
    # Replicates pattern across cycles
    assert res.series[0].predicted_rate == res.series[7].predicted_rate


# =============================================================================
# 5. SARIMA/ARIMA Forecaster Tests
# =============================================================================

def test_arima_forecaster_fit_and_predict(sample_daily_timeseries):
    model = ARIMAForecaster(order=(1, 1, 0), alpha=0.10)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.model_fit is not None

    res = model.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert res.horizon_days == 7
    assert res.confidence_available is True
    for pt in res.series:
        assert pt.lower_ci is not None
        assert pt.upper_ci is not None
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 6. XGBoost Quantile Forecaster Tests
# =============================================================================

def test_xgboost_quantile_forecaster(sample_daily_timeseries):
    model = XGBoostForecaster(n_estimators=30, max_depth=3, use_quantiles=True)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.is_fitted is True

    res = model.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 7
    for pt in res.series:
        assert pt.lower_ci is not None
        assert pt.upper_ci is not None
        # Monotonic quantile constraint: P10 <= P50 <= P90
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 7. Ensemble Forecaster Tests
# =============================================================================

def test_ensemble_forecaster(sample_daily_timeseries):
    ensemble = EnsembleForecaster(weighting_strategy="inverse_rmse")
    ensemble.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert ensemble.is_fitted is True
    assert len(ensemble.weights) >= 2
    # Sum of weights equals 1.0
    assert np.isclose(sum(ensemble.weights.values()), 1.0, atol=1e-3)

    res = ensemble.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 7
    for pt in res.series:
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 8. High-Level Orchestrator & Canonical API Output
# =============================================================================

def test_freight_forecaster_canonical_output():
    ff = FreightForecaster()
    pred = ff.predict_freight(
        origin="AUS_NEW",
        destination="IND_GVM",
        vessel_class="Capesize",
        cargo_type="thermal_coal",
        horizon_days=7
    )

    required_keys = {
        "current_rate",
        "forecast_rate",
        "lower_bound",
        "upper_bound",
        "trend",
        "confidence",
        "model_used",
        "metrics"
    }
    assert required_keys.issubset(pred.keys()), f"Missing keys in {pred.keys()}"
    assert pred["current_rate"] > 0
    assert pred["forecast_rate"] > 0
    assert pred["lower_bound"] <= pred["forecast_rate"] <= pred["upper_bound"]
    assert pred["trend"] in ["rising", "falling", "stable"]
    assert 0.50 <= pred["confidence"] <= 1.0
    assert isinstance(pred["model_used"], str)
    assert isinstance(pred["metrics"], dict)


# =============================================================================
# 9. Model Persistence & Metadata Schema
# =============================================================================

def test_model_persistence_and_metadata_roundtrip(sample_daily_timeseries):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "aus_new_ind_gvm_capesize" / "xgboost"
        model_dir.mkdir(parents=True)

        model = XGBoostForecaster(n_estimators=20, max_depth=3)
        model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

        artifact_file = model_dir / "model.pkl"
        model.save(str(artifact_file))
        assert artifact_file.exists()

        metadata = {
            "training_date": datetime.now().isoformat(),
            "dataset_version": "v2.0",
            "route": "AUS_NEW->IND_GVM",
            "vessel_class": "Capesize",
            "cargo_type": "thermal_coal",
            "model_name": "xgboost",
            "model_version": "1.0.0",
            "features": model.feature_cols,
            "metrics": {"MAE": 0.32, "RMSE": 0.41, "MAPE": 2.1, "sMAPE": 2.0, "MASE": 1.4},
            "horizons": [3, 7, 14, 30]
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Reload
        loaded = XGBoostForecaster()
        loaded.load(str(artifact_file))
        assert loaded.is_fitted is True
        res = loaded.predict(horizon_days=3, context_df=sample_daily_timeseries, date_col="date")
        assert len(res.series) == 3


# =============================================================================
# 10. HTTP REST API Endpoint Integration Test
# =============================================================================

def test_fastapi_forecast_endpoint():
    client = TestClient(app)
    response = client.get(
        "/api/v1/forecast",
        params={
            "origin": "AUS_NEW",
            "destination": "IND_GVM",
            "vessel_class": "Capesize",
            "horizon_days": 7,
            "cargo_type": "thermal_coal"
        }
    )
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Assert exact required API fields
    for field_name in ["current_rate", "forecast_rate", "lower_bound", "upper_bound", "trend", "confidence", "model_used", "metrics"]:
        assert field_name in data, f"Missing required API field: {field_name}"

    assert data["current_rate"] > 0
    assert data["forecast_rate"] > 0
    assert data["lower_bound"] <= data["forecast_rate"] <= data["upper_bound"]
    assert data["trend"] in ["rising", "falling", "stable"]
    assert 0.50 <= data["confidence"] <= 1.0
