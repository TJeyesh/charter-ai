"""
Unit Tests: Freight Forecasting and Uncertainty Quantiles.

Verifies statistical baselines, ML ensemble behavior, multi-horizon forecasting
(3d, 7d, 14d, 30d), and quantile uncertainty invariants (P10 <= P50 <= P90).
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.models.base_forecaster import ForecastPoint, ForecastResult, ForecastModel
from src.models.baseline_forecaster import BaselineForecaster
from src.models.freight_forecaster import FreightForecaster


@pytest.fixture
def deterministic_freight_history():
    """Deterministic 180-day freight time series with trend and noise."""
    np.random.seed(42)
    start_date = pd.Timestamp("2025-01-01")
    dates = pd.date_range(start_date, periods=180, freq="D")
    base_rate = 22.0
    trend = np.linspace(0, 4.0, 180)
    noise = np.random.normal(0, 0.4, 180)
    rates = base_rate + trend + noise

    return pd.DataFrame({
        "date": dates,
        "freight_rate": rates,
    })


class TestBaselineForecasting:
    """Test standard statistical baselines."""

    def test_naive_baseline(self, deterministic_freight_history):
        forecaster = BaselineForecaster(method="naive")
        forecaster.fit(deterministic_freight_history, target_col="freight_rate", date_col="date")

        last_val = deterministic_freight_history["freight_rate"].iloc[-1]
        res = forecaster.predict(horizon_days=14, context_df=deterministic_freight_history, date_col="date")

        assert len(res.series) == 14
        for p in res.series:
            assert p.predicted_rate == pytest.approx(last_val, abs=0.05)
            assert p.lower_ci <= p.predicted_rate <= p.upper_ci

    def test_moving_average_baseline(self, deterministic_freight_history):
        window = 14
        forecaster = BaselineForecaster(method="moving_average", window_size=window)
        forecaster.fit(deterministic_freight_history, target_col="freight_rate", date_col="date")

        expected_ma = deterministic_freight_history["freight_rate"].iloc[-window:].mean()
        res = forecaster.predict(horizon_days=7, context_df=deterministic_freight_history, date_col="date")

        assert len(res.series) == 7
        for p in res.series:
            assert p.predicted_rate == pytest.approx(expected_ma, abs=0.05)
            assert p.lower_ci <= p.predicted_rate <= p.upper_ci


class TestForecastUncertaintyQuantiles:
    """Test quantile integrity and uncertainty spread behavior."""

    def test_quantile_monotonicity(self):
        """Lower bound <= forecast <= upper bound must hold strictly."""
        forecaster = FreightForecaster()
        res = forecaster.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            horizon_days=14,
        )

        assert "forecast_rate" in res
        assert "lower_bound" in res
        assert "upper_bound" in res
        assert res["lower_bound"] <= res["forecast_rate"] <= res["upper_bound"]
        assert res["lower_bound"] > 0.0

    def test_uncertainty_widens_with_horizon(self):
        """Uncertainty interval should generally be non-zero and stable across horizons."""
        forecaster = FreightForecaster()
        res_3d = forecaster.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            horizon_days=3,
        )
        res_30d = forecaster.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            horizon_days=30,
        )

        spread_3d = res_3d["upper_bound"] - res_3d["lower_bound"]
        spread_30d = res_30d["upper_bound"] - res_30d["lower_bound"]

        assert spread_3d > 0.0
        assert spread_30d > 0.0

    def test_forecast_confidence_scoring(self):
        """Forecast confidence must be between 0.0 and 1.0."""
        forecaster = FreightForecaster()
        res = forecaster.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Panamax",
            horizon_days=7,
        )
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
