"""
Charter-AI — Freight Rate Forecaster.

Ensemble of XGBoost (non-linear exogenous features) and SARIMA (seasonal
time-series structure) for freight rate prediction.

NOTE: This is the interface/scaffold only. Model training logic will be
implemented in the next phase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastPoint:
    """Single point in a forecast time series."""
    date: date
    predicted_rate: float
    lower_ci: float  # Lower bound of confidence interval
    upper_ci: float  # Upper bound of confidence interval


@dataclass
class ForecastResult:
    """Complete forecast result."""
    origin_port_id: str
    destination_port_id: str
    vessel_class: str
    horizon_days: int
    model_version: str
    series: List[ForecastPoint] = field(default_factory=list)
    mae: Optional[float] = None  # Mean Absolute Error on validation set
    mape: Optional[float] = None  # Mean Absolute Percentage Error

    @property
    def predicted_trend(self) -> str:
        """Determine if rates are trending up, down, or flat."""
        if len(self.series) < 2:
            return "insufficient_data"
        first = self.series[0].predicted_rate
        last = self.series[-1].predicted_rate
        pct_change = (last - first) / first * 100
        if pct_change > 5:
            return "rising"
        elif pct_change < -5:
            return "falling"
        return "stable"


class BaseForecaster(ABC):
    """Abstract base for all forecasting models."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Train the model on historical data."""
        ...

    @abstractmethod
    def predict(self, horizon_days: int) -> ForecastResult:
        """Generate a forecast for the specified horizon."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the trained model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Deserialize a trained model from disk."""
        ...


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based freight rate forecaster.

    Captures non-linear interactions between exogenous features
    (coal prices, FX, BDI, congestion) and freight rates.
    """

    def __init__(self):
        self.model = None
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        # TODO: Implement XGBoost training with TimeSeriesSplit cross-validation
        logger.info("XGBoostForecaster.fit() — not yet implemented")
        raise NotImplementedError("XGBoost training will be implemented in the ML phase.")

    def predict(self, horizon_days: int) -> ForecastResult:
        # TODO: Implement recursive multi-step forecasting
        raise NotImplementedError("XGBoost prediction will be implemented in the ML phase.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


class SARIMAForecaster(BaseForecaster):
    """
    SARIMA-based freight rate forecaster.

    Captures seasonal patterns (monsoon, cyclone season, Chinese demand cycles)
    in the time-series structure of freight rates.
    """

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        # TODO: Implement SARIMA fitting with auto_arima or manual order selection
        logger.info("SARIMAForecaster.fit() — not yet implemented")
        raise NotImplementedError("SARIMA training will be implemented in the ML phase.")

    def predict(self, horizon_days: int) -> ForecastResult:
        # TODO: Implement SARIMA forecasting with confidence intervals
        raise NotImplementedError("SARIMA prediction will be implemented in the ML phase.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


class EnsembleForecaster:
    """
    Ensemble forecaster combining XGBoost and SARIMA predictions.

    Weighting strategy:
    - Short horizon (≤14 days): Heavier SARIMA weight (0.6 SARIMA, 0.4 XGBoost)
    - Medium horizon (15-60 days): Equal weight (0.5, 0.5)
    - Long horizon (>60 days): Heavier XGBoost weight (0.4 SARIMA, 0.6 XGBoost)
    """

    def __init__(self):
        self.xgb = XGBoostForecaster()
        self.sarima = SARIMAForecaster()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        """Train both sub-models."""
        # TODO: Implement ensemble training
        raise NotImplementedError("Ensemble training will be implemented in the ML phase.")

    def predict(
        self,
        origin_port_id: str,
        destination_port_id: str,
        vessel_class: str,
        horizon_days: int = 30,
    ) -> ForecastResult:
        """
        Generate ensemble forecast.

        Args:
            origin_port_id: Origin port identifier.
            destination_port_id: Destination port identifier.
            vessel_class: Vessel class name.
            horizon_days: Number of days to forecast.

        Returns:
            ForecastResult with weighted ensemble predictions.
        """
        # TODO: Implement weighted ensemble combination
        raise NotImplementedError("Ensemble prediction will be implemented in the ML phase.")

    def _get_weights(self, horizon_days: int) -> tuple[float, float]:
        """Return (sarima_weight, xgb_weight) based on horizon."""
        if horizon_days <= 14:
            return 0.6, 0.4
        elif horizon_days <= 60:
            return 0.5, 0.5
        else:
            return 0.4, 0.6
