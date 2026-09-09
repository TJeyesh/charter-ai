"""
Charter-AI — Idle Time & Demurrage Risk Predictor.

Predicts expected vessel waiting time at destination ports based on
congestion, weather, seasonal, and vessel-class features.

NOTE: Scaffold only — ML implementation deferred.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IdleTimePrediction:
    """Predicted idle/waiting time at a port."""
    port_id: str
    predicted_date: date
    predicted_waiting_days: float
    prediction_lower: float  # Lower bound (optimistic)
    prediction_upper: float  # Upper bound (pessimistic)
    confidence: float
    demurrage_risk_level: str  # "low", "moderate", "high", "critical"
    key_factors: dict  # Top contributing features


class IdleTimePredictor:
    """
    XGBoost regression model for predicting vessel idle time at port.

    Features used:
    - Current congestion (vessels_waiting, berth_occupancy_pct)
    - Historical congestion patterns (lag, rolling mean)
    - Weather conditions (wind, wave, sea state)
    - Calendar (month, day_of_week, is_monsoon, is_cyclone_season)
    - Vessel class (larger vessels may wait longer for deep-draft berths)
    """

    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train idle time prediction model.

        Args:
            train_df: DataFrame with congestion, weather, and calendar features
                     plus target column 'avg_waiting_time_days'.
        """
        # TODO: XGBoost regression with TimeSeriesSplit
        raise NotImplementedError("Idle time model training deferred to ML phase.")

    def predict(
        self,
        port_id: str,
        target_date: date,
        vessel_class: Optional[str] = None,
    ) -> IdleTimePrediction:
        """
        Predict expected waiting time at a port on a given date.

        Args:
            port_id: Destination port identifier.
            target_date: Expected arrival date.
            vessel_class: Vessel class (affects berth availability).

        Returns:
            IdleTimePrediction with expected days and confidence.
        """
        # TODO: Implement prediction with SHAP explanations for key factors
        raise NotImplementedError("Idle time prediction deferred to ML phase.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
