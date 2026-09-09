"""
Unified interface for Freight Forecasting models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple, Any
import pandas as pd

@dataclass
class ForecastPoint:
    """Single point in a forecast time series."""
    date: date
    predicted_rate: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None

@dataclass
class ForecastResult:
    """Complete forecast result returned by any model."""
    route: str
    vessel_type: str
    horizon_days: int
    model_used: str
    series: List[ForecastPoint] = field(default_factory=list)
    confidence_available: bool = False

class ForecastModel(ABC):
    """
    Abstract base class for all freight forecasting models.
    Enforces a unified interface for model training, prediction, evaluation, and persistence.
    """
    
    @abstractmethod
    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Train the model on the provided historical data.
        """
        pass

    @abstractmethod
    def predict(self, horizon_days: int, context_df: pd.DataFrame, date_col: str, **kwargs) -> ForecastResult:
        """
        Generate a forecast for the specified horizon.
        
        Args:
            horizon_days: Number of days into the future to forecast.
            context_df: Recent historical data required by the model to generate the forecast 
                        (e.g., for calculating lags or setting the ARIMA initial state).
            date_col: Name of the date column.
            **kwargs: Additional model-specific parameters (e.g. route, vessel_type).
        """
        pass

    @abstractmethod
    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        """
        Evaluate the model against a test dataset.
        Returns a dictionary of metrics.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Serialize the trained model to disk.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Deserialize a trained model from disk.
        """
        pass
