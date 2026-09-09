"""
Charter-AI — Model Training Orchestration.

Handles cross-validation, hyperparameter tuning, evaluation metrics,
and coordination of the training pipeline.

NOTE: Scaffold only — training implementation deferred.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    model_type: str  # "xgboost", "sarima", "ensemble"
    target_column: str = "freight_rate_usd_per_day"
    test_size_days: int = 90  # Hold out last N days for evaluation
    n_cv_splits: int = 5  # TimeSeriesSplit folds
    random_state: int = 42
    hyperparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Results from a training run."""
    model_type: str
    model_version: str
    trained_at: datetime
    metrics: Dict[str, float]  # {"mae": ..., "rmse": ..., "mape": ...}
    hyperparams: Dict[str, Any]
    artifact_path: str
    feature_importances: Optional[Dict[str, float]] = None


class TrainingPipeline:
    """
    Orchestrates model training end-to-end.

    Steps:
    1. Load and validate data
    2. Engineer features
    3. Split into train/validation/test
    4. Train model(s) with cross-validation
    5. Evaluate on test set
    6. Save model artifacts + register in model registry
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.settings = get_settings()

    def run(self, data_df: pd.DataFrame) -> TrainingResult:
        """
        Execute the full training pipeline.

        Args:
            data_df: Raw data DataFrame (pre-feature-engineering).

        Returns:
            TrainingResult with metrics and artifact path.
        """
        # TODO: Implement full training pipeline
        raise NotImplementedError("Training pipeline will be implemented in the ML phase.")

    def _time_series_split(
        self, df: pd.DataFrame
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time-series-aware train/validation splits.

        Unlike random k-fold, this respects temporal ordering —
        training data always precedes validation data.
        """
        # TODO: Implement expanding-window TimeSeriesSplit
        raise NotImplementedError

    def _evaluate(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        import numpy as np

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        return {"mae": mae, "rmse": rmse, "mape": mape}
