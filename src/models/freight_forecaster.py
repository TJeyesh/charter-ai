"""
Charter-AI — Freight Forecaster Engine & Pipeline Orchestrator.

Provides the end-to-end forecasting pipeline for dry-bulk freight rates across:
  origin -> destination -> vessel_class -> cargo_type
for horizons:
  3 days, 7 days, 14 days, 30 days

Exposes the canonical output dictionary:
{
    "current_rate": float,
    "forecast_rate": float,
    "lower_bound": float,
    "upper_bound": float,
    "trend": "rising" | "falling" | "stable",
    "confidence": float,
    "model_used": str,
    "metrics": {"MAE": float, "RMSE": float, "MAPE": float, "sMAPE": float, "MASE": float}
}
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json

from src.models.base_forecaster import ForecastModel, ForecastPoint, ForecastResult
from src.models.baseline_forecaster import BaselineForecaster, NaiveBaselineForecaster, MovingAverageForecaster, SeasonalBaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import FreightFeatureBuilder, build_forecast_features
from src.models.model_evaluation import evaluate_forecast, walk_forward_cv
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Re-exports for complete backward compatibility
__all__ = [
    "ForecastModel",
    "ForecastPoint",
    "ForecastResult",
    "FreightForecaster",
    "BaselineForecaster",
    "NaiveBaselineForecaster",
    "MovingAverageForecaster",
    "SeasonalBaselineForecaster",
    "ARIMAForecaster",
    "XGBoostForecaster",
    "EnsembleForecaster",
]


class FreightForecaster:
    """
    High-level orchestrator for dry-bulk freight forecasting.
    Binds data loading, feature generation, model loading / fitting,
    uncertainty intervals, and canonical API serialization.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        default_model_type: str = "ensemble"
    ):
        settings = get_settings()
        root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = Path(models_dir) if models_dir else (root / "models")
        self.data_dir = Path(data_dir) if data_dir else (root / "data" / "processed" if (root / "data" / "processed").exists() else root / "data" / "demo")
        self.default_model_type = default_model_type.lower()
        self.feature_builder = FreightFeatureBuilder(data_dir=str(self.data_dir))

    def load_historical_data(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        cargo_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical freight rates matching the route and vessel class.
        """
        freight_path = self.data_dir / "freight_rates.csv"
        if not freight_path.exists():
            raise FileNotFoundError(f"Freight rates file not found at {freight_path}")

        df = pd.read_csv(freight_path)
        df["date"] = pd.to_datetime(df["date"])

        # Filter by route and vessel
        mask = (
            (df["origin"].str.upper() == origin.upper()) &
            (df["destination"].str.upper() == destination.upper()) &
            (df["vessel_class"].str.lower() == vessel_class.lower())
        )
        if cargo_type:
            mask = mask & (df["cargo_type"].str.lower() == cargo_type.lower())

        filtered = df[mask].sort_values("date").reset_index(drop=True)

        if filtered.empty:
            # Broader fallback: match route only or vessel only if specific combo missing
            mask_fallback = (
                (df["origin"].str.upper() == origin.upper()) &
                (df["destination"].str.upper() == destination.upper())
            )
            filtered = df[mask_fallback].sort_values("date").reset_index(drop=True)

        if filtered.empty:
            # Global fallback to avoid crash
            logger.warning(f"No rates found for {origin}->{destination} {vessel_class}. Using all historical records.")
            filtered = df.sort_values("date").reset_index(drop=True)

        return filtered

    def _get_model_key(self, origin: str, destination: str, vessel_class: str) -> str:
        return f"{origin}_{destination}_{vessel_class}".replace(" ", "_").lower()

    def get_trained_model(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        model_name: Optional[str] = None
    ) -> Tuple[Optional[ForecastModel], Optional[Dict[str, Any]]]:
        """
        Check if a serialized model and its metadata exist under models/.
        """
        key = self._get_model_key(origin, destination, vessel_class)
        route_dir = self.models_dir / key

        if not route_dir.exists():
            return None, None

        # Search for available model versions
        m_type = model_name or self.default_model_type
        target_dir = route_dir / m_type
        if not target_dir.exists():
            # Check subdirectories
            subdirs = [d for d in route_dir.iterdir() if d.is_dir()]
            if not subdirs:
                return None, None
            target_dir = subdirs[0]

        artifact_file = target_dir / "model.pkl"
        metadata_file = target_dir / "metadata.json"

        if not artifact_file.exists():
            return None, None

        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")

        # Instantiate model based on directory name or metadata
        m_name = metadata.get("model_name", target_dir.name).lower()
        if "xgboost" in m_name:
            model = XGBoostForecaster()
        elif "arima" in m_name:
            model = ARIMAForecaster()
        elif "ensemble" in m_name:
            model = EnsembleForecaster()
        else:
            model = BaselineForecaster()

        try:
            model.load(str(artifact_file))
            return model, metadata
        except Exception as e:
            logger.warning(f"Failed loading model artifact {artifact_file}: {e}")
            return None, None

    def predict_freight(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        cargo_type: str = "thermal_coal",
        horizon_days: int = 7,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate end-to-end freight forecast returning canonical schema:
        {
            "current_rate": float,
            "forecast_rate": float,
            "lower_bound": float,
            "upper_bound": float,
            "trend": str,
            "confidence": float,
            "model_used": str,
            "metrics": dict
        }
        """
        # Validate horizon
        valid_horizons = [3, 7, 14, 30]
        if horizon_days not in valid_horizons:
            # Pick closest horizon or use requested
            horizon_days = min(valid_horizons, key=lambda x: abs(x - horizon_days))

        route_str = f"{origin}->{destination}"
        selected_type = (model_type or self.default_model_type).lower()

        # 1. Load historical context data
        df_history = self.load_historical_data(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type
        )
        current_rate = float(df_history["freight_rate"].iloc[-1])

        # 2. Enrich features
        featured_df = self.feature_builder.build_features(
            freight_df=df_history,
            target_col="freight_rate",
            date_col="date",
            destination_port=destination
        )

        # 3. Check for pre-trained model in models/
        model, metadata = self.get_trained_model(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            model_name=selected_type
        )

        metrics: Dict[str, float] = {}
        if model is None:
            # On-the-fly fit on historical data
            logger.info(f"No pre-trained artifact found for {route_str} {vessel_class}. Training on-the-fly {selected_type} model.")

            if selected_type == "naive":
                model = NaiveBaselineForecaster()
            elif selected_type == "moving_average":
                model = MovingAverageForecaster(window_size=7)
            elif selected_type == "seasonal":
                model = SeasonalBaselineForecaster(seasonal_period=7)
            elif selected_type == "arima":
                model = ARIMAForecaster(order=(1, 1, 1))
            elif selected_type == "xgboost":
                model = XGBoostForecaster()
            else:
                model = EnsembleForecaster(weighting_strategy="inverse_rmse")

            model.fit(featured_df, target_col="freight_rate", date_col="date")

            # Evaluate on in-sample validation tail
            val_df = featured_df.tail(min(30, len(featured_df)))
            metrics = model.evaluate(val_df, target_col="freight_rate", date_col="date")
            # Clean up Model key from metrics
            metrics = {k: v for k, v in metrics.items() if k != "Model"}
        else:
            metrics = metadata.get("metrics", {})

        # 4. Generate forecast
        forecast_res = model.predict(
            horizon_days=horizon_days,
            context_df=featured_df,
            date_col="date",
            target_col="freight_rate",
            route=route_str,
            vessel_class=vessel_class
        )

        last_point = forecast_res.series[-1]
        forecast_rate = float(last_point.predicted_rate)
        lower_bound = float(last_point.lower_ci if last_point.lower_ci is not None else forecast_rate * 0.95)
        upper_bound = float(last_point.upper_ci if last_point.upper_ci is not None else forecast_rate * 1.05)

        # 5. Determine trend
        pct_change = ((forecast_rate - current_rate) / max(0.1, current_rate)) * 100.0
        if pct_change > 2.0:
            trend = "rising"
        elif pct_change < -2.0:
            trend = "falling"
        else:
            trend = "stable"

        # 6. Statistically defensible confidence score
        # Bounded between 0.60 and 0.98, derived from relative uncertainty width
        rel_uncertainty = (upper_bound - lower_bound) / (2.0 * max(1.0, forecast_rate))
        confidence = float(np.clip(1.0 - rel_uncertainty, 0.60, 0.98))

        return {
            "current_rate": round(current_rate, 2),
            "forecast_rate": round(forecast_rate, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "trend": trend,
            "confidence": round(confidence, 2),
            "model_used": forecast_res.model_used,
            "metrics": metrics
        }
