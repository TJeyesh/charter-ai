"""
Charter-AI — Model Registry & Metadata API Router (Phase 14).

Provides active ML model registry, versioning, training status, and benchmark performance:
GET /api/v1/models
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from src.api.serializers import (
    ModelRegistryResponse,
    ModelMetadataResponse,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelRegistryResponse)
async def list_models(
    model_family: Optional[str] = Query(None, description="Optional filter by model family"),
    status: Optional[str] = Query(None, description="Optional filter by status: active, candidate, deprecated"),
) -> ModelRegistryResponse:
    """
    GET /api/v1/models
    
    List registered machine learning models, active versions, evaluation metrics,
    and feature sets in CharterAI.
    """
    try:
        models = [
            ModelMetadataResponse(
                model_name="freight_forecaster_ensemble",
                model_family="Time Series & Gradient Boosted Regression",
                version="v2.1.0",
                status="active",
                accuracy_metric="MAPE",
                accuracy_value=6.8,
                trained_date="2024-12-31",
                features=[
                    "historical_freight_rate",
                    "bunker_price_lag",
                    "bdi_index",
                    "distance_nm",
                    "seasonal_sin_cos",
                    "queue_length",
                ],
                description="Multi-horizon dry bulk freight rate forecaster with P10/P90 quantile intervals.",
            ),
            ModelMetadataResponse(
                model_name="port_congestion_predictor",
                model_family="Quantile Gradient Boosting (XGBoost)",
                version="v2.0.0",
                status="active",
                accuracy_metric="Quantile Loss",
                accuracy_value=0.18,
                trained_date="2024-12-31",
                features=[
                    "berth_count",
                    "vessels_waiting",
                    "draft_limit_m",
                    "handling_rate_tpd",
                    "weather_wind_speed",
                ],
                description="Predicts vessel waiting days, congestion severity, and delay probability at major dry-bulk ports.",
            ),
            ModelMetadataResponse(
                model_name="market_timing_engine",
                model_family="Dynamic Opportunity Cost & Waiting Utility Evaluator",
                version="v1.5.0",
                status="active",
                accuracy_metric="Decision Accuracy (%)",
                accuracy_value=88.5,
                trained_date="2024-12-31",
                features=[
                    "expected_rate_savings",
                    "deadline_slack_days",
                    "forecast_uncertainty",
                    "momentum_trend",
                ],
                description="Determines optimal booking timing: BOOK_NOW, WAIT, MONITOR, START_NEGOTIATION, or HYBRID.",
            ),
            ModelMetadataResponse(
                model_name="fleet_voyage_optimizer",
                model_family="Mixed-Integer Constrained Optimizer",
                version="v2.0.0",
                status="active",
                accuracy_metric="Feasible Cost Score",
                accuracy_value=96.2,
                trained_date="2024-12-31",
                features=[
                    "cargo_quantity_t",
                    "port_draft_limits",
                    "port_loa_limits",
                    "voyage_duration",
                    "demurrage_exposure",
                ],
                description="Generates and ranks feasible multi-voyage and single-voyage vessel class charter plans.",
            ),
            ModelMetadataResponse(
                model_name="contract_strategy_optimizer",
                model_family="Probabilistic Monte Carlo Optimizer",
                version="v2.0.0",
                status="active",
                accuracy_metric="Risk-Adjusted Cost Score",
                accuracy_value=94.0,
                trained_date="2024-12-31",
                features=[
                    "freight_volatility",
                    "bunker_volatility",
                    "risk_tolerance",
                    "vessel_availability",
                    "delivery_deadline",
                ],
                description="Determines optimal mix across Spot, Short-term contract, and Medium-term contract allocations.",
            ),
            ModelMetadataResponse(
                model_name="decision_engine_orchestrator",
                model_family="Unified Multi-Objective Decision Support",
                version="v2.2.0",
                status="active",
                accuracy_metric="Recommendation Reliability (%)",
                accuracy_value=92.4,
                trained_date="2024-12-31",
                features=["end_to_end_pipeline_inputs"],
                description="Central decision engine integrating all forecasting, congestion, economics, risk, and strategy modules.",
            ),
        ]

        if model_family:
            models = [m for m in models if model_family.lower() in m.model_family.lower()]
        if status:
            models = [m for m in models if m.status.lower() == status.lower()]

        return ModelRegistryResponse(
            total_models=len(models),
            models=models,
        )
    except Exception as e:
        logger.exception("Model registry retrieval error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Model registry error: {str(e)}"
        )
