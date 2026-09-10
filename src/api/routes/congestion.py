"""
Charter-AI — Port Congestion & Idle Time Prediction Endpoints.

Returns expected vessel waiting time in days, P10/P50/P90 quantile intervals,
delay probabilities, and congestion level for loading and discharge ports.
"""

from typing import Optional
from datetime import date
from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel

from src.services.congestion_service import CongestionService
from src.api.serializers import (
    CongestionPredictionRequest,
    CongestionPredictionResponse,
)

router = APIRouter(prefix="/congestion", tags=["Congestion"])
predict_router = APIRouter(prefix="/predict", tags=["Prediction"])


class CongestionResponse(BaseModel):
    expected_wait_days: float
    p10_wait_days: float
    p50_wait_days: float
    p90_wait_days: float
    delay_probability: float
    congestion_level: str  # "LOW", "MODERATE", "HIGH", "SEVERE"
    confidence: float
    data_source: Optional[str] = None
    model_used: Optional[str] = None


def _handle_congestion_prediction(
    service: CongestionService,
    port_id: str,
    target_date: Optional[str] = None,
    vessel_class: Optional[str] = "Panamax",
    cargo_type: Optional[str] = "thermal_coal",
    cargo_quantity: Optional[float] = 75000.0,
) -> CongestionPredictionResponse:
    result = service.predict_congestion(
        port_id=port_id,
        target_date=target_date,
        vessel_class=vessel_class,
        cargo_type=cargo_type,
        cargo_quantity=cargo_quantity,
    )
    return CongestionPredictionResponse(
        port_id=port_id,
        expected_wait_days=result.get("expected_wait_days", 2.0),
        p10_wait_days=result.get("p10_wait_days", 1.0),
        p50_wait_days=result.get("p50_wait_days", 2.0),
        p90_wait_days=result.get("p90_wait_days", 4.0),
        delay_probability=result.get("delay_probability", 0.15),
        congestion_level=result.get("congestion_level", "MODERATE"),
        confidence=result.get("confidence", 0.88),
        data_source=result.get("data_source", "SYNTHETIC_DEMO"),
        model_used=result.get("model_used", "QuantileXGBoost"),
    )


@predict_router.post("/congestion", response_model=CongestionPredictionResponse)
async def post_predict_congestion(
    payload: CongestionPredictionRequest,
    request: Request,
) -> CongestionPredictionResponse:
    """
    POST /api/v1/predict/congestion
    
    Predict vessel waiting time, quantiles (P10, P50, P90), and delay probability
    using machine learning quantile regression.
    """
    service: CongestionService = getattr(request.app.state, "congestion_service", None)
    if service is None:
        service = CongestionService()

    try:
        return _handle_congestion_prediction(
            service=service,
            port_id=payload.port_id,
            target_date=payload.target_date,
            vessel_class=payload.vessel_class,
            cargo_type=payload.cargo_type,
            cargo_quantity=payload.cargo_quantity,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Congestion prediction error: {str(e)}"
        )


@router.post("/predict", response_model=CongestionPredictionResponse)
async def post_congestion_predict(
    payload: CongestionPredictionRequest,
    request: Request,
) -> CongestionPredictionResponse:
    """
    POST /api/v1/congestion/predict
    
    Alias for POST /api/v1/predict/congestion.
    """
    return await post_predict_congestion(payload, request)


@router.get("/predict", response_model=CongestionResponse)
async def predict_congestion(
    request: Request,
    port_id: str = Query(..., description="UN/LOCODE or port code, e.g. IND_PAR, IND_GVM, AUS_NEW"),
    target_date: Optional[str] = Query(None, description="Expected arrival date (YYYY-MM-DD)"),
    vessel_class: Optional[str] = Query("Panamax", description="Vessel class, e.g. Capesize, Panamax"),
    cargo_type: Optional[str] = Query("thermal_coal", description="Cargo type, e.g. thermal_coal"),
    cargo_quantity: Optional[float] = Query(75000.0, description="Cargo quantity in metric tonnes"),
):
    """
    Predict vessel waiting time, uncertainty quantiles, and delay probability at a port.
    """
    service: CongestionService = getattr(request.app.state, "congestion_service", None)
    if service is None:
        service = CongestionService()

    try:
        result = service.predict_congestion(
            port_id=port_id,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Congestion prediction error: {str(e)}"
        )

