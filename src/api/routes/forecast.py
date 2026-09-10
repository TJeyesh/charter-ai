"""
Charter-AI — Freight Forecast Endpoints.

Returns real statistical and machine-learning freight rate forecasts
for a given route, vessel class, cargo type, and horizon.

Returns:
{
    "current_rate": float,
    "forecast_rate": float,
    "lower_bound": float,
    "upper_bound": float,
    "trend": "rising" | "falling" | "stable",
    "confidence": float,
    "model_used": str,
    "metrics": dict
}
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, Query, HTTPException

from src.api.serializers import (
    ForecastResponse,
    FreightForecastApiResponse,
    FreightForecastApiRequest,
    FreightForecastResponse,
)
from src.services.freight_forecast_service import FreightForecastService
from src.models.market_timing import MarketTimingEngine, MarketTimingInputs

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.post("/freight", response_model=FreightForecastResponse)
async def post_freight_forecast(
    payload: FreightForecastApiRequest,
    request: Request,
) -> FreightForecastResponse:
    """
    Generate high-accuracy machine learning & statistical freight rate forecast
    for specified corridor, vessel class, and forecast horizon (e.g., 3, 7, 14, 30 days).
    """
    service: FreightForecastService = getattr(
        request.app.state, "forecast_service", None
    )
    if service is None:
        service = FreightForecastService()

    try:
        pred = service.predict_freight_api(
            origin=payload.origin,
            destination=payload.destination,
            vessel_class=payload.vessel_class,
            cargo_type=payload.cargo_type,
            horizon_days=payload.horizon_days,
            model_type=payload.model_type,
        )

        return FreightForecastResponse(
            current_rate=pred["current_rate"],
            forecast_rate=pred["forecast_rate"],
            lower_bound=pred["lower_bound"],
            upper_bound=pred["upper_bound"],
            trend=pred["trend"],
            confidence=pred["confidence"],
            model_used=pred["model_used"],
            metrics=pred.get("metrics", {}),
            origin=payload.origin,
            destination=payload.destination,
            vessel_class=payload.vessel_class,
            horizon_days=payload.horizon_days,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Freight forecasting engine error: {str(e)}"
        )



@router.get("", response_model=ForecastResponse)
@router.get("/predict", response_model=ForecastResponse)
async def get_forecast(
    request: Request,
    origin: str = Query(..., description="Origin port code, e.g. AUS_NEW"),
    destination: str = Query(..., description="Destination port code, e.g. IND_GVM"),
    vessel_class: str = Query(..., description="Vessel class, e.g. Capesize, Panamax"),
    horizon_days: int = Query(7, description="Forecast horizon: 3, 7, 14, 30 days"),
    cargo_type: str = Query("thermal_coal", description="Cargo type, e.g. thermal_coal"),
    model_type: Optional[str] = Query(None, description="Optional model family: naive, moving_average, seasonal, arima, xgboost, ensemble")
):
    """
    Generate dry-bulk freight rate forecast for specified corridor and horizon.
    """
    service: FreightForecastService = getattr(
        request.app.state, "forecast_service", None
    )
    if service is None:
        service = FreightForecastService()

    try:
        pred = service.predict_freight_api(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            horizon_days=horizon_days,
            model_type=model_type
        )

        # Also populate route metadata for backward compatibility
        pred["origin_port_id"] = origin
        pred["destination_port_id"] = destination
        pred["vessel_class"] = vessel_class
        pred["horizon_days"] = horizon_days
        pred["model_version"] = "v2.0"

        return pred
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Freight forecasting engine error: {str(e)}"
        )


@router.get("/intelligence")
async def get_market_intelligence(
    request: Request,
    origin: str = Query(..., description="Origin port code, e.g. AUS_NEW"),
    destination: str = Query(..., description="Destination port code, e.g. IND_GVM"),
    vessel_class: str = Query("Panamax", description="Vessel class, e.g. Capesize, Panamax"),
    cargo_type: str = Query("thermal_coal", description="Cargo type, e.g. thermal_coal"),
):
    """
    Produce multi-horizon forecast intelligence (3d, 7d, 14d, 30d) and historical/forecast
    chart trajectory with P10/P90 uncertainty intervals.
    """
    from datetime import datetime, timedelta

    service: FreightForecastService = getattr(
        request.app.state, "forecast_service", None
    )
    if service is None:
        service = FreightForecastService()

    try:
        # 1. Multi-horizon forecasts
        horizons = [3, 7, 14, 30]
        preds = {}
        for h in horizons:
            preds[f"forecast_{h}d"] = service.predict_freight_api(
                origin=origin,
                destination=destination,
                vessel_class=vessel_class,
                cargo_type=cargo_type,
                horizon_days=h,
            )

        current_rate = float(preds["forecast_7d"]["current_rate"])

        # 2. Historical data points
        df_hist = service.forecaster.load_historical_data(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
        )
        historical_records = []
        if not df_hist.empty:
            tail_df = df_hist.tail(30)
            for _, row in tail_df.iterrows():
                d_str = str(row["date"])[:10]
                historical_records.append({
                    "date": d_str,
                    "rate": round(float(row["freight_rate"]), 2),
                })

        # 3. Build continuous forecast trajectory with uncertainty band for chart
        forecast_trajectory = []
        today = datetime.now().date()
        forecast_trajectory.append({
            "date": today.isoformat(),
            "rate": current_rate,
            "p10": current_rate,
            "p90": current_rate,
            "is_forecast": False,
        })

        pred_30 = preds["forecast_30d"]
        target_rate = float(pred_30["forecast_rate"])
        lower_target = float(pred_30["lower_bound"])
        upper_target = float(pred_30["upper_bound"])

        for day in range(1, 31):
            p_date = today + timedelta(days=day)
            alpha = day / 30.0
            p_val = current_rate + alpha * (target_rate - current_rate)
            band_width_lower = alpha * (target_rate - lower_target)
            band_width_upper = alpha * (upper_target - target_rate)
            forecast_trajectory.append({
                "date": p_date.isoformat(),
                "rate": round(p_val, 2),
                "p10": round(max(0.0, p_val - band_width_lower), 2),
                "p90": round(p_val + band_width_upper, 2),
                "is_forecast": True,
            })

        return {
            "current_rate": current_rate,
            "forecast_3d": {
                "rate": round(float(preds["forecast_3d"]["forecast_rate"]), 2),
                "p10": round(float(preds["forecast_3d"]["lower_bound"]), 2),
                "p90": round(float(preds["forecast_3d"]["upper_bound"]), 2),
                "trend": preds["forecast_3d"]["trend"],
                "confidence": preds["forecast_3d"]["confidence"],
            },
            "forecast_7d": {
                "rate": round(float(preds["forecast_7d"]["forecast_rate"]), 2),
                "p10": round(float(preds["forecast_7d"]["lower_bound"]), 2),
                "p90": round(float(preds["forecast_7d"]["upper_bound"]), 2),
                "trend": preds["forecast_7d"]["trend"],
                "confidence": preds["forecast_7d"]["confidence"],
            },
            "forecast_14d": {
                "rate": round(float(preds["forecast_14d"]["forecast_rate"]), 2),
                "p10": round(float(preds["forecast_14d"]["lower_bound"]), 2),
                "p90": round(float(preds["forecast_14d"]["upper_bound"]), 2),
                "trend": preds["forecast_14d"]["trend"],
                "confidence": preds["forecast_14d"]["confidence"],
            },
            "forecast_30d": {
                "rate": round(float(preds["forecast_30d"]["forecast_rate"]), 2),
                "p10": round(float(preds["forecast_30d"]["lower_bound"]), 2),
                "p90": round(float(preds["forecast_30d"]["upper_bound"]), 2),
                "trend": preds["forecast_30d"]["trend"],
                "confidence": preds["forecast_30d"]["confidence"],
            },
            "historical_rates": historical_records,
            "forecast_trajectory": forecast_trajectory,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Freight intelligence engine error: {str(e)}"
        )


class MarketTimingApiRequest(BaseModel):
    """Request body for market timing recommendation."""
    current_freight_rate: float = Field(..., gt=0, description="Current spot freight rate USD/MT")
    forecast_rate: float = Field(..., gt=0, description="Forecasted freight rate USD/MT")
    p10_forecast: Optional[float] = Field(None, gt=0, description="10th percentile freight forecast")
    p50_forecast: Optional[float] = Field(None, gt=0, description="50th percentile (median) freight forecast")
    p90_forecast: Optional[float] = Field(None, gt=0, description="90th percentile freight forecast")
    forecast_confidence: float = Field(0.85, ge=0.0, le=1.0)
    market_momentum: float = Field(0.0, description="Market momentum indicator ($/t or %)")
    freight_volatility: float = Field(1.5, ge=0.0, description="Freight rate volatility ($/t)")
    vessel_availability: str = Field("BALANCED", description="TIGHT, BALANCED, or SURPLUS")
    congestion_forecast: float = Field(2.5, ge=0.0, description="Predicted port waiting days")
    cargo_deadline_days: Optional[float] = Field(None, gt=0, description="Cargo deadline in days from now")
    cargo_quantity_t: float = Field(75000.0, gt=0, description="Shipment cargo volume")
    route_voyage_days: float = Field(18.0, gt=0, description="Estimated voyage duration in days")


_timing_engine = MarketTimingEngine()


@router.post("/market-timing")
async def evaluate_market_timing(request: MarketTimingApiRequest) -> Dict[str, Any]:
    """
    Phase 7: Evaluate expected economic waiting benefits, deadline pressure, and tonnage availability
    to recommend whether to BOOK_NOW, WAIT, MONITOR, START_NEGOTIATION, or use HYBRID_BOOKING.
    """
    p50 = request.p50_forecast if request.p50_forecast is not None else request.forecast_rate
    p10 = request.p10_forecast if request.p10_forecast is not None else p50 * 0.92
    p90 = request.p90_forecast if request.p90_forecast is not None else p50 * 1.08

    inputs = MarketTimingInputs(
        current_freight_rate=request.current_freight_rate,
        forecast_rate=request.forecast_rate,
        p10_forecast=p10,
        p50_forecast=p50,
        p90_forecast=p90,
        forecast_confidence=request.forecast_confidence,
        market_momentum=request.market_momentum,
        freight_volatility=request.freight_volatility,
        vessel_availability=request.vessel_availability,
        congestion_forecast=request.congestion_forecast,
        cargo_deadline=request.cargo_deadline_days,
        cargo_quantity_t=request.cargo_quantity_t,
        route_voyage_days=request.route_voyage_days,
    )

    result = _timing_engine.evaluate_timing(inputs)
    return result.to_dict()
