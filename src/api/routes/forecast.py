"""
Charter-AI — Forecast Endpoint.

Returns freight rate forecasts for a given route, vessel class, and horizon.
"""

from fastapi import APIRouter, HTTPException

from src.api.serializers import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("", response_model=ForecastResponse)
async def get_forecast(
    origin: str,
    destination: str,
    vessel_class: str,
    horizon_days: int = 30,
):
    """
    Get freight rate forecast for a route and vessel class.

    NOTE: ML models are not yet trained. This endpoint will return
    an error until the forecasting model is implemented.
    """
    # TODO: Load trained model and generate forecast
    raise HTTPException(
        status_code=501,
        detail=(
            "Forecasting model not yet trained. "
            "This endpoint will be available after ML model implementation."
        ),
    )
