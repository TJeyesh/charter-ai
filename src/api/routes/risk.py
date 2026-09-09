"""
Charter-AI — Risk Assessment Endpoint.
"""

from datetime import date

from fastapi import APIRouter

from src.api.serializers import RiskAssessmentResponse, RiskDimensionResponse
from src.risk.aggregator import RiskAggregator
from src.risk.operational_risk import OperationalRiskAssessor
from src.risk.weather_risk import WeatherRiskAssessor

router = APIRouter(prefix="/risk", tags=["Risk"])

_weather_assessor = WeatherRiskAssessor()
_operational_assessor = OperationalRiskAssessor()
_aggregator = RiskAggregator()


@router.get("", response_model=RiskAssessmentResponse)
async def assess_risk(
    origin_port_id: str,
    destination_port_id: str,
    target_date: date,
    sailing_distance_nm: float = 5000.0,
    routing_note: str = "",
):
    """
    Get composite risk assessment for a route and date.

    Currently includes weather and operational risk.
    Market and port risk require historical data queries
    and will be added when the full pipeline is connected.
    """
    # Weather risk at destination
    weather = _weather_assessor.assess(
        port_id=destination_port_id,
        target_date=target_date,
    )

    # Operational risk based on origin
    operational = _operational_assessor.assess(
        origin_port_id=origin_port_id,
        sailing_distance_nm=sailing_distance_nm,
        routing_note=routing_note,
    )

    # Aggregate (market and port risk added when DB is connected)
    composite = _aggregator.aggregate(
        weather=weather,
        operational=operational,
    )

    breakdown = {}
    for dim_name, dim_summary in composite.breakdown.items():
        breakdown[dim_name] = RiskDimensionResponse(
            score=dim_summary.score,
            level=dim_summary.level,
            detail=dim_summary.detail,
        )

    return RiskAssessmentResponse(
        composite_score=composite.composite_score,
        level=composite.level.value,
        breakdown=breakdown,
        dominant_risk=composite.dominant_risk,
        recommendation=composite.recommendation,
    )
