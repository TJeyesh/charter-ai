"""
Charter-AI — Voyage Economics Endpoint.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import CostBreakdownResponse, VoyageEconomicsResponse
from src.economics.voyage_calculator import VoyageCalculator

router = APIRouter(prefix="/economics", tags=["Economics"])

_calculator = VoyageCalculator()


class VoyageEconomicsRequest(BaseModel):
    """Request body for voyage economics calculation."""
    origin_port_id: str
    destination_port_id: str
    vessel_class: str
    cargo_tonnage: int = Field(..., gt=0)
    sailing_distance_nm: float = Field(..., gt=0)
    freight_rate_usd_per_day: float = Field(..., gt=0)
    vlsfo_price_usd_per_tonne: float = Field(default=550.0, gt=0)
    predicted_idle_days: float = Field(default=2.0, ge=0)


@router.post("", response_model=VoyageEconomicsResponse)
async def calculate_voyage_economics(request: VoyageEconomicsRequest):
    """
    Calculate full voyage economics with cost breakdown.

    Includes freight, bunker fuel, port charges, insurance,
    and expected demurrage.
    """
    result = _calculator.calculate(
        vessel_class=request.vessel_class,
        cargo_tonnage=request.cargo_tonnage,
        sailing_distance_nm=request.sailing_distance_nm,
        freight_rate_usd_per_day=request.freight_rate_usd_per_day,
        vlsfo_price_usd_per_tonne=request.vlsfo_price_usd_per_tonne,
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        predicted_idle_days=request.predicted_idle_days,
    )

    return VoyageEconomicsResponse(
        total_voyage_cost_usd=result.total_voyage_cost_usd,
        cost_per_tonne_usd=result.cost_per_tonne_usd,
        cargo_tonnage=result.cargo_tonnage,
        vessel_class=result.vessel_class,
        sailing_days=result.sailing_days,
        total_voyage_days=result.total_voyage_days,
        breakdown=CostBreakdownResponse(
            freight_cost_usd=result.breakdown.freight_cost_usd,
            bunker_cost_usd=result.breakdown.bunker_cost_usd,
            load_port_charges_usd=result.breakdown.load_port_charges_usd,
            discharge_port_charges_usd=result.breakdown.discharge_port_charges_usd,
            insurance_usd=result.breakdown.insurance_usd,
            expected_demurrage_usd=result.breakdown.expected_demurrage_usd,
            miscellaneous_usd=result.breakdown.miscellaneous_usd,
        ),
    )
