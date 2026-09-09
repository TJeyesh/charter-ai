"""
Charter-AI — Pydantic Response Serializers.

API response models used by FastAPI for automatic OpenAPI documentation
and response validation.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# Common
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    model_version: Optional[str] = None
    sih_demo_mode: bool = False


# =============================================================================
# Ports
# =============================================================================

class PortResponse(BaseModel):
    port_id: str
    port_name: str
    state: str
    country: str
    latitude: float
    longitude: float
    port_type: str
    operator: str
    berths_total: Optional[int] = None
    max_draft_m: Optional[float] = None
    max_loa_m: Optional[float] = None
    max_beam_m: Optional[float] = None
    max_dwt: Optional[int] = None
    annual_capacity_mtpa: Optional[str] = None
    primary_cargo: Optional[str] = None


class PortCongestionResponse(BaseModel):
    port_id: str
    date: date
    vessels_waiting: int
    avg_waiting_time_days: float
    berth_occupancy_pct: float


# =============================================================================
# Routes
# =============================================================================

class RouteResponse(BaseModel):
    origin_port_id: str
    origin_port_name: str
    origin_country: str
    destination_port_id: str
    destination_port_name: str
    great_circle_nm: float
    est_sailing_distance_nm: float
    typical_cargo: str
    routing_note: Optional[str] = None


# =============================================================================
# Vessels
# =============================================================================

class VesselClassResponse(BaseModel):
    class_name: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    draft_max_m: float
    loa_max_m: float
    beam_max_m: float


class VesselCompatibilityResponse(BaseModel):
    vessel_class: str
    is_compatible: bool
    violations: List[str] = []


class VesselSelectionResponse(BaseModel):
    origin_port_id: str
    destination_port_id: str
    feasible: List[VesselCompatibilityResponse]
    excluded: List[VesselCompatibilityResponse]


# =============================================================================
# Explainable AI (XAI) Schemas
# =============================================================================
class AlternativeExplanation(BaseModel):
    vessel_class: str
    reasons_rejected: List[str]

class ExplainabilityReport(BaseModel):
    recommendation_summary: str
    primary_reasons: List[str]
    alternatives_rejected: List[AlternativeExplanation]


# =============================================================================
# Forecast
# =============================================================================

class ForecastPointResponse(BaseModel):
    date: date
    predicted_rate: float
    lower_ci: float
    upper_ci: float


class ForecastResponse(BaseModel):
    origin_port_id: str
    destination_port_id: str
    vessel_class: str
    horizon_days: int
    model_version: str
    series: List[ForecastPointResponse]
    trend: str  # "rising", "falling", "stable"


# =============================================================================
# Risk
# =============================================================================

class RiskDimensionResponse(BaseModel):
    score: float
    level: str
    detail: str


class RiskAssessmentResponse(BaseModel):
    composite_score: float
    level: str
    breakdown: Dict[str, RiskDimensionResponse]
    dominant_risk: Optional[str] = None
    recommendation: str


# =============================================================================
# Economics
# =============================================================================

class CostBreakdownResponse(BaseModel):
    freight_cost_usd: float
    bunker_cost_usd: float
    load_port_charges_usd: float
    discharge_port_charges_usd: float
    insurance_usd: float
    expected_demurrage_usd: float
    miscellaneous_usd: float


class VoyageEconomicsResponse(BaseModel):
    total_voyage_cost_usd: float
    cost_per_tonne_usd: float
    cargo_tonnage: int
    vessel_class: str
    sailing_days: float
    total_voyage_days: float
    breakdown: CostBreakdownResponse


# =============================================================================
# Recommendation (Primary endpoint)
# =============================================================================

class ContractRecommendationResponse(BaseModel):
    contract_type: str
    duration_months: Optional[int] = None
    reasoning: str
    confidence: float


class PrimaryRecommendation(BaseModel):
    vessel_class: str
    optimal_booking_window: Dict[str, str]  # {"start": ..., "end": ...}
    contract: ContractRecommendationResponse
    estimated_rate_usd_per_day: float
    confidence: float
    reasoning: str


class ExplainabilityResponse(BaseModel):
    top_factors: List[Dict[str, float]]  # [{"feature": ..., "impact": ..., "direction": ...}]


class RecommendationRequest(BaseModel):
    """Request body for the /recommend endpoint."""
    origin_port_id: str
    destination_port_id: str
    cargo_type: str = "coal"
    cargo_tonnage: int = Field(..., gt=0, description="Cargo quantity in tonnes")
    earliest_date: date
    latest_date: date
    risk_appetite: str = Field(
        default="moderate",
        description="Risk tolerance: low, moderate, or high",
    )


class RecommendationResponse(BaseModel):
    request_id: str
    generated_at: datetime
    recommendation: PrimaryRecommendation
    alternatives: List[PrimaryRecommendation] = []
    forecast: Optional[ForecastResponse] = None
    economics: Optional[VoyageEconomicsResponse] = None
    risk: Optional[RiskAssessmentResponse] = None
    vessel_compatibility: Optional[VesselSelectionResponse] = None

class AnalyzeVoyageRequest(BaseModel):
    cargo_type: str
    cargo_quantity: float
    origin: str
    destination: str
    required_delivery_date: date
    number_of_voyages: int
    contract_preference: Optional[str] = "ANY"

class AnalyzeVoyageResponse(BaseModel):
    status: str
    error_message: Optional[str] = None
    market_forecast: Dict[str, Any] = Field(default_factory=dict)
    recommended_vessel: Dict[str, Any] = Field(default_factory=dict)
    port_analysis: Dict[str, Any] = Field(default_factory=dict)
    voyage_economics: Dict[str, Any] = Field(default_factory=dict)
    risk_analysis: Dict[str, Any] = Field(default_factory=dict)
    contract_strategy: Dict[str, Any] = Field(default_factory=dict)
    final_recommendation: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[ExplainabilityReport] = None
