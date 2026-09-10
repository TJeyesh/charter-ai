"""
Charter-AI — Pydantic Response Serializers.

API response models used by FastAPI for automatic OpenAPI documentation
and response validation.
"""

import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field


# =============================================================================
# Production Operational Metadata & Versions
# =============================================================================

DEFAULT_API_VERSION: str = "v1.0"

DEFAULT_MODEL_VERSIONS: Dict[str, str] = {
    "freight_forecaster": "v2.1.0",
    "congestion_predictor": "v2.0.0",
    "market_timing": "v1.5.0",
    "fleet_optimizer": "v2.0.0",
    "contract_optimizer": "v2.0.0",
    "risk_engine": "v2.0.0",
    "decision_engine": "v2.2.0",
}

DEFAULT_DATA_VERSIONS: Dict[str, str] = {
    "ports_dataset": "2026.1",
    "vessels_dataset": "2026.1",
    "routes_dataset": "2026.1",
    "freight_indices": "2026.1",
    "bunker_indices": "2026.1",
    "data_source": "SYNTHETIC_DEMO",
}


def generate_request_id(prefix: str = "req") -> str:
    """Generate a unique request ID string."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def get_current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Common
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    model_version: Optional[str] = None
    sih_demo_mode: bool = False
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))


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
    structured_answers: Optional[Dict[str, Any]] = None
    vessel_explanation: Optional[Dict[str, Any]] = None
    forecast_explanation: Optional[Dict[str, Any]] = None
    risk_explanation: Optional[Dict[str, Any]] = None
    shap_analysis: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Forecast
# =============================================================================

class ForecastPointResponse(BaseModel):
    date: date
    predicted_rate: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None


class FreightForecastApiResponse(BaseModel):
    current_rate: float
    forecast_rate: float
    lower_bound: float
    upper_bound: float
    trend: str  # "rising", "falling", "stable"
    confidence: float
    model_used: str
    metrics: Dict[str, Any] = {}


class ForecastResponse(BaseModel):
    current_rate: float
    forecast_rate: float
    lower_bound: float
    upper_bound: float
    trend: str  # "rising", "falling", "stable"
    confidence: float
    model_used: str
    metrics: Dict[str, Any] = {}
    # Optional backwards compatibility fields
    origin_port_id: Optional[str] = None
    destination_port_id: Optional[str] = None
    vessel_class: Optional[str] = None
    horizon_days: Optional[int] = None
    model_version: Optional[str] = None
    series: Optional[List[ForecastPointResponse]] = None


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


class RiskSimulationRequest(BaseModel):
    cargo_quantity_t: float = Field(..., gt=0, description="Cargo quantity in metric tonnes")
    base_freight_rate: float = Field(default=20.0, description="Freight rate ($/MT)")
    freight_volatility_pct: float = Field(default=15.0, description="Market freight volatility (%)")
    freight_rate_p10: Optional[float] = None
    freight_rate_p90: Optional[float] = None
    base_bunker_price: float = Field(default=650.0, description="Bunker fuel price ($/MT)")
    bunker_volatility_pct: float = Field(default=12.0, description="Bunker price volatility (%)")
    sea_distance_nm: float = Field(default=4500.0, description="Sea sailing distance (nautical miles)")
    service_speed_knots: float = Field(default=12.5, description="Vessel service speed (knots)")
    fuel_consumption_t_day: float = Field(default=28.0, description="Daily bunker consumption (MT/day)")
    expected_wait_days: float = Field(default=2.0, description="Expected port waiting days")
    p90_wait_days: Optional[float] = None
    port_handling_rate_t_day: float = Field(default=15000.0, description="Loading/discharge rate (MT/day)")
    agreed_laytime_days: Optional[float] = None
    demurrage_rate_usd_day: float = Field(default=20000.0, description="Daily demurrage rate ($/day)")
    port_charges_usd: float = Field(default=45000.0, description="Port dues and charges ($)")
    canal_charges_usd: float = Field(default=0.0, description="Canal transit tolls ($)")
    delivery_deadline_days: Optional[float] = Field(default=25.0, description="Delivery deadline window (days)")
    vessel_availability_probability: float = Field(default=0.95, description="Probability vessel remains available")
    n_simulations: int = Field(default=10000, description="Number of Monte Carlo iterations")
    seed: int = Field(default=42, description="Simulation seed for exact reproducibility")
    cost_threshold_usd: Optional[float] = None


class RiskSimulationResponse(BaseModel):
    expected_cost: float
    p10_cost: float
    p50_cost: float
    p90_cost: float
    demurrage_probability: float
    late_delivery_probability: float
    risk_score: float
    probability_of_infeasibility: float = 0.0
    scenarios: Dict[str, Any]
    cost_distribution: List[Dict[str, Any]] = []
    risk_assessment: Optional[Dict[str, Any]] = None
    seed: int = 42
    n_simulations: int = 10000


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


class DeliveredCostResponse(BaseModel):
    freight_cost: float
    bunker_cost: float
    port_cost: float
    waiting_cost: float
    demurrage_exposure: float
    positioning_cost: float
    miscellaneous_cost: float
    total_cost: float
    cost_per_tonne: float
    voyage_days: float
    delivery_probability: float
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Phase 9: Contract Optimization
# =============================================================================

class ContractOptimizationApiRequest(BaseModel):
    cargo_quantity_t: float = Field(..., gt=0, description="Cargo quantity in metric tonnes")
    spot_freight_rate: float = Field(default=20.0, description="Spot freight rate ($/MT)")
    short_term_freight_rate: Optional[float] = None
    medium_term_freight_rate: Optional[float] = None
    freight_volatility_pct: float = Field(default=16.0, description="Market freight volatility (%)")
    base_bunker_price: float = Field(default=650.0, description="Bunker fuel price ($/MT)")
    sea_distance_nm: float = Field(default=4500.0, description="Sea sailing distance (nautical miles)")
    delivery_deadline_days: Optional[float] = Field(default=26.0, description="Delivery deadline window (days)")
    risk_tolerance: str = Field(default="MEDIUM", description="Risk tolerance: LOW, MEDIUM, or HIGH")
    vessel_availability: str = Field(default="TIGHT", description="Vessel availability: ABUNDANT, TIGHT, or SHORTAGE")
    number_of_voyages: int = Field(default=1, description="Number of required voyages")
    custom_strategies: Optional[List[Dict[str, Any]]] = None
    n_simulations: int = Field(default=5000, description="Number of Monte Carlo simulation runs")
    seed: int = Field(default=42, description="Simulation seed")


ContractOptimizationRequest = ContractOptimizationApiRequest


class ContractOptimizationApiResponse(BaseModel):
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    recommended_strategy: str
    spot_percentage: float
    short_term_percentage: float
    medium_term_percentage: float
    expected_cost: float
    p90_cost: float
    risk_score: float
    flexibility_score: float
    reasons: List[str]
    evaluated_strategies: List[Dict[str, Any]] = []


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


CargoRequest = RecommendationRequest


class RecommendationResponse(BaseModel):
    request_id: str = Field(default_factory=generate_request_id)
    generated_at: datetime
    timestamp: Optional[str] = None
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    recommendation: PrimaryRecommendation
    alternatives: List[PrimaryRecommendation] = []
    forecast: Optional[ForecastResponse] = None
    economics: Optional[VoyageEconomicsResponse] = None
    risk: Optional[RiskAssessmentResponse] = None
    vessel_compatibility: Optional[VesselSelectionResponse] = None


class AnalyzeVoyageRequest(BaseModel):
    cargo_type: str
    cargo_quantity: float = Field(..., gt=0, description="Cargo quantity in metric tonnes")
    origin: str
    destination: str
    required_delivery_date: date
    number_of_voyages: int = Field(default=1, gt=0, description="Number of voyages")
    contract_preference: Optional[str] = "ANY"


class AnalyzeVoyageResponse(BaseModel):
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
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


# =============================================================================
# Phase 10: Unified Decision Engine Schemas
# =============================================================================

class MarketAnalysisResponse(BaseModel):
    current_rate: float
    forecast: float
    direction: str
    confidence: float
    volatility: float


class RecommendedPlanResponse(BaseModel):
    vessel_class: str
    vessel_count: int
    voyages: int
    cargo_allocation: List[float] = Field(default_factory=list)
    port_compatibility: Dict[str, Any] = Field(default_factory=dict)
    utilization: float
    total_cost: float
    cost_per_tonne: float
    voyage_duration: float
    expected_waiting: float
    demurrage_probability: float
    delivery_probability: float
    risk_score: float


class DecisionExplanationResponse(BaseModel):
    summary: str
    primary_reasons: List[str]
    tradeoff_analysis: str
    alternatives_rejected: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation_summary: Optional[str] = None
    structured_answers: Optional[Dict[str, Any]] = None
    vessel_explanation: Optional[Dict[str, Any]] = None
    forecast_explanation: Optional[Dict[str, Any]] = None
    risk_explanation: Optional[Dict[str, Any]] = None
    shap_analysis: Optional[List[Dict[str, Any]]] = None


class DecisionResponse(BaseModel):
    decision_id: str
    request_id: Optional[str] = None
    timestamp: str
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    request_summary: Dict[str, Any] = Field(default_factory=dict)
    market_analysis: MarketAnalysisResponse
    freight_forecast: Dict[str, Any] = Field(default_factory=dict)
    market_timing: Dict[str, Any] = Field(default_factory=dict)
    recommended_plan: RecommendedPlanResponse
    alternative_plans: List[Dict[str, Any]] = Field(default_factory=list)
    economics: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    contract_strategy: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    explanation: DecisionExplanationResponse
    scenario_analysis: Optional[Dict[str, Any]] = None
    monte_carlo: Optional[Dict[str, Any]] = None
    multi_horizon_forecast: Optional[Dict[str, Any]] = None
    historical_rates: Optional[List[Dict[str, Any]]] = None
    forecast_trajectory: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Phase 14: Production API Schemas
# =============================================================================

class FreightForecastApiRequest(BaseModel):
    """Request payload for POST /api/v1/forecast/freight."""
    origin: str = Field(..., description="Origin port code, e.g. AUS_NEW")
    destination: str = Field(..., description="Destination port code, e.g. IND_GVM")
    vessel_class: str = Field(default="Panamax", description="Vessel class: Capesize, Panamax, Supramax, Handysize")
    horizon_days: int = Field(default=7, description="Forecast horizon in days (e.g. 3, 7, 14, 30)")
    cargo_type: str = Field(default="thermal_coal", description="Dry bulk commodity name")
    model_type: Optional[str] = Field(default=None, description="Model family override: ensemble, arima, xgboost, moving_average")


class FreightForecastResponse(BaseModel):
    """Response schema for POST /api/v1/forecast/freight."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    current_rate: float
    forecast_rate: float
    lower_bound: float
    upper_bound: float
    trend: str
    confidence: float
    model_used: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    origin: str
    destination: str
    vessel_class: str
    horizon_days: int


class CongestionPredictionRequest(BaseModel):
    """Request payload for POST /api/v1/predict/congestion."""
    port_id: str = Field(..., description="Port UN/LOCODE or ID, e.g. IND_PAR, IND_GVM, AUS_NEW")
    target_date: Optional[str] = Field(default=None, description="Expected arrival date (YYYY-MM-DD)")
    vessel_class: Optional[str] = Field(default="Panamax", description="Vessel class, e.g. Capesize, Panamax")
    cargo_type: Optional[str] = Field(default="thermal_coal", description="Cargo type, e.g. thermal_coal")
    cargo_quantity: Optional[float] = Field(default=75000.0, description="Cargo quantity in metric tonnes")


class CongestionPredictionResponse(BaseModel):
    """Response schema for POST /api/v1/predict/congestion."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    port_id: str
    expected_wait_days: float
    p10_wait_days: float
    p50_wait_days: float
    p90_wait_days: float
    delay_probability: float
    congestion_level: str
    confidence: float
    data_source: Optional[str] = None
    model_used: Optional[str] = None


class VesselOptimizationRequest(BaseModel):
    """Request payload for POST /api/v1/optimize/vessels."""
    origin_port_id: str = Field(..., description="Origin port code, e.g. AUS_NEW")
    destination_port_id: str = Field(..., description="Destination port code, e.g. IND_GVM")
    cargo_tonnage: Optional[float] = Field(default=80000.0, gt=0, description="Total cargo quantity in metric tonnes")
    cargo_type: Optional[str] = Field(default="thermal_coal", description="Commodity type")


class VesselOptimizationResponse(BaseModel):
    """Response schema for POST /api/v1/optimize/vessels."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    origin_port_id: str
    destination_port_id: str
    cargo_tonnage: Optional[float] = None
    cargo_type: Optional[str] = None
    feasible: List[VesselCompatibilityResponse] = Field(default_factory=list)
    excluded: List[VesselCompatibilityResponse] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class VoyageOptimizationRequest(BaseModel):
    """Request payload for POST /api/v1/optimize/voyage."""
    cargo_quantity_t: float = Field(..., gt=0, description="Total cargo volume in metric tonnes (MT)")
    origin_port_id: str = Field(default="AUS_NEW", description="Origin load port ID")
    destination_port_id: str = Field(default="IND_GVM", description="Destination discharge port ID")
    cargo_type: str = Field(default="Coal", description="Dry bulk commodity name")
    route_distance_nm: float = Field(default=4800.0, gt=0, description="Actual nautical distance")
    freight_rate_usd: float = Field(default=22.0, gt=0, description="Baseline freight rate USD/MT")
    delivery_deadline_days: Optional[float] = Field(default=None, gt=0, description="Target delivery deadline (days)")
    max_voyages: int = Field(default=5, ge=1, le=10, description="Maximum voyage count")
    allow_mixed_classes: bool = Field(default=True, description="Enable mixed vessel class combinations")
    cost_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    schedule_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    risk_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    utilization_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    demurrage_weight: float = Field(default=0.10, ge=0.0, le=1.0)


class VoyageOptimizationResponse(BaseModel):
    """Response schema for POST /api/v1/optimize/voyage."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    total_plans_evaluated: int = 0
    ranked_plans: List[Dict[str, Any]] = Field(default_factory=list)
    best_plan: Optional[Dict[str, Any]] = None


class PortsListResponse(BaseModel):
    """Response schema for GET /api/v1/ports."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    total_count: int
    ports: List[PortResponse]


class VesselsListResponse(BaseModel):
    """Response schema for GET /api/v1/vessels."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    total_count: int
    vessels: List[Dict[str, Any]] = Field(default_factory=list)
    vessel_classes: List[VesselClassResponse] = Field(default_factory=list)


class RoutesListResponse(BaseModel):
    """Response schema for GET /api/v1/routes."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    total_count: int
    routes: List[RouteResponse]


class MarketSummaryResponse(BaseModel):
    """Response schema for GET /api/v1/market."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    market_status: str
    benchmark_freight: Dict[str, float]
    bunker_prices: Dict[str, float]
    dry_bulk_indices: Dict[str, float]
    market_sentiment: str
    volatility_30d: float
    commentary: str


class ModelMetadataResponse(BaseModel):
    """Metadata item for an ML model in the registry."""
    model_name: str
    model_family: str
    version: str
    status: str
    accuracy_metric: str
    accuracy_value: float
    trained_date: str
    features: List[str]
    description: str


class ModelRegistryResponse(BaseModel):
    """Response schema for GET /api/v1/models."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    total_models: int
    models: List[ModelMetadataResponse]


class BacktestApiRequest(BaseModel):
    """Request payload for POST /api/v1/backtest."""
    years: List[int] = Field(default=[2023, 2024], description="Test years for walk-forward evaluation")
    horizons: List[int] = Field(default=[7, 14], description="Forecast horizons in days")
    n_scenarios: Optional[int] = Field(default=5, ge=1, le=50, description="Number of tender scenarios to replay")
    seed: int = Field(default=42, description="Random seed")


class BacktestApiResponse(BaseModel):
    """Response schema for POST /api/v1/backtest."""
    request_id: str = Field(default_factory=generate_request_id)
    timestamp: str = Field(default_factory=get_current_iso_timestamp)
    api_version: str = DEFAULT_API_VERSION
    model_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))
    data_versions: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DATA_VERSIONS))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    forecast_evaluation: Dict[str, Any] = Field(default_factory=dict)
    optimization_evaluation: Dict[str, Any] = Field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = Field(default_factory=dict)
    savings_summary: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized error envelope preventing raw traceback leaks."""
    error: str
    status_code: int
    request_id: str
    timestamp: str
    message: Optional[str] = None
    details: Optional[Any] = None

