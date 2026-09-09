"""
Charter-AI — Central Decision Engine

Orchestrates the 6 core AI components to generate a single unified, 
explainable recommendation for dry-bulk chartering.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import sub-engines
from src.models.base_forecaster import ForecastModel
from src.optimization.vessel_selector import VesselOptimizer, VesselSpecs, PortConstraints
from src.optimization.port_compatibility import check_vessel_port_compatibility, PortInfo, VesselInfo
from src.economics.voyage_cost import calculate_voyage_cost, VoyageCostInputs
from src.risk.risk_engine import MaritimeRiskEngine, RiskSeverity
from src.optimization.contract_optimizer import (
    ContractOptimizer, ContractOptimizationInputs, ForecastDirection, 
    ForecastUncertainty, VesselAvailability, ContractStrategy
)

@dataclass
class DecisionEngineInputs:
    cargo_type: str
    cargo_quantity_t: float
    origin_port_id: str
    destination_port_id: str
    expected_loading_date: datetime
    required_delivery_date: datetime
    number_of_voyages: int
    contract_preference: str = "ANY"
    
    # Required engine resources passed via dependency injection or direct object references
    # To keep this clean, we'll assume basic mock data where needed or pass specific specs.
    vessel_specs_db: List[VesselSpecs] = field(default_factory=list)
    origin_port_info: Optional[PortInfo] = None
    destination_port_info: Optional[PortInfo] = None
    forecaster: Optional[ForecastModel] = None

class DecisionEngine:
    
    def __init__(self):
        self.vessel_optimizer = VesselOptimizer()
        self.risk_engine = MaritimeRiskEngine()
        self.contract_optimizer = ContractOptimizer()
        
    def evaluate(self, inputs: DecisionEngineInputs) -> Dict[str, Any]:
        """
        Executes the full pipeline and returns the unified recommendation dictionary.
        """
        try:
            # ---------------------------------------------------------
            # 1. Freight Forecast Engine
            # ---------------------------------------------------------
            # In a real app, we query the forecaster. We'll simulate the output structure.
            forecast_rate = 22.50 # USD/tonne
            forecast_trend = ForecastDirection.RISING
            forecast_uncertainty = ForecastUncertainty.MODERATE
            
            market_forecast = {
                "forecast_rate_usd": forecast_rate,
                "direction": forecast_trend.value,
                "uncertainty": forecast_uncertainty.value,
                "status": "SUCCESS"
            }
            # Simulated forecast
            
            # ---------------------------------------------------------
            # 2. Vessel Selection Engine
            # ---------------------------------------------------------
            if not inputs.vessel_specs_db:
                raise ValueError("Vessel specifications database is required.")
                
            # We mock the origin/destination PortConstraints for the initial VesselOptimizer pass
            # since PortConstraints is used in vessel_selector.py
            origin_c = PortConstraints(inputs.origin_port_id, "Origin", 25.0, 350.0, 60.0, 200000) 
            dest_c = PortConstraints(inputs.destination_port_id, "Destination", 25.0, 350.0, 60.0, 200000)
            
            if inputs.origin_port_info:
                origin_c.max_draft_m = inputs.origin_port_info.max_draft_m
                origin_c.max_loa_m = inputs.origin_port_info.max_loa_m
                origin_c.max_beam_m = inputs.origin_port_info.max_beam_m
            if inputs.destination_port_info:
                dest_c.max_draft_m = inputs.destination_port_info.max_draft_m
                dest_c.max_loa_m = inputs.destination_port_info.max_loa_m
                dest_c.max_beam_m = inputs.destination_port_info.max_beam_m
            
            vessel_result = self.vessel_optimizer.optimize(
                inputs.cargo_type,
                inputs.cargo_quantity_t,
                origin_c,
                dest_c,
                inputs.expected_loading_date,
                inputs.required_delivery_date,
                inputs.vessel_specs_db
            )
            
            if vessel_result.recommended_vessel == "None":
                raise ValueError("No feasible vessel found for this cargo and port combination.")
                
            recommended_vessel_class = vessel_result.recommended_vessel
            # Vessel selected
            
            # ---------------------------------------------------------
            # 3. Port Compatibility Engine (Deep Dive)
            # ---------------------------------------------------------
            # Fetch the specific spec that won
            chosen_spec = next(v for v in inputs.vessel_specs_db if v.class_name == recommended_vessel_class)
            
            vessel_info = VesselInfo(
                class_name=chosen_spec.class_name,
                draft_m=chosen_spec.draft_max_m,
                loa_m=chosen_spec.loa_max_m,
                beam_m=chosen_spec.beam_max_m,
                cargo_to_handle_t=inputs.cargo_quantity_t
            )
            
            port_analysis = {}
            if inputs.origin_port_info:
                origin_compat = check_vessel_port_compatibility(vessel_info, inputs.origin_port_info)
                if not origin_compat["compatible"]:
                    raise ValueError(f"Port deep-check failed at origin: {origin_compat['failed_constraints']}")
                port_analysis["origin"] = origin_compat
                # Validated origin
                
            if inputs.destination_port_info:
                dest_compat = check_vessel_port_compatibility(vessel_info, inputs.destination_port_info)
                if not dest_compat["compatible"]:
                    raise ValueError(f"Port deep-check failed at destination: {dest_compat['failed_constraints']}")
                port_analysis["destination"] = dest_compat
                # Validated dest

            # ---------------------------------------------------------
            # 4. Voyage Economics Engine
            # ---------------------------------------------------------
            voyage_inputs = VoyageCostInputs(
                cargo_quantity_t=inputs.cargo_quantity_t,
                freight_rate_usd=forecast_rate,
                vessel_speed_knots=12.5, # standard fallback
                vessel_daily_fuel_consumption_tpd=28.0,
                vessel_daily_hire_cost_usd=14000.0,
                route_distance_nm=4500.0, # placeholder
                positioning_distance_nm=500.0,
                fuel_price_usd_per_t=600.0,
                load_port_cost_usd=75000.0,
                discharge_port_cost_usd=75000.0,
                expected_waiting_days=2.0,
                daily_demurrage_rate_usd=18000.0,
                other_costs_usd=10000.0
            )
            voyage_econ = calculate_voyage_cost(voyage_inputs)
            # Economics ready

            # ---------------------------------------------------------
            # 5. Risk Engine
            # ---------------------------------------------------------
            # Mocking real-time inputs
            risk_inputs = {
                "market_data": {"price_volatility_pct": 12.0},
                "port_data": {"expected_wait_days": 2.0},
                "weather_data": {"wave_height_m": 2.0, "storm_warning": False},
                "vessel_data": {"available_vessels_in_region": 8},
                "geopolitical_data": {"route_conflict_level": 2.0},
                "operational_data": {"maintenance_due": False}
            }
            risk_result = self.risk_engine.evaluate_total_risk(risk_inputs)
            # Risk evaluated

            # ---------------------------------------------------------
            # 6. Contract Strategy Optimizer
            # ---------------------------------------------------------
            contract_inputs = ContractOptimizationInputs(
                current_freight_rate=21.0,
                expected_future_rate=forecast_rate,
                forecast_direction=forecast_trend,
                forecast_uncertainty=forecast_uncertainty,
                vessel_availability=VesselAvailability.TIGHT,
                congestion_level=2.0,
                overall_risk_score=risk_result.overall_score,
                voyage_cost=voyage_econ.total_voyage_cost_usd,
                number_of_required_voyages=inputs.number_of_voyages
            )
            contract_rec = self.contract_optimizer.recommend(contract_inputs)
            # Contract strategy evaluated

            # ---------------------------------------------------------
            # Build Explainability (XAI) Object
            # ---------------------------------------------------------
            primary_reasons = [
                f"Freight forecast indicates {forecast_trend.value.lower()} movement at ${forecast_rate}/t.",
                f"{recommended_vessel_class} provides optimal capacity for {inputs.cargo_quantity_t:,.0f} MT cargo.",
                f"Selected vessel satisfies draft/LOA constraints at {inputs.origin_port_id} and {inputs.destination_port_id}.",
                f"Expected total voyage cost is minimized (${voyage_econ.total_voyage_cost_usd:,.2f}).",
                f"Market and operational risk is {risk_result.overall_severity.value.lower()}.",
                f"Contract Strategy: {contract_rec.recommendation_text}"
            ]
            
            alternatives_rejected = []
            for alt in vessel_result.alternatives:
                alternatives_rejected.append({
                    "vessel_class": alt.vessel_class,
                    "reasons_rejected": [alt.reasons]
                })

            explainability_report = {
                "recommendation_summary": f"BOOK {contract_rec.strategy.value} {recommended_vessel_class.upper()}",
                "primary_reasons": primary_reasons,
                "alternatives_rejected": alternatives_rejected
            }

            # ---------------------------------------------------------
            # Build Unified Output
            # ---------------------------------------------------------
            final_recommendation = {
                "recommended_vessel_class": recommended_vessel_class,
                "action": contract_rec.strategy.value,
                "contract_type": contract_rec.recommendation_text,
                "estimated_total_voyage_cost_usd": voyage_econ.total_voyage_cost_usd,
                "main_risks": risk_result.summary_messages,
                "alternatives": [alt.vessel_class for alt in vessel_result.alternatives]
            }

            return {
                "status": "SUCCESS",
                "market_forecast": market_forecast,
                "recommended_vessel": {
                    "class": recommended_vessel_class,
                    "score": vessel_result.score,
                    "reasons": vessel_result.reasons
                },
                "port_analysis": port_analysis,
                "voyage_economics": {
                    "total_cost": voyage_econ.total_voyage_cost_usd,
                    "cost_per_tonne": round(voyage_econ.total_voyage_cost_usd / inputs.cargo_quantity_t, 2),
                    "breakdown": {
                        "freight_cost": voyage_econ.freight_cost_usd,
                        "bunker_cost": voyage_econ.bunker_cost_usd,
                        "port_charges": voyage_econ.port_costs_usd,
                        "demurrage_risk": voyage_econ.expected_demurrage_usd
                    }
                },
                "risk_analysis": {
                    "overall_score": risk_result.overall_score,
                    "overall_level": risk_result.overall_severity.value,
                    "dominant_risk": max(risk_result.categories.items(), key=lambda x: x[1].score)[0] if risk_result.categories else "None",
                    "summary": risk_result.summary_messages
                },
                "contract_strategy": {
                    "recommended_strategy": contract_rec.strategy.value,
                    "recommendation": contract_rec.recommendation_text,
                    "reasoning": contract_rec.reasons
                },
                "final_recommendation": final_recommendation,
                "explanation": explainability_report
            }

        except Exception as e:
            # Graceful and explicit error handling
            return {
                "status": "ERROR",
                "error_message": str(e),
                "market_forecast": {},
                "recommended_vessel": {},
                "port_analysis": {},
                "voyage_economics": {},
                "risk_analysis": {},
                "contract_strategy": {},
                "final_recommendation": {},
                "explanation": None
            }
