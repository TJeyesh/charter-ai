"""
Charter-AI — Contract Strategy Optimization Engine.

Evaluates market forecasts, physical constraints, and risk to 
recommend the optimal chartering contract strategy (Spot vs Term).
"""

from enum import Enum
from dataclasses import dataclass
from typing import List

class ContractStrategy(str, Enum):
    BOOK_NOW = "BOOK_NOW"
    WAIT = "WAIT"
    SPOT = "SPOT"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    HYBRID = "HYBRID"

class ForecastDirection(str, Enum):
    RISING = "RISING"
    FALLING = "FALLING"
    UNCERTAIN = "UNCERTAIN"
    STABLE = "STABLE"

class ForecastUncertainty(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class VesselAvailability(str, Enum):
    ABUNDANT = "ABUNDANT"
    TIGHT = "TIGHT"
    SHORTAGE = "SHORTAGE"

@dataclass
class ContractOptimizationInputs:
    current_freight_rate: float
    expected_future_rate: float
    forecast_direction: ForecastDirection
    forecast_uncertainty: ForecastUncertainty
    vessel_availability: VesselAvailability
    congestion_level: float # 0.0 to 10.0
    overall_risk_score: float # 0.0 to 100.0
    voyage_cost: float
    number_of_required_voyages: int

@dataclass
class ContractStrategyRecommendation:
    strategy: ContractStrategy
    recommendation_text: str
    reasons: List[str]


class ContractOptimizer:
    """
    Decision engine for selecting contract strategies based on transparent rules.
    """
    
    def recommend(self, inputs: ContractOptimizationInputs) -> ContractStrategyRecommendation:
        reasons = []
        
        # 1. Single Voyage Override
        if inputs.number_of_required_voyages == 1:
            reasons.append("Only a single voyage is required.")
            if inputs.forecast_direction == ForecastDirection.FALLING and inputs.vessel_availability == VesselAvailability.ABUNDANT:
                reasons.append("Forecast indicates falling rates and vessels are abundant. Suggest waiting slightly.")
                return ContractStrategyRecommendation(ContractStrategy.WAIT, "Delay booking briefly to capture falling spot rates.", reasons)
            else:
                reasons.append("Immediate booking recommended for single execution.")
                return ContractStrategyRecommendation(ContractStrategy.SPOT, "Book spot immediately for the single voyage.", reasons)

        # Multi-voyage Logic
        reasons.append(f"Multiple voyages required ({inputs.number_of_required_voyages}).")

        # 2. Vessel Shortage Override
        if inputs.vessel_availability == VesselAvailability.SHORTAGE:
            reasons.append("Critical vessel shortage overrides market price optimization.")
            return ContractStrategyRecommendation(
                ContractStrategy.MEDIUM_TERM,
                "Lock in a medium-term contract immediately to guarantee vessel availability.",
                reasons
            )
            
        # 3. High Risk / High Uncertainty -> Hybrid Diversification
        if inputs.forecast_uncertainty == ForecastUncertainty.HIGH or inputs.overall_risk_score >= 70.0:
            reasons.append("High forecast uncertainty or overall risk mandates risk diversification.")
            
            if inputs.forecast_direction == ForecastDirection.RISING:
                short_term_pct = 60
                spot_pct = 40
                reasons.append(f"Rates are expected to rise. {short_term_pct}% protects baseline, {spot_pct}% spot offers flexibility.")
            elif inputs.forecast_direction == ForecastDirection.FALLING:
                short_term_pct = 30
                spot_pct = 70
                reasons.append(f"Rates expected to fall. {spot_pct}% spot captures downside, {short_term_pct}% term secures core operations.")
            else:
                short_term_pct = 50
                spot_pct = 50
                reasons.append("Uncertain/stable market. Balanced 50/50 split optimizes risk.")
                
            recommendation_text = f"{short_term_pct}% short-term contract, {spot_pct}% spot exposure."
            return ContractStrategyRecommendation(ContractStrategy.HYBRID, recommendation_text, reasons)

        # 4. Clear Rising Trend
        if inputs.forecast_direction == ForecastDirection.RISING:
            reasons.append("Forecast indicates a clear upward trend in freight rates.")
            if inputs.forecast_uncertainty == ForecastUncertainty.LOW:
                reasons.append("High confidence in forecast allows for longer-term lock-in.")
                return ContractStrategyRecommendation(
                    ContractStrategy.MEDIUM_TERM,
                    "Secure a medium-term contract now before rates rise further.",
                    reasons
                )
            else:
                reasons.append("Moderate confidence suggests locking in near-term only.")
                return ContractStrategyRecommendation(
                    ContractStrategy.SHORT_TERM,
                    "Secure a short-term contract to hedge against immediate rate increases.",
                    reasons
                )

        # 5. Clear Falling Trend
        if inputs.forecast_direction == ForecastDirection.FALLING:
            reasons.append("Forecast indicates falling freight rates.")
            if inputs.vessel_availability == VesselAvailability.TIGHT:
                reasons.append("Vessels are somewhat tight. Secure baseline capacity.")
                return ContractStrategyRecommendation(
                    ContractStrategy.HYBRID,
                    "50% short-term to secure tight capacity, 50% spot to capture falling rates.",
                    reasons
                )
            else:
                reasons.append("Vessels are abundant. Avoid locking in long-term contracts.")
                return ContractStrategyRecommendation(
                    ContractStrategy.SPOT,
                    "Rely on spot market to capitalize on declining freight rates.",
                    reasons
                )

        # 6. Stable/Uncertain without extreme risk
        reasons.append("Market is relatively stable or flat.")
        return ContractStrategyRecommendation(
            ContractStrategy.SHORT_TERM,
            "Maintain baseline coverage with short-term contracts while monitoring the market.",
            reasons
        )
