"""
Maritime Risk Engine

Evaluates chartering risks across 6 primary dimensions.
Designed to integrate with external APIs in the future.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any

class RiskSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

def get_severity(score: float) -> RiskSeverity:
    if score < 40:
        return RiskSeverity.LOW
    elif score < 70:
        return RiskSeverity.MODERATE
    elif score < 90:
        return RiskSeverity.HIGH
    else:
        return RiskSeverity.CRITICAL

@dataclass
class RiskCategoryResult:
    category: str
    score: float # 0-100
    severity: RiskSeverity
    contributing_factors: List[str]
    warnings: List[str]
    
@dataclass
class ComprehensiveRiskAssessment:
    overall_score: float
    overall_severity: RiskSeverity
    categories: Dict[str, RiskCategoryResult]
    summary_messages: List[str]


class MaritimeRiskEngine:
    """
    Core engine evaluating transparent risk scores across multiple dimensions.
    """
    
    def evaluate_total_risk(self, inputs: Dict[str, Any]) -> ComprehensiveRiskAssessment:
        """
        Calculates all risk categories and aggregates them.
        
        inputs: Dictionary of raw data simulating API feeds for the various domains.
        """
        
        market_risk = self.calculate_market_risk(inputs.get("market_data", {}))
        port_risk = self.calculate_port_risk(inputs.get("port_data", {}))
        weather_risk = self.calculate_weather_risk(inputs.get("weather_data", {}))
        vessel_risk = self.calculate_vessel_availability_risk(inputs.get("vessel_data", {}))
        geo_risk = self.calculate_geopolitical_risk(inputs.get("geopolitical_data", {}))
        ops_risk = self.calculate_operational_risk(inputs.get("operational_data", {}))
        
        categories_dict = {
            "Market": market_risk,
            "Port": port_risk,
            "Weather": weather_risk,
            "Vessel Availability": vessel_risk,
            "Geopolitical": geo_risk,
            "Operational": ops_risk
        }
        
        # Weighted average for composite score
        # Example weights:
        weights = {
            "Market": 0.20,
            "Port": 0.20,
            "Weather": 0.15,
            "Vessel Availability": 0.15,
            "Geopolitical": 0.15,
            "Operational": 0.15
        }
        
        overall_score = sum(cat.score * weights[name] for name, cat in categories_dict.items())
        overall_score = max(0.0, min(100.0, overall_score))
        
        overall_severity = get_severity(overall_score)
        
        summary_messages = []
        for name, cat in categories_dict.items():
            if cat.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL):
                summary_messages.append(f"{name} Risk is {cat.severity.value}: {cat.warnings[0] if cat.warnings else 'Elevated risk detected.'}")
                
        if not summary_messages:
            summary_messages.append("All risk dimensions are within acceptable limits.")
            
        return ComprehensiveRiskAssessment(
            overall_score=round(overall_score, 1),
            overall_severity=overall_severity,
            categories=categories_dict,
            summary_messages=summary_messages
        )
        
    def calculate_market_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: price_volatility_pct, market_trend (Bull/Bear)
        volatility = data.get("price_volatility_pct", 10.0)
        score = min(100.0, volatility * 3.0)
        factors = [f"Market Volatility: {volatility}%"]
        warnings = ["High freight market volatility detected."] if score >= 70 else []
        return RiskCategoryResult("Market Risk", score, get_severity(score), factors, warnings)

    def calculate_port_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: expected_wait_days, berth_occupancy_pct
        wait_days = data.get("expected_wait_days", 2.0)
        score = min(100.0, wait_days * 15.0)
        factors = [f"Expected Wait: {wait_days} days"]
        warnings = [f"Severe port congestion ({wait_days} days wait)."] if score >= 70 else []
        return RiskCategoryResult("Port Risk", score, get_severity(score), factors, warnings)

    def calculate_weather_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: wave_height_m, storm_warning
        wave_height = data.get("wave_height_m", 1.5)
        storm = data.get("storm_warning", False)
        
        score = min(100.0, wave_height * 10.0)
        factors = [f"Wave Height: {wave_height}m"]
        warnings = []
        
        if storm:
            score = max(score, 85.0)
            factors.append("Storm Warning Active")
            warnings.append("Active storm warning on route.")
            
        return RiskCategoryResult("Weather Risk", score, get_severity(score), factors, warnings)

    def calculate_vessel_availability_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: available_vessels_in_region
        available = data.get("available_vessels_in_region", 10)
        score = max(0.0, 100.0 - (available * 5.0))
        factors = [f"Available Vessels: {available}"]
        warnings = ["Severe vessel shortage in region."] if score >= 70 else []
        return RiskCategoryResult("Vessel Availability", score, get_severity(score), factors, warnings)

    def calculate_geopolitical_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: route_conflict_level (0 to 10)
        conflict = data.get("route_conflict_level", 0.0)
        score = min(100.0, conflict * 10.0)
        factors = [f"Conflict Index: {conflict}/10"]
        warnings = ["Route transits high-risk geopolitical zones."] if score >= 70 else []
        return RiskCategoryResult("Geopolitical Risk", score, get_severity(score), factors, warnings)

    def calculate_operational_risk(self, data: Dict[str, Any]) -> RiskCategoryResult:
        # Example inputs: crew_changes_required, maintenance_due
        maint = data.get("maintenance_due", False)
        score = 80.0 if maint else 20.0
        factors = ["Maintenance Due"] if maint else ["Normal Operations"]
        warnings = ["Vessel requires maintenance soon."] if score >= 70 else []
        return RiskCategoryResult("Operational Risk", score, get_severity(score), factors, warnings)
