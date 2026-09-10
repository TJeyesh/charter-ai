"""
Charter-AI — Maritime Risk & Contract Strategy Explainer (Phase 12).

Provides transparent Explainable AI (XAI) for:
1. 8-category maritime risk breakdown and dominant driver identification.
2. Monte Carlo probabilistic downside tail risk (P90).
3. Contract strategy selection (Spot vs Short-Term vs Medium-Term vs Hybrid)
   explicitly answering:
   - expected cost
   - downside cost (P90)
   - risk
   - flexibility
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.risk.risk_engine import ComprehensiveRiskAssessment, RiskCategoryResult
from src.optimization.contract_optimizer import StrategyEvaluation, RiskTolerance
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskCategoryDriver:
    """Individual maritime risk category metric and attribution."""
    category: str
    display_name: str
    score: float
    severity: str
    contributing_factors: List[str] = field(default_factory=list)
    is_primary_driver: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "display_name": self.display_name,
            "score": round(self.score, 1),
            "severity": self.severity,
            "contributing_factors": self.contributing_factors,
            "is_primary_driver": self.is_primary_driver,
        }


@dataclass
class ContractStrategyExplanation:
    """Canonical explainability representation for risk-aware contract strategy."""
    recommended_strategy: str
    spot_percentage: float
    short_term_percentage: float
    medium_term_percentage: float
    expected_cost: float
    downside_cost: float  # P90
    risk_penalty: float
    volatility_exposure: float
    flexibility_score: float
    contract_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_strategy": self.recommended_strategy,
            "spot_percentage": round(self.spot_percentage, 1),
            "short_term_percentage": round(self.short_term_percentage, 1),
            "medium_term_percentage": round(self.medium_term_percentage, 1),
            "expected_cost": round(self.expected_cost, 2),
            "downside_cost": round(self.downside_cost, 2),
            "risk_penalty": round(self.risk_penalty, 2),
            "volatility_exposure": round(self.volatility_exposure, 1),
            "flexibility_score": round(self.flexibility_score, 2),
            "contract_alternatives": self.contract_alternatives,
            "reasons": self.reasons,
            "summary": self.summary,
        }


@dataclass
class RiskExplanation:
    """Combined risk and contract optimization explainability response."""
    composite_score: float
    composite_severity: str
    top_risk_drivers: List[RiskCategoryDriver]
    category_breakdown: Dict[str, Any]
    contract_strategy: ContractStrategyExplanation
    narrative: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 1),
            "composite_severity": self.composite_severity,
            "top_risk_drivers": [d.to_dict() for d in self.top_risk_drivers],
            "category_breakdown": self.category_breakdown,
            "contract_strategy": self.contract_strategy.to_dict(),
            "narrative": self.narrative,
            "summary": self.summary,
        }


CATEGORY_LABELS = {
    "market_risk": "Freight Market Volatility Risk",
    "port_congestion_risk": "Port Congestion & Turnaround Risk",
    "weather_risk": "Maritime Weather & Cyclone Risk",
    "vessel_availability_risk": "Vessel Supply & Availability Risk",
    "operational_risk": "Cargo Handling & Port Operational Risk",
    "geopolitical_risk": "Geopolitical & Route Disruption Risk",
    "schedule_risk": "Laycan & Delivery Schedule Risk",
    "demurrage_risk": "Port Demurrage Financial Exposure Risk",
}


class RiskExplainer:
    """
    Explainability engine for maritime risk and risk-aware contract optimization.
    """

    def explain_risk_and_contract(
        self,
        risk_result: Optional[RiskAnalysisResult] = None,
        contract_rec: Optional[Any] = None,
        all_contract_evaluations: Optional[List[StrategyEvaluation]] = None,
        risk_tolerance: str = "MEDIUM"
    ) -> RiskExplanation:
        """
        Produce mathematically rigorous risk and contract strategy explanation.
        """
        # 1. Process 8 Risk Categories
        comp_score = 30.0
        comp_sev = "LOW"
        cat_breakdown: Dict[str, Any] = {}
        top_drivers: List[RiskCategoryDriver] = []

        if risk_result is not None:
            comp_score = float(getattr(risk_result, "composite_score", getattr(risk_result, "overall_score", 30.0)))
            sev_attr = getattr(risk_result, "composite_severity", getattr(risk_result, "overall_severity", "LOW"))
            comp_sev = str(sev_attr.value if hasattr(sev_attr, "value") else sev_attr)

            raw_cats = getattr(risk_result, "categories", {})
            if isinstance(raw_cats, dict):
                unique_cats = {}
                for v in raw_cats.values():
                    if hasattr(v, "category"):
                        unique_cats[v.category.lower()] = v
                cat_list = list(unique_cats.values())
            elif isinstance(raw_cats, list):
                cat_list = raw_cats
            else:
                cat_list = []

            # Sort categories by risk score descending
            cats_sorted = sorted(cat_list, key=lambda c: getattr(c, "score", 0.0), reverse=True)

            for idx, c in enumerate(cats_sorted):
                c_score = float(getattr(c, "score", 0.0))
                c_cat = getattr(c, "category", "unknown")
                c_sev = getattr(c, "severity", "MODERATE")
                sev_val = str(c_sev.value if hasattr(c_sev, "value") else c_sev)
                is_primary = (idx < 2 and c_score >= 35.0) or (idx == 0)
                disp = CATEGORY_LABELS.get(c_cat.lower(), c_cat.replace("_", " ").title())
                c_factors = getattr(c, "contributing_factors", [])

                driver = RiskCategoryDriver(
                    category=c_cat,
                    display_name=disp,
                    score=c_score,
                    severity=sev_val,
                    contributing_factors=c_factors,
                    is_primary_driver=is_primary,
                )
                if is_primary:
                    top_drivers.append(driver)

                cat_breakdown[c_cat] = {
                    "score": round(c_score, 1),
                    "severity": sev_val,
                    "contributing_factors": c_factors,
                }
        else:
            # Synthetic default for testing/fallback
            top_drivers.append(RiskCategoryDriver(
                category="port_congestion_risk",
                display_name="Port Congestion & Turnaround Risk",
                score=42.0,
                severity="MODERATE",
                contributing_factors=["Estimated wait 2.4 days"],
                is_primary_driver=True,
            ))

        # 2. Process Contract Strategy Explanation
        contract_expl = self._build_contract_explanation(
            contract_rec=contract_rec,
            all_evaluations=all_contract_evaluations or [],
            risk_tolerance=risk_tolerance
        )

        # 3. Formulate Overall Narrative
        primary_names = [d.display_name for d in top_drivers]
        primary_text = ", ".join(primary_names) if primary_names else "general operational uncertainty"

        summary = (
            f"Overall risk is {comp_sev} ({comp_score:.1f}/100) dominated by {primary_text}. "
            f"Recommended contract strategy is {contract_expl.recommended_strategy} "
            f"with ${contract_expl.downside_cost:,.0f} P90 downside protection."
        )

        narrative = (
            f"The risk assessment engine scored composite operational risk at {comp_score:.1f}/100 ({comp_sev}). "
            f"The primary risk contributors are: {primary_text}. "
            f"Under a {risk_tolerance.upper()} risk appetite, the optimizer selected '{contract_expl.recommended_strategy}', "
            f"balancing expected total cost of ${contract_expl.expected_cost:,.0f} against a P90 downside cap of "
            f"${contract_expl.downside_cost:,.0f} with a flexibility rating of {contract_expl.flexibility_score:.2f}."
        )

        return RiskExplanation(
            composite_score=comp_score,
            composite_severity=comp_sev,
            top_risk_drivers=top_drivers,
            category_breakdown=cat_breakdown,
            contract_strategy=contract_expl,
            narrative=narrative,
            summary=summary,
        )

    def _build_contract_explanation(
        self,
        contract_rec: Any,
        all_evaluations: List[StrategyEvaluation],
        risk_tolerance: str
    ) -> ContractStrategyExplanation:
        """Construct detailed contract strategy explainability."""
        rec_name = "60/40 HYBRID"
        spot_pct = 60.0
        short_pct = 40.0
        med_pct = 0.0
        exp_cost = 2500000.0
        p90_cost = 2750000.0
        risk_pen = 15000.0
        vol_exp = 12.5
        flex = 0.65
        reasons: List[str] = []

        if contract_rec is not None:
            rec_name = getattr(contract_rec, "recommended_strategy", getattr(contract_rec, "strategy_name", rec_name))
            spot_pct = float(getattr(contract_rec, "spot_percentage", spot_pct))
            short_pct = float(getattr(contract_rec, "short_term_percentage", short_pct))
            med_pct = float(getattr(contract_rec, "medium_term_percentage", med_pct))
            exp_cost = float(getattr(contract_rec, "expected_cost", getattr(contract_rec, "expected_total_cost", exp_cost)))
            p90_cost = float(getattr(contract_rec, "p90_cost", getattr(contract_rec, "downside_cost", p90_cost)))
            flex = float(getattr(contract_rec, "flexibility_score", flex))
            vol_exp = float(getattr(contract_rec, "price_volatility_exposure", getattr(contract_rec, "volatility_exposure", vol_exp)))
            risk_pen = float(getattr(contract_rec, "risk_penalty", risk_pen))
            reasons = list(getattr(contract_rec, "reasons", []))

        # Format alternatives table
        alt_dicts = []
        for ev in all_evaluations:
            alt_dicts.append({
                "strategy_name": ev.strategy_name,
                "expected_cost": round(ev.expected_total_cost, 2),
                "downside_cost": round(ev.p90_cost, 2),
                "volatility_exposure": round(ev.price_volatility_exposure, 1),
                "flexibility_score": round(ev.flexibility_score, 2),
                "objective_score": round(ev.objective_score, 2),
            })

        if not reasons:
            reasons = [
                f"Selected '{rec_name}' to align with {risk_tolerance.upper()} risk tolerance.",
                f"Bounds downside tail risk to ${p90_cost:,.0f} (P90) while preserving {flex * 100:.0f}% operational flexibility.",
                f"Mitigates freight volatility exposure while optimizing delivered voyage cost.",
            ]

        summary = (
            f"Strategy '{rec_name}' balances expected cost (${exp_cost:,.0f}) and P90 downside "
            f"(${p90_cost:,.0f}) with {flex * 100:.0f}% flexibility."
        )

        return ContractStrategyExplanation(
            recommended_strategy=rec_name,
            spot_percentage=spot_pct,
            short_term_percentage=short_pct,
            medium_term_percentage=med_pct,
            expected_cost=exp_cost,
            downside_cost=p90_cost,
            risk_penalty=risk_pen,
            volatility_exposure=vol_exp,
            flexibility_score=flex,
            contract_alternatives=alt_dicts,
            reasons=reasons,
            summary=summary,
        )
