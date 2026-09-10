"""
Charter-AI — Master Decision Explainer (Phase 12).

Synthesizes the central explainability layer across all intelligence services,
providing comprehensive, mathematically consistent answers to the 7 core questions:
1. What was selected?
2. Why was it selected?
3. What alternatives were considered?
4. What constraints eliminated alternatives?
5. What economic factors influenced the result?
6. What risks influenced the result?
7. How confident is the recommendation?
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.explainability.forecast_explainer import ForecastExplainer, ForecastExplanation
from src.explainability.vessel_explainer import VesselExplainer, VesselPlanExplanation
from src.explainability.risk_explainer import RiskExplainer, RiskExplanation
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StructuredSevenAnswers:
    """Explicit, auditable answers to the 7 core explainability questions."""
    what_was_selected: Dict[str, Any]
    why_was_it_selected: List[str]
    alternatives_considered: List[Dict[str, Any]]
    constraints_eliminated: List[Dict[str, Any]]
    economic_factors: Dict[str, Any]
    risks_influencing_result: Dict[str, Any]
    confidence_assessment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "what_was_selected": self.what_was_selected,
            "why_was_it_selected": self.why_was_it_selected,
            "alternatives_considered": self.alternatives_considered,
            "constraints_eliminated": self.constraints_eliminated,
            "economic_factors": self.economic_factors,
            "risks_influencing_result": self.risks_influencing_result,
            "confidence_assessment": self.confidence_assessment,
        }


@dataclass
class UnifiedDecisionExplanation:
    """Root explainability payload conforming to Phase 10 & Phase 12 contracts."""
    summary: str
    recommendation_summary: str
    primary_reasons: List[str]
    tradeoff_analysis: str
    alternatives_rejected: List[Dict[str, Any]]
    structured_answers: StructuredSevenAnswers
    vessel_explanation: Dict[str, Any]
    forecast_explanation: Dict[str, Any]
    risk_explanation: Dict[str, Any]
    shap_analysis: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "recommendation_summary": self.recommendation_summary,
            "primary_reasons": self.primary_reasons,
            "tradeoff_analysis": self.tradeoff_analysis,
            "alternatives_rejected": self.alternatives_rejected,
            "structured_answers": self.structured_answers.to_dict(),
            "vessel_explanation": self.vessel_explanation,
            "forecast_explanation": self.forecast_explanation,
            "risk_explanation": self.risk_explanation,
            "shap_analysis": self.shap_analysis,
        }


class DecisionExplainer:
    """
    Master explainability synthesizer for CharterAI.
    Combines ForecastExplainer, VesselExplainer, and RiskExplainer into an auditable report.
    """

    def __init__(self):
        self.forecast_explainer = ForecastExplainer()
        self.vessel_explainer = VesselExplainer()
        self.risk_explainer = RiskExplainer()

    def generate_decision_explanation(
        self,
        # Vessel & fleet inputs
        recommended_plan: Any,
        all_candidate_plans: List[Any],
        port_compatibility_info: Optional[Dict[str, Any]] = None,
        origin_port_id: str = "AUS_NEW",
        destination_port_id: str = "IND_GVM",
        # Freight forecast inputs
        forecast_dict: Optional[Dict[str, Any]] = None,
        forecaster_model_obj: Optional[Any] = None,
        feature_context: Optional[Any] = None,
        # Risk & contract inputs
        risk_result: Optional[Any] = None,
        contract_rec: Optional[Any] = None,
        all_contract_evaluations: Optional[List[Any]] = None,
        risk_tolerance: str = "MEDIUM",
        # Market timing
        timing_action: str = "BOOK_NOW",
        composite_confidence: float = 0.88,
    ) -> UnifiedDecisionExplanation:
        """
        Build the canonical unified explainability report answering Questions 1-7.
        """
        forecast_dict = forecast_dict or {}

        # 1. Run Sub-Explainers
        vessel_exp: VesselPlanExplanation = self.vessel_explainer.explain_vessel_selection(
            recommended_plan=recommended_plan,
            all_candidate_plans=all_candidate_plans,
            port_compatibility_info=port_compatibility_info,
            origin_port_id=origin_port_id,
            destination_port_id=destination_port_id,
        )

        forecast_exp: ForecastExplanation = self.forecast_explainer.explain_forecast(
            forecast_dict=forecast_dict,
            model_obj=forecaster_model_obj,
            feature_context=feature_context,
            top_k=5,
        )

        risk_exp: RiskExplanation = self.risk_explainer.explain_risk_and_contract(
            risk_result=risk_result,
            contract_rec=contract_rec,
            all_contract_evaluations=all_contract_evaluations,
            risk_tolerance=risk_tolerance,
        )

        # 2. Answer Question 1: What was selected?
        selected_vessel_class = vessel_exp.selected_plan.vessel_class
        selected_plan_id = vessel_exp.selected_plan.plan_id
        contract_strategy_name = risk_exp.contract_strategy.recommended_strategy

        what_selected = {
            "vessel_plan": {
                "plan_id": selected_plan_id,
                "vessel_class": selected_vessel_class,
                "vessel_count": vessel_exp.selected_plan.number_of_vessels,
                "voyages": vessel_exp.selected_plan.number_of_voyages,
                "cargo_utilization_pct": round(vessel_exp.selected_plan.cargo_utilization * 100.0, 1),
                "total_delivered_cost": round(vessel_exp.selected_plan.total_cost, 2),
                "cost_per_tonne": round(vessel_exp.selected_plan.cost_per_tonne, 2),
            },
            "contract_strategy": {
                "strategy_name": contract_strategy_name,
                "spot_pct": risk_exp.contract_strategy.spot_percentage,
                "term_pct": risk_exp.contract_strategy.short_term_percentage + risk_exp.contract_strategy.medium_term_percentage,
                "expected_cost": round(risk_exp.contract_strategy.expected_cost, 2),
                "downside_cost_p90": round(risk_exp.contract_strategy.downside_cost, 2),
            },
            "market_timing": {
                "action": timing_action,
                "forecast_trend": forecast_exp.trend,
                "expected_freight_p50": round(forecast_exp.p50, 2),
            },
        }

        # 3. Answer Question 2: Why was it selected?
        why_selected = [
            vessel_exp.comparative_justification,
            (
                f"Achieved optimal balance of delivered economics (${vessel_exp.selected_plan.cost_per_tonne:.2f}/t) "
                f"and operational feasibility with {vessel_exp.selected_plan.cargo_utilization * 100:.1f}% deadweight utilization."
            ),
            (
                f"Predicted port wait days at {destination_port_id}: {vessel_exp.selected_plan.waiting_time:.1f} wait days "
                f"(expected waiting time: {vessel_exp.selected_plan.waiting_time:.1f} days, demurrage probability {vessel_exp.selected_plan.demurrage_probability * 100:.1f}%)."
            ),
            (
                f"Freight forecast and timing: {forecast_exp.trend.lower()} at ${forecast_exp.p50:.2f}/t "
                f"({forecast_exp.confidence * 100:.0f}% confidence), recommending {timing_action}."
            ),
            (
                f"Mitigates contract risk via '{contract_strategy_name}' (P90 tail risk bounded at "
                f"${risk_exp.contract_strategy.downside_cost:,.0f} with {risk_exp.contract_strategy.flexibility_score * 100:.0f}% flexibility)."
            ),
            (
                f"High delivery schedule confidence of {vessel_exp.selected_plan.delivery_probability * 100:.1f}% "
                f"ensures laycan compliance."
            ),
        ]

        # 4. Answer Question 3: What alternatives were considered?
        alts_considered = [alt.to_dict() for alt in vessel_exp.alternatives_considered]

        # 5. Answer Question 4: What constraints eliminated alternatives?
        constraints_eliminated = [ce.to_dict() for ce in vessel_exp.constraints_eliminated]

        # 6. Answer Question 5: What economic factors influenced the result?
        economic_factors = {
            "freight_rate_base": round(forecast_exp.historical_rate, 2),
            "forecast_rate_p50": round(forecast_exp.p50, 2),
            "total_delivered_cost": round(vessel_exp.selected_plan.total_cost, 2),
            "cost_per_tonne": round(vessel_exp.selected_plan.cost_per_tonne, 2),
            "expected_demurrage_exposure": round(
                vessel_exp.selected_plan.total_cost * vessel_exp.selected_plan.demurrage_probability * 0.15,
                2
            ),
            "key_drivers": [
                f"Economies of scale for {selected_vessel_class} parcels",
                f"Forecast freight rate trajectory ({forecast_exp.rate_delta_pct:+.1f}%)",
                f"Bunker fuel and port turnaround economics",
            ],
        }

        # 7. Answer Question 6: What risks influenced the result?
        dominant_drivers = [d.to_dict() for d in risk_exp.top_risk_drivers]
        risks_influencing = {
            "composite_risk_score": round(risk_exp.composite_score, 1),
            "composite_severity": risk_exp.composite_severity,
            "top_risk_categories": dominant_drivers,
            "demurrage_probability": round(vessel_exp.selected_plan.demurrage_probability, 4),
            "schedule_delivery_risk": round(1.0 - vessel_exp.selected_plan.delivery_probability, 4),
            "downside_tail_cost_p90": round(risk_exp.contract_strategy.downside_cost, 2),
        }

        # 8. Answer Question 7: How confident is the recommendation?
        confidence_assessment = {
            "composite_confidence": round(composite_confidence, 2),
            "forecast_model_confidence": round(forecast_exp.confidence, 2),
            "delivery_schedule_probability": round(vessel_exp.selected_plan.delivery_probability, 2),
            "data_fidelity_score": 0.95,
            "narrative": (
                f"Recommendation carries {composite_confidence * 100:.0f}% composite confidence, supported by "
                f"{forecast_exp.confidence * 100:.0f}% forecast model confidence and "
                f"{vessel_exp.selected_plan.delivery_probability * 100:.0f}% on-time arrival reliability."
            ),
        }

        structured_answers = StructuredSevenAnswers(
            what_was_selected=what_selected,
            why_was_it_selected=why_selected,
            alternatives_considered=alts_considered,
            constraints_eliminated=constraints_eliminated,
            economic_factors=economic_factors,
            risks_influencing_result=risks_influencing,
            confidence_assessment=confidence_assessment,
        )

        # 9. Build backward-compatible rejection reasons
        alternatives_rejected_legacy = []
        for alt in vessel_exp.alternatives_considered:
            if not alt.feasibility:
                reason_str = alt.failed_constraints[0] if alt.failed_constraints else "Physical port limits exceeded"
            else:
                diff = alt.cost_per_tonne - vessel_exp.selected_plan.cost_per_tonne
                reason_str = f"Higher delivered cost (+${diff:.2f}/t) and lower composite score ({alt.score:.1f}/100)"
            alternatives_rejected_legacy.append({
                "plan_id": alt.plan_id,
                "vessel_class": alt.vessel_class,
                "reasons_rejected": [reason_str],
            })

        overall_summary = (
            f"RECOMMEND {timing_action} {contract_strategy_name} with {selected_plan_id} ({selected_vessel_class}): "
            f"{vessel_exp.comparative_justification}"
        )

        return UnifiedDecisionExplanation(
            summary=overall_summary,
            recommendation_summary=overall_summary,
            primary_reasons=why_selected,
            tradeoff_analysis=vessel_exp.tradeoff_analysis,
            alternatives_rejected=alternatives_rejected_legacy,
            structured_answers=structured_answers,
            vessel_explanation=vessel_exp.to_dict(),
            forecast_explanation=forecast_exp.to_dict(),
            risk_explanation=risk_exp.to_dict(),
            shap_analysis=forecast_exp.top_drivers,
        )
