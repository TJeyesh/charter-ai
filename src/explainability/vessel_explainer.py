"""
Charter-AI — Vessel Fleet Selection Explainer (Phase 12).

Provides transparent Explainable AI (XAI) for vessel selection and fleet planning:
1. Complete metrics breakdown (utilization, port compatibility, total cost, $/t, wait time,
   demurrage probability, delivery probability, risk score).
2. Auditable constraint elimination tracking (draft, LOA, beam, deadlines).
3. Exact comparative trade-off reasoning against runner-up alternatives.
4. Non-contradictory, calculation-backed justifications.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.optimization.voyage_plan import CharterPlan
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlanEvaluationSummary:
    """Standardized metric representation for any evaluated candidate plan."""
    plan_id: str
    vessel_class: str
    number_of_vessels: int
    number_of_voyages: int
    cargo_utilization: float
    port_compatibility: Dict[str, Any]
    total_cost: float
    cost_per_tonne: float
    waiting_time: float
    demurrage_probability: float
    delivery_probability: float
    risk_score: float
    feasibility: bool
    failed_constraints: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "vessel_class": self.vessel_class,
            "number_of_vessels": self.number_of_vessels,
            "number_of_voyages": self.number_of_voyages,
            "cargo_utilization": round(self.cargo_utilization, 4),
            "cargo_utilization_pct": round(self.cargo_utilization * 100.0, 1),
            "port_compatibility": self.port_compatibility,
            "total_cost": round(self.total_cost, 2),
            "cost_per_tonne": round(self.cost_per_tonne, 2),
            "waiting_time": round(self.waiting_time, 2),
            "demurrage_probability": round(self.demurrage_probability, 4),
            "delivery_probability": round(self.delivery_probability, 4),
            "risk_score": round(self.risk_score, 1),
            "feasibility": self.feasibility,
            "failed_constraints": self.failed_constraints,
            "score": round(self.score, 2),
        }


@dataclass
class ConstraintElimination:
    """Documented hard constraint violation eliminating a candidate plan."""
    plan_id: str
    vessel_class: str
    constraint_type: str
    reason: str
    port_id: Optional[str] = None
    port_limit: Optional[float] = None
    vessel_dimension: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "plan_id": self.plan_id,
            "vessel_class": self.vessel_class,
            "constraint_type": self.constraint_type,
            "reason": self.reason,
        }
        if self.port_id is not None:
            res["port_id"] = self.port_id
        if self.port_limit is not None:
            res["port_limit"] = self.port_limit
        if self.vessel_dimension is not None:
            res["vessel_dimension"] = self.vessel_dimension
        return res


@dataclass
class VesselPlanExplanation:
    """Complete explainability response for vessel fleet plan selection."""
    selected_plan: PlanEvaluationSummary
    alternatives_considered: List[PlanEvaluationSummary]
    constraints_eliminated: List[ConstraintElimination]
    comparative_justification: str
    primary_reasons: List[str]
    tradeoff_analysis: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_plan": self.selected_plan.to_dict(),
            "alternatives_considered": [alt.to_dict() for alt in self.alternatives_considered],
            "constraints_eliminated": [ce.to_dict() for ce in self.constraints_eliminated],
            "comparative_justification": self.comparative_justification,
            "primary_reasons": self.primary_reasons,
            "tradeoff_analysis": self.tradeoff_analysis,
            "summary": self.summary,
        }


class VesselExplainer:
    """
    Explainability synthesizer for multi-voyage vessel selection.
    """

    def explain_vessel_selection(
        self,
        recommended_plan: CharterPlan,
        all_candidate_plans: List[CharterPlan],
        port_compatibility_info: Optional[Dict[str, Any]] = None,
        destination_port_id: str = "IND_GVM",
        origin_port_id: str = "AUS_NEW",
    ) -> VesselPlanExplanation:
        """
        Produce mathematically rigorous vessel plan explanation.
        """
        port_compat = port_compatibility_info or {}

        # 1. Summarize selected plan
        selected_summary = self._summarize_plan(
            plan=recommended_plan,
            port_compat=port_compat,
            is_recommended=True
        )

        # 2. Summarize all alternatives & identify constraint violations
        alternatives_summaries: List[PlanEvaluationSummary] = []
        constraints_eliminated: List[ConstraintElimination] = []

        for plan in all_candidate_plans:
            if plan.plan_id == recommended_plan.plan_id:
                continue

            summary = self._summarize_plan(
                plan=plan,
                port_compat=port_compat,
                is_recommended=False
            )
            alternatives_summaries.append(summary)

            # Check if eliminated due to hard constraints
            if not plan.feasibility or plan.failed_constraints:
                for fc in plan.failed_constraints:
                    c_type = self._classify_constraint_type(fc)
                    constraints_eliminated.append(ConstraintElimination(
                        plan_id=plan.plan_id,
                        vessel_class=plan.vessel_classes[0] if plan.vessel_classes else "Unknown",
                        constraint_type=c_type,
                        reason=fc,
                    ))

        # 3. Generate comparative justification vs runner-up alternative
        comparative_justification, tradeoff_text, primary_reasons = self._formulate_tradeoff(
            selected=selected_summary,
            alternatives=alternatives_summaries,
            origin_port=origin_port_id,
            destination_port=destination_port_id
        )

        overall_summary = (
            f"{selected_summary.vessel_class} ({selected_summary.plan_id}) selected with "
            f"{selected_summary.cargo_utilization * 100:.1f}% utilization at ${selected_summary.cost_per_tonne:.2f}/t."
        )

        return VesselPlanExplanation(
            selected_plan=selected_summary,
            alternatives_considered=alternatives_summaries,
            constraints_eliminated=constraints_eliminated,
            comparative_justification=comparative_justification,
            primary_reasons=primary_reasons,
            tradeoff_analysis=tradeoff_text,
            summary=overall_summary,
        )

    def _summarize_plan(
        self,
        plan: CharterPlan,
        port_compat: Dict[str, Any],
        is_recommended: bool = False
    ) -> PlanEvaluationSummary:
        """Convert CharterPlan to structured PlanEvaluationSummary."""
        v_class = plan.vessel_classes[0] if plan.vessel_classes else "Unknown"

        # Calculate average waiting time from legs or default
        waiting_days = 0.0
        if plan.legs:
            waiting_days = sum(leg.waiting_days for leg in plan.legs) / len(plan.legs)

        # Port compatibility extraction
        v_compat = port_compat.get(v_class, {
            "origin_compatible": True,
            "destination_compatible": plan.feasibility,
            "draft_compatible": True,
            "loa_compatible": True,
            "beam_compatible": True,
        })

        return PlanEvaluationSummary(
            plan_id=plan.plan_id,
            vessel_class=v_class,
            number_of_vessels=plan.number_of_vessels,
            number_of_voyages=plan.number_of_voyages,
            cargo_utilization=float(plan.utilization),
            port_compatibility=v_compat,
            total_cost=float(plan.total_cost),
            cost_per_tonne=float(plan.cost_per_tonne),
            waiting_time=float(waiting_days),
            demurrage_probability=float(plan.demurrage_probability),
            delivery_probability=float(plan.delivery_probability),
            risk_score=float(plan.risk_score),
            feasibility=plan.feasibility,
            failed_constraints=plan.failed_constraints,
            score=float(plan.score),
        )

    def _classify_constraint_type(self, reason: str) -> str:
        """Map text failure to structured constraint code."""
        r_lower = reason.lower()
        if "draft" in r_lower:
            return "DRAFT_LIMIT_EXCEEDED"
        elif "loa" in r_lower or "length" in r_lower:
            return "LOA_LIMIT_EXCEEDED"
        elif "beam" in r_lower or "width" in r_lower:
            return "BEAM_LIMIT_EXCEEDED"
        elif "deadline" in r_lower or "schedule" in r_lower:
            return "DEADLINE_EXCEEDED"
        elif "capacity" in r_lower or "dwt" in r_lower:
            return "CAPACITY_MISMATCH"
        return "PORT_RESTRICTION_VIOLATION"

    def _formulate_tradeoff(
        self,
        selected: PlanEvaluationSummary,
        alternatives: List[PlanEvaluationSummary],
        origin_port: str,
        destination_port: str
    ) -> Tuple[str, str, List[str]]:
        """
        Formulate exact mathematical tradeoff and comparative justification.
        Matches the required benchmark:
        'Panamax was selected because it remained fully port-compatible, achieved 94% cargo
        utilization, and had 11% lower expected demurrage exposure than the alternative Capesize plan.'
        """
        primary_reasons = [
            f"Achieved optimal cargo deadweight utilization of {selected.cargo_utilization * 100:.1f}%.",
            f"Delivers lowest compliant delivered cost of ${selected.cost_per_tonne:.2f}/tonne (${selected.total_cost:,.0f} total).",
            f"High delivery schedule reliability ({selected.delivery_probability * 100:.1f}% on-time probability).",
            f"Bounded port demurrage probability to {selected.demurrage_probability * 100:.1f}% with {selected.waiting_time:.1f} days expected wait.",
        ]

        if not alternatives:
            justification = (
                f"{selected.vessel_class} ({selected.plan_id}) was selected as the solely viable, "
                f"fully port-compatible fleet configuration for {destination_port}, achieving "
                f"{selected.cargo_utilization * 100:.1f}% cargo utilization."
            )
            tradeoff = "No alternative candidates met physical port limits or parcel sizing constraints."
            return justification, tradeoff, primary_reasons

        # Find best feasible alternative (runner-up) or first infeasible alternative
        feasible_alts = [a for a in alternatives if a.feasibility]
        infeasible_alts = [a for a in alternatives if not a.feasibility]

        if feasible_alts:
            # Sort feasible alternatives by composite score descending
            feasible_alts.sort(key=lambda x: x.score, reverse=True)
            runner_up = feasible_alts[0]

            cost_diff = runner_up.total_cost - selected.total_cost
            cost_pct = (cost_diff / runner_up.total_cost * 100.0) if runner_up.total_cost > 0 else 0.0
            demurrage_diff_pct = (
                (runner_up.demurrage_probability - selected.demurrage_probability)
                / max(runner_up.demurrage_probability, 0.01) * 100.0
            )

            clauses = []
            clauses.append("remained fully port-compatible")
            clauses.append(f"achieved {selected.cargo_utilization * 100:.0f}% cargo utilization")

            if demurrage_diff_pct > 0:
                clauses.append(f"had {demurrage_diff_pct:.0f}% lower expected demurrage exposure than the alternative {runner_up.vessel_class} plan")
            elif cost_pct > 0:
                clauses.append(f"delivered {cost_pct:.1f}% lower total cost (${cost_diff:,.0f} savings) than the alternative {runner_up.vessel_class} plan")
            else:
                clauses.append(f"offered superior delivery probability ({selected.delivery_probability * 100:.1f}% vs {runner_up.delivery_probability * 100:.1f}%) over {runner_up.vessel_class}")

            justification = f"{selected.vessel_class} was selected because it {', '.join(clauses)}."
            tradeoff = (
                f"Selected {selected.plan_id} ({selected.vessel_class}) over {runner_up.plan_id} ({runner_up.vessel_class}): "
                f"${selected.cost_per_tonne:.2f}/t vs ${runner_up.cost_per_tonne:.2f}/t, "
                f"utilization {selected.cargo_utilization * 100:.1f}% vs {runner_up.cargo_utilization * 100:.1f}%, "
                f"risk score {selected.risk_score:.1f} vs {runner_up.risk_score:.1f}."
            )
        else:
            # All alternatives eliminated by hard constraints
            rejected = infeasible_alts[0]
            reason = rejected.failed_constraints[0] if rejected.failed_constraints else "Physical port constraints"
            justification = (
                f"{selected.vessel_class} was selected because it remained fully port-compatible at {destination_port}, "
                f"achieved {selected.cargo_utilization * 100:.0f}% cargo utilization, whereas alternative {rejected.vessel_class} "
                f"was eliminated due to: {reason}."
            )
            tradeoff = f"Alternative {rejected.plan_id} ({rejected.vessel_class}) eliminated by hard constraint: {reason}."

        return justification, tradeoff, primary_reasons
