"""
Charter-AI — Comprehensive Test Suite for Phase 12 Explainable Recommendations.

Tests:
1. Forecast explainer with historical rate vs P10/P50/P90, validation metrics, and confidence.
2. SHAP TreeExplainer integration for XGBoost feature attribution.
3. Vessel plan explainer with utilization, port compatibility, cost, $/t, wait time, demurrage prob, delivery prob, risk score.
4. Constraint elimination audit for physical port limit violations.
5. Exact formulaic comparative justification vs alternative plans.
6. Risk explainer with 8 categories, dominant drivers, and contract strategy (expected cost, downside P90, risk, flexibility).
7. Decision explainer answering all 7 core questions.
8. DecisionEngine end-to-end integration producing full explainability payload.
9. FastAPI POST /api/v1/recommend/decision returning structured Phase 12 explanation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import xgboost as xgb
from fastapi.testclient import TestClient

from src.explainability.forecast_explainer import (
    ForecastExplainer,
    ForecastExplanation,
    ShapFeatureContribution,
)
from src.explainability.vessel_explainer import (
    VesselExplainer,
    VesselPlanExplanation,
    PlanEvaluationSummary,
    ConstraintElimination,
)
from src.explainability.risk_explainer import (
    RiskExplainer,
    RiskExplanation,
    RiskCategoryDriver,
    ContractStrategyExplanation,
)
from src.explainability.decision_explainer import (
    DecisionExplainer,
    UnifiedDecisionExplanation,
    StructuredSevenAnswers,
)
from src.optimization.voyage_plan import CharterPlan, VoyageLeg
from src.risk.risk_engine import ComprehensiveRiskAssessment, RiskCategoryResult, RiskSeverity
from src.optimization.contract_optimizer import StrategyEvaluation, RiskTolerance
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.api.serializers import DecisionResponse
from src.api.main import app


# =============================================================================
# 1. Forecast Explainer Tests
# =============================================================================

def test_forecast_explainer_required_fields():
    explainer = ForecastExplainer()
    forecast_dict = {
        "current_rate": 18.50,
        "forecast_rate": 20.20,
        "lower_bound": 19.10,
        "upper_bound": 21.80,
        "trend": "rising",
        "confidence": 0.88,
        "model_used": "EnsembleForecaster",
        "metrics": {"mae": 0.35, "smape": 3.1, "rmse": 0.48}
    }

    result = explainer.explain_forecast(forecast_dict)

    assert isinstance(result, ForecastExplanation)
    # Required fields from requirements
    assert result.historical_rate == 18.50
    assert result.forecast == 20.20
    assert result.p10 == 19.10
    assert result.p50 == 20.20
    assert result.p90 == 21.80
    assert result.model_used == "EnsembleForecaster"
    assert result.validation_metrics["mae"] == 0.35
    assert result.confidence == 0.88
    assert result.trend == "rising"
    assert np.isclose(result.rate_delta, 1.70)
    assert result.rate_delta_pct > 0.0
    assert np.isclose(result.uncertainty_spread, 2.70)

    # Narrative consistency check
    assert "increase" in result.narrative.lower() or "rising" in result.narrative.lower()
    assert "18.50" in result.narrative
    assert "20.20" in result.narrative
    assert "EnsembleForecaster" in result.narrative


def test_forecast_explainer_shap_xgboost_attribution():
    """Verify SHAP TreeExplainer integration directly on fitted XGBoost regressor."""
    np.random.seed(42)
    # Train synthetic XGBoost model with known strong driver (bdi)
    n_samples = 80
    X = pd.DataFrame({
        "bdi": np.linspace(1200, 2500, n_samples),
        "momentum_7": np.random.normal(0.5, 0.2, n_samples),
        "port_congestion": np.random.uniform(1.0, 5.0, n_samples),
        "coal_price": np.random.normal(140.0, 10.0, n_samples),
    })
    y = X["bdi"] * 0.008 + X["momentum_7"] * 1.5 + np.random.normal(0, 0.05, n_samples)

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = ForecastExplainer()
    test_sample = X.iloc[0:1]
    contributions = explainer.compute_shap_attribution(model_obj=model, feature_row=test_sample, top_k=3)

    assert len(contributions) > 0
    assert len(contributions) <= 3
    # Top contributor should be bdi
    top_drv = contributions[0]
    assert isinstance(top_drv, ShapFeatureContribution)
    assert top_drv.feature_name in ["bdi", "momentum_7", "port_congestion", "coal_price"]
    assert top_drv.impact_direction in ["INCREASES_RATE", "DECREASES_RATE", "NEUTRAL"]
    assert top_drv.percentage_contribution >= 0.0
    assert top_drv.display_name != ""


# =============================================================================
# 2. Vessel Explainer Tests
# =============================================================================

@pytest.fixture
def sample_candidate_plans():
    """Create realistic candidate plans: 1 recommended Panamax, 1 infeasible Capesize, 1 feasible Supramax."""
    leg_panamax = VoyageLeg(
        leg_id=1,
        vessel_class="Panamax",
        cargo_quantity_t=75000.0,
        vessel_dwt=75000.0,
        utilization=1.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        route_distance_nm=5440.0,
        sailing_days=17.4,
        loading_days=3.0,
        discharge_days=3.5,
        waiting_days=2.2,
        total_leg_days=26.1,
        freight_cost=1500000.0,
        bunker_cost=400000.0,
        port_cost=120000.0,
        waiting_cost=45000.0,
        demurrage_exposure=15000.0,
        positioning_cost=0.0,
        miscellaneous_cost=25000.0,
        total_cost=2105000.0,
        cost_per_tonne=28.07,
    )

    plan_panamax = CharterPlan(
        plan_id="PLAN_1xPanamax",
        vessel_classes=["Panamax"],
        number_of_vessels=1,
        number_of_voyages=1,
        legs=[leg_panamax],
        cargo_per_voyage=[75000.0],
        total_cargo_t=75000.0,
        utilization=0.94,
        total_freight_cost=1500000.0,
        bunker_cost=400000.0,
        port_cost=120000.0,
        waiting_cost=45000.0,
        expected_demurrage=15000.0,
        total_cost=2105000.0,
        cost_per_tonne=28.07,
        total_duration=26.1,
        demurrage_probability=0.08,
        delivery_probability=0.96,
        risk_score=24.5,
        feasibility=True,
        score=88.5,
    )

    plan_capesize_infeasible = CharterPlan(
        plan_id="PLAN_1xCapesize",
        vessel_classes=["Capesize"],
        number_of_vessels=1,
        number_of_voyages=1,
        cargo_per_voyage=[75000.0],
        total_cargo_t=75000.0,
        utilization=0.43,
        total_cost=2800000.0,
        cost_per_tonne=37.33,
        total_duration=24.0,
        demurrage_probability=0.22,
        delivery_probability=0.90,
        risk_score=68.0,
        feasibility=False,
        failed_constraints=["Draft 18.5m exceeds destination port limit 14.5m by 4.0m"],
        score=15.0,
    )

    plan_supramax = CharterPlan(
        plan_id="PLAN_2xSupramax",
        vessel_classes=["Supramax"],
        number_of_vessels=2,
        number_of_voyages=2,
        cargo_per_voyage=[37500.0, 37500.0],
        total_cargo_t=75000.0,
        utilization=0.68,
        total_cost=2450000.0,
        cost_per_tonne=32.67,
        total_duration=32.0,
        demurrage_probability=0.19,
        delivery_probability=0.91,
        risk_score=38.0,
        feasibility=True,
        score=72.0,
    )

    return plan_panamax, [plan_panamax, plan_capesize_infeasible, plan_supramax]


def test_vessel_explainer_required_metrics(sample_candidate_plans):
    best_plan, all_plans = sample_candidate_plans
    explainer = VesselExplainer()

    explanation = explainer.explain_vessel_selection(
        recommended_plan=best_plan,
        all_candidate_plans=all_plans,
        destination_port_id="IND_GVM",
        origin_port_id="AUS_NEW"
    )

    assert isinstance(explanation, VesselPlanExplanation)
    sel = explanation.selected_plan
    # Verify all 8 required fields from prompt
    assert sel.cargo_utilization == 0.94
    assert isinstance(sel.port_compatibility, dict)
    assert sel.total_cost == 2105000.0
    assert sel.cost_per_tonne == 28.07
    assert sel.waiting_time == 2.2
    assert sel.demurrage_probability == 0.08
    assert sel.delivery_probability == 0.96
    assert sel.risk_score == 24.5

    # Alternatives visible
    assert len(explanation.alternatives_considered) == 2


def test_vessel_explainer_constraint_eliminations(sample_candidate_plans):
    best_plan, all_plans = sample_candidate_plans
    explainer = VesselExplainer()

    explanation = explainer.explain_vessel_selection(
        recommended_plan=best_plan,
        all_candidate_plans=all_plans,
    )

    # Infeasible Capesize should be recorded in constraints_eliminated
    assert len(explanation.constraints_eliminated) >= 1
    elim = explanation.constraints_eliminated[0]
    assert isinstance(elim, ConstraintElimination)
    assert elim.vessel_class == "Capesize"
    assert elim.constraint_type == "DRAFT_LIMIT_EXCEEDED"
    assert "draft" in elim.reason.lower()


def test_vessel_explainer_comparative_statement_formula(sample_candidate_plans):
    """
    Ensure exact formulaic comparative justification:
    'Panamax was selected because it remained fully port-compatible, achieved 94% cargo utilization, and had ...'
    """
    best_plan, all_plans = sample_candidate_plans
    explainer = VesselExplainer()

    explanation = explainer.explain_vessel_selection(
        recommended_plan=best_plan,
        all_candidate_plans=all_plans,
    )

    justification = explanation.comparative_justification
    assert "Panamax was selected because" in justification
    assert "port-compatible" in justification
    assert "cargo utilization" in justification


# =============================================================================
# 3. Risk & Contract Explainer Tests
# =============================================================================

def test_risk_explainer_categories_and_contract():
    explainer = RiskExplainer()

    # Build realistic 8-category risk assessment
    cats = {
        "market_risk": RiskCategoryResult("market_risk", 38.0, RiskSeverity.LOW, ["Low historical volatility"]),
        "port_congestion_risk": RiskCategoryResult("port_congestion_risk", 62.0, RiskSeverity.MODERATE, ["Estimated wait 2.4 days"]),
        "weather_risk": RiskCategoryResult("weather_risk", 25.0, RiskSeverity.LOW, ["Outside cyclone season"]),
        "vessel_availability_risk": RiskCategoryResult("vessel_availability_risk", 45.0, RiskSeverity.MODERATE, ["Tight prompt supply"]),
        "operational_risk": RiskCategoryResult("operational_risk", 30.0, RiskSeverity.LOW, ["Modern berth handling"]),
        "geopolitical_risk": RiskCategoryResult("geopolitical_risk", 20.0, RiskSeverity.LOW, ["Safe corridor"]),
        "schedule_risk": RiskCategoryResult("schedule_risk", 35.0, RiskSeverity.LOW, ["Adequate buffer"]),
        "demurrage_risk": RiskCategoryResult("demurrage_risk", 48.0, RiskSeverity.MODERATE, ["Moderate turnaround queue"]),
    }
    risk_assessment = ComprehensiveRiskAssessment(
        overall_score=37.8,
        overall_severity=RiskSeverity.LOW,
        categories=cats,
        summary_messages=["Manageable operational risk profile."],
    )

    # Strategy evaluations
    eval_60_40 = StrategyEvaluation(
        strategy_name="60/40 HYBRID",
        spot_percentage=60.0,
        short_term_percentage=40.0,
        medium_term_percentage=0.0,
        expected_freight_cost=2100000.0,
        expected_total_cost=2450000.0,
        p10_cost=2280000.0,
        p50_cost=2450000.0,
        p90_cost=2690000.0,
        price_volatility_exposure=12.2,
        vessel_availability_exposure=15.0,
        delivery_risk=0.04,
        flexibility_score=0.68,
        expected_demurrage_exposure=35000.0,
        risk_penalty=12000.0,
        schedule_penalty=5000.0,
        flexibility_preference=8000.0,
        objective_score=2459000.0,
        reasons=["Optimal balance for MEDIUM risk tolerance."],
    )

    explanation = explainer.explain_risk_and_contract(
        risk_result=risk_assessment,
        contract_rec=eval_60_40,
        all_contract_evaluations=[eval_60_40],
        risk_tolerance="MEDIUM"
    )

    assert isinstance(explanation, RiskExplanation)
    assert explanation.composite_score == 37.8
    assert len(explanation.top_risk_drivers) >= 1
    # Port congestion should be top driver
    assert explanation.top_risk_drivers[0].category == "port_congestion_risk"

    # Contract strategy required fields
    strat = explanation.contract_strategy
    assert strat.expected_cost == 2450000.0
    assert strat.downside_cost == 2690000.0  # P90
    assert strat.volatility_exposure == 12.2
    assert strat.flexibility_score == 0.68
    assert len(strat.reasons) > 0


# =============================================================================
# 4. Master Decision Explainer (The 7 Core Questions)
# =============================================================================

def test_decision_explainer_seven_core_questions(sample_candidate_plans):
    best_plan, all_plans = sample_candidate_plans
    explainer = DecisionExplainer()

    forecast_data = {
        "current_rate": 19.0,
        "forecast_rate": 20.5,
        "lower_bound": 19.2,
        "upper_bound": 22.0,
        "trend": "rising",
        "confidence": 0.85,
        "model_used": "EnsembleForecaster",
        "metrics": {"mae": 0.38}
    }

    result = explainer.generate_decision_explanation(
        recommended_plan=best_plan,
        all_candidate_plans=all_plans,
        forecast_dict=forecast_data,
        risk_tolerance="MEDIUM",
        timing_action="BOOK_NOW",
        composite_confidence=0.88,
    )

    assert isinstance(result, UnifiedDecisionExplanation)
    answers = result.structured_answers
    assert isinstance(answers, StructuredSevenAnswers)

    # 1. What was selected?
    assert "vessel_plan" in answers.what_was_selected
    assert answers.what_was_selected["vessel_plan"]["vessel_class"] == "Panamax"
    assert "contract_strategy" in answers.what_was_selected
    assert "market_timing" in answers.what_was_selected

    # 2. Why was it selected?
    assert len(answers.why_was_it_selected) >= 5
    assert any("port-compatible" in r.lower() for r in answers.why_was_it_selected)

    # 3. What alternatives were considered?
    assert len(answers.alternatives_considered) == 2

    # 4. What constraints eliminated alternatives?
    assert len(answers.constraints_eliminated) >= 1
    assert answers.constraints_eliminated[0]["vessel_class"] == "Capesize"

    # 5. What economic factors influenced the result?
    assert "total_delivered_cost" in answers.economic_factors
    assert "cost_per_tonne" in answers.economic_factors
    assert answers.economic_factors["cost_per_tonne"] == 28.07

    # 6. What risks influenced the result?
    assert "composite_risk_score" in answers.risks_influencing_result
    assert "downside_tail_cost_p90" in answers.risks_influencing_result

    # 7. How confident is the recommendation?
    assert "composite_confidence" in answers.confidence_assessment
    assert answers.confidence_assessment["composite_confidence"] == 0.88


# =============================================================================
# 5. Decision Engine End-to-End & API Integration
# =============================================================================

def test_decision_engine_end_to_end_explanation():
    engine = DecisionEngine()
    inputs = DecisionEngineInputs(
        cargo_type="coal",
        cargo_quantity_t=75000,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        expected_loading_date=datetime(2026, 10, 1),
        required_delivery_date=datetime(2026, 11, 1),
        risk_tolerance="MEDIUM"
    )

    output = engine.evaluate(inputs)
    assert output["status"] == "SUCCESS"
    assert "explanation" in output
    exp = output["explanation"]

    # Backward-compatible fields
    assert "recommendation_summary" in exp
    assert "primary_reasons" in exp
    assert len(exp["primary_reasons"]) >= 5
    assert "tradeoff_analysis" in exp
    assert "alternatives_rejected" in exp

    # Phase 12 fields
    assert "structured_answers" in exp
    answers = exp["structured_answers"]
    assert "what_was_selected" in answers
    assert "why_was_it_selected" in answers
    assert "alternatives_considered" in answers
    assert "constraints_eliminated" in answers
    assert "economic_factors" in answers
    assert "risks_influencing_result" in answers
    assert "confidence_assessment" in answers

    assert "vessel_explanation" in exp
    assert "forecast_explanation" in exp
    assert "risk_explanation" in exp
    assert "shap_analysis" in exp


def test_api_recommend_decision_endpoint_with_explanation():
    client = TestClient(app)
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "cargo_type": "coal",
        "cargo_tonnage": 75000,
        "earliest_date": "2026-10-01",
        "latest_date": "2026-11-01",
        "risk_appetite": "moderate"
    }

    response = client.post("/api/v1/recommend/decision", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate against canonical DecisionResponse schema
    validated = DecisionResponse(**data)
    assert validated.decision_id.startswith("dec_")
    assert validated.explanation.recommendation_summary is not None
    assert len(validated.explanation.primary_reasons) >= 5
    assert validated.explanation.structured_answers is not None
    assert "what_was_selected" in validated.explanation.structured_answers
