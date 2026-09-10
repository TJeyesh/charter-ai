"""
Charter-AI — Explainability Module (Phase 12).

Exports Explainable AI (XAI) engines for:
1. Freight Forecasting with SHAP TreeExplainer attribution.
2. Multi-voyage vessel selection and constraint tracking.
3. Maritime risk analysis and contract strategy optimization.
4. Master decision explanation answering the 7 core questions.
"""

from src.explainability.forecast_explainer import (
    ForecastExplainer,
    ForecastExplanation,
    ShapFeatureContribution,
    FEATURE_DISPLAY_NAMES,
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

__all__ = [
    "ForecastExplainer",
    "ForecastExplanation",
    "ShapFeatureContribution",
    "FEATURE_DISPLAY_NAMES",
    "VesselExplainer",
    "VesselPlanExplanation",
    "PlanEvaluationSummary",
    "ConstraintElimination",
    "RiskExplainer",
    "RiskExplanation",
    "RiskCategoryDriver",
    "ContractStrategyExplanation",
    "DecisionExplainer",
    "UnifiedDecisionExplanation",
    "StructuredSevenAnswers",
]
