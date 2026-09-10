"""
Integration Tests: Unified Decision Engine Pipeline.

Validates the full 8-step decision pipeline:
Cargo Request
  ↓
Freight Forecast
  ↓
Congestion Prediction
  ↓
Vessel Optimization
  ↓
Voyage Economics
  ↓
Risk Engine & Monte Carlo
  ↓
Contract Strategy
  ↓
Final Recommendation & Explainability
"""

from datetime import datetime, timedelta
import pytest

from src.optimization.decision_engine import (
    DecisionEngine,
    DecisionEngineInputs,
    MODEL_VERSIONS,
    DATA_VERSIONS,
)
from src.api.serializers import DecisionResponse
from src.data.mock_db import get_mock_port_info, get_mock_vessel_db


@pytest.fixture
def decision_engine():
    return DecisionEngine()


class TestDecisionPipelineIntegration:
    """Test full sequential pipeline integration with real domain services."""

    def test_complete_decision_pipeline_execution(self, decision_engine):
        """
        Executes complete pipeline:
        Cargo Request -> Forecast -> Congestion -> Vessel Optimizer -> Economics -> Risk -> Contract -> Decision
        """
        now = datetime(2026, 10, 1, 10, 0, 0)
        origin_p = get_mock_port_info("AUS_NEW")
        dest_p = get_mock_port_info("IND_GVM")

        inputs = DecisionEngineInputs(
            cargo_type="coal",
            cargo_quantity_t=82000.0,
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            expected_loading_date=now,
            required_delivery_date=now + timedelta(days=32),
            number_of_voyages=1,
            risk_tolerance="MEDIUM",
            origin_port_info=origin_p,
            destination_port_info=dest_p,
            vessel_specs_db=get_mock_vessel_db(),
        )

        result = decision_engine.evaluate(inputs)
        assert result["status"] == "SUCCESS"

        # Validate against canonical typed Pydantic schema
        validated = DecisionResponse(**result)

        # 1. Pipeline metadata
        assert validated.decision_id.startswith("dec_")
        assert validated.confidence > 0.0
        assert validated.model_versions == MODEL_VERSIONS
        assert validated.data_versions == DATA_VERSIONS

        # 2. Freight forecast step
        assert validated.market_analysis.current_rate > 0.0
        assert validated.market_analysis.forecast > 0.0
        assert validated.market_analysis.direction in ["RISING", "FALLING", "STABLE"]

        # 3. Vessel optimization step
        plan = validated.recommended_plan
        assert plan.vessel_class in ["Capesize", "Panamax", "Supramax", "Handysize"]
        assert plan.vessel_count >= 1
        assert sum(plan.cargo_allocation) == pytest.approx(82000.0, rel=1e-2)
        assert plan.utilization > 0.50

        # 4. Voyage economics step
        econ = validated.economics
        assert econ["total_cost"] > 0.0
        assert econ["freight_cost"] > 0.0
        assert econ["bunker_cost"] > 0.0
        assert econ["port_charges"] > 0.0
        assert econ["cost_per_tonne"] > 0.0

        # 5. Risk engine step
        risk = validated.risk
        assert risk["composite_score"] > 0.0
        assert len(risk["categories"]) >= 6

        # 6. Contract optimization step
        strat = validated.contract_strategy
        assert strat["recommended_strategy"] in [
            "100% SPOT", "100% SHORT_TERM", "100% MEDIUM_TERM",
            "75/25 HYBRID", "60/40 HYBRID", "50/50 HYBRID", "CUSTOM_HYBRID"
        ]
        total_contract_pct = (
            strat["spot_percentage"]
            + strat["short_term_percentage"]
            + strat["medium_term_percentage"]
        )
        assert total_contract_pct == pytest.approx(100.0)

        # 7. Final decision & explainability step
        exp = validated.explanation
        assert len(exp.primary_reasons) >= 3
        assert len(exp.tradeoff_analysis) > 5

    def test_pipeline_dynamic_responsiveness(self, decision_engine):
        """Pipeline must respond dynamically when cargo quantity changes (no hardcoded answers)."""
        now = datetime(2026, 10, 1, 10, 0, 0)
        # Small cargo -> Handysize / Supramax
        inputs_small = DecisionEngineInputs(
            cargo_type="coal",
            cargo_quantity_t=35000.0,
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            expected_loading_date=now,
            required_delivery_date=now + timedelta(days=30),
            number_of_voyages=1,
            origin_port_info=get_mock_port_info("AUS_NEW"),
            destination_port_info=get_mock_port_info("IND_GVM"),
            vessel_specs_db=get_mock_vessel_db(),
        )
        # Large cargo -> Capesize / Panamax
        inputs_large = DecisionEngineInputs(
            cargo_type="coal",
            cargo_quantity_t=160000.0,
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            expected_loading_date=now,
            required_delivery_date=now + timedelta(days=35),
            number_of_voyages=1,
            origin_port_info=get_mock_port_info("AUS_NEW"),
            destination_port_info=get_mock_port_info("IND_GVM"),
            vessel_specs_db=get_mock_vessel_db(),
        )

        res_small = decision_engine.evaluate(inputs_small)
        res_large = decision_engine.evaluate(inputs_large)

        assert res_small["economics"]["total_cost"] < res_large["economics"]["total_cost"]
        assert res_small["recommended_plan"]["vessel_class"] != res_large["recommended_plan"]["vessel_class"]
