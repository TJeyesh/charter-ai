"""
Unit Tests: Market Timing and Contract Strategy Optimization.

Verifies market timing action recommendations (BOOK_NOW, WAIT, MONITOR, etc.),
waiting benefit calculations, and contract strategy allocations (Spot vs Short vs Medium term).
"""

from datetime import datetime, timedelta
import pytest

from src.models.market_timing import (
    MarketTimingEngine,
    MarketTimingInputs,
    TimingAction,
)
from src.optimization.contract_optimizer import (
    RiskAwareContractOptimizer,
    RiskAwareContractInputs,
    RiskTolerance,
    VesselAvailability,
)


class TestMarketTimingEngine:
    """Test timing action logic, net waiting benefits, and deadline penalties."""

    def test_timing_book_now_when_rates_rising(self):
        """When rates are projected to rise significantly, engine recommends BOOK_NOW or START_NEGOTIATION."""
        engine = MarketTimingEngine()
        inputs = MarketTimingInputs(
            current_freight_rate=20.0,
            forecast_rate=25.0,  # +$5/t increase expected
            p10_forecast=22.0,
            p50_forecast=25.0,
            p90_forecast=29.0,
            market_momentum=2.0,
            freight_volatility=2.5,
            vessel_availability="TIGHT",
            congestion_forecast=3.0,
            cargo_deadline=12.0,
            cargo_quantity_t=75000.0,
        )
        res = engine.evaluate_timing(inputs)

        assert res.recommendation in [TimingAction.BOOK_NOW.value, TimingAction.START_NEGOTIATION.value]
        assert res.confidence > 0.0
        assert res.net_waiting_benefit <= 0.0

    def test_timing_wait_when_rates_falling(self):
        """When rates are projected to decline and deadline allows, engine recommends WAIT."""
        engine = MarketTimingEngine()
        inputs = MarketTimingInputs(
            current_freight_rate=30.0,
            forecast_rate=22.0,  # -$8/t decline expected
            p10_forecast=18.0,
            p50_forecast=22.0,
            p90_forecast=25.0,
            market_momentum=-3.0,
            freight_volatility=1.5,
            vessel_availability="SURPLUS",
            congestion_forecast=1.0,
            cargo_deadline=50.0,  # 50 days deadline gives 30 days buffer
            cargo_quantity_t=75000.0,
        )
        res = engine.evaluate_timing(inputs)

        assert res.recommendation in [TimingAction.WAIT.value, TimingAction.MONITOR.value]
        assert res.expected_savings > 0.0


class TestRiskAwareContractOptimizer:
    """Test contract strategy optimization (Spot, Short-Term CoA, Medium-Term)."""

    def test_contract_allocation_percentages_sum_to_100(self):
        optimizer = RiskAwareContractOptimizer()
        inputs = RiskAwareContractInputs(
            cargo_quantity_t=80000.0,
            spot_freight_rate=22.50,
            freight_volatility_pct=20.0,
            risk_tolerance=RiskTolerance.MEDIUM,
            n_simulations=1000,
            seed=42,
        )
        rec = optimizer.optimize_contract_strategy(inputs)

        assert rec.recommended_strategy is not None
        total_pct = rec.spot_percentage + rec.short_term_percentage + rec.medium_term_percentage
        assert total_pct == pytest.approx(100.0, abs=1e-2)

    def test_low_risk_tolerance_favors_term_cover(self):
        """Low risk tolerance should allocate less to spot than high risk tolerance."""
        optimizer = RiskAwareContractOptimizer()
        inputs_low = RiskAwareContractInputs(
            cargo_quantity_t=100000.0,
            spot_freight_rate=25.0,
            short_term_freight_rate=26.0,
            medium_term_freight_rate=26.5,
            freight_volatility_pct=35.0,
            risk_tolerance=RiskTolerance.LOW,
            vessel_availability=VesselAvailability.TIGHT,
            n_simulations=1000,
            seed=42,
        )
        inputs_high = RiskAwareContractInputs(
            cargo_quantity_t=100000.0,
            spot_freight_rate=25.0,
            short_term_freight_rate=26.0,
            medium_term_freight_rate=26.5,
            freight_volatility_pct=35.0,
            risk_tolerance=RiskTolerance.HIGH,
            vessel_availability=VesselAvailability.TIGHT,
            n_simulations=1000,
            seed=42,
        )
        rec_low = optimizer.optimize_contract_strategy(inputs_low)
        rec_high = optimizer.optimize_contract_strategy(inputs_high)

        assert rec_low.spot_percentage <= rec_high.spot_percentage
        assert rec_high.flexibility_score >= rec_low.flexibility_score
