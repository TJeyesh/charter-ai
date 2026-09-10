"""
Unit Tests: Port Congestion and Demurrage Engine.

Verifies port waiting time prediction, queue metrics, laytime allowances,
demurrage penalty calculations, and despatch savings credits.
"""

from datetime import date
import pytest

from src.models.congestion_predictor import CongestionPredictor
from src.economics.voyage_cost import (
    VoyageCostInputs,
    calculate_voyage_cost,
)


class TestPortCongestionPrediction:
    """Test congestion prediction service and waiting days quantiles."""

    def test_predict_congestion_quantiles(self):
        predictor = CongestionPredictor()
        res = predictor.predict_congestion(
            port_id="IND_GVM",
            target_date=date(2026, 4, 15),
            vessel_class="Capesize",
        )

        assert res.expected_wait_days >= 0.0
        assert res.p10_wait_days <= res.expected_wait_days <= res.p90_wait_days
        assert 0.0 <= res.delay_probability <= 1.0

    def test_congestion_risk_level_assignment(self):
        predictor = CongestionPredictor()
        res = predictor.predict_congestion(
            port_id="IND_HLD",
            target_date=date(2026, 4, 15),
            vessel_class="Panamax",
        )
        assert res.congestion_level in ["LOW", "MODERATE", "HIGH", "SEVERE"]


class TestDemurrageAndLaytimeCalculation:
    """Test contractual laytime, excess time, and demurrage/despatch computation."""

    def test_zero_demurrage_within_laytime(self):
        """No demurrage incurred if total port days <= laytime allowed."""
        inputs = VoyageCostInputs(
            cargo_quantity_t=50000.0,
            freight_rate_usd=20.0,
            vessel_speed_knots=12.0,
            vessel_daily_fuel_consumption_tpd=25.0,
            vessel_daily_hire_cost_usd=15000.0,
            route_distance_nm=2880.0,
            port_handling_rate_tpd=25000.0,
            discharge_port_handling_rate_tpd=25000.0,
            expected_waiting_days=1.0,
            laytime_allowed_days=6.0,
            daily_demurrage_rate_usd=20000.0,
        )
        res = calculate_voyage_cost(inputs)
        assert res.total_port_days == 5.0
        assert res.excess_time_days == 0.0
        assert res.demurrage_exposure == 0.0

    def test_demurrage_incurred_on_excess_time(self):
        """Demurrage is strictly excess_time_days * daily_demurrage_rate_usd."""
        inputs = VoyageCostInputs(
            cargo_quantity_t=60000.0,
            freight_rate_usd=22.0,
            vessel_speed_knots=12.5,
            vessel_daily_fuel_consumption_tpd=28.0,
            vessel_daily_hire_cost_usd=18000.0,
            route_distance_nm=3000.0,
            port_handling_rate_tpd=20000.0,
            discharge_port_handling_rate_tpd=20000.0,
            expected_waiting_days=4.5,
            laytime_allowed_days=6.5,
            daily_demurrage_rate_usd=30000.0,
        )
        res = calculate_voyage_cost(inputs)
        assert res.excess_time_days == pytest.approx(4.0, abs=1e-3)
        assert res.demurrage_exposure == pytest.approx(4.0 * 30000.0, abs=1e-2)

    def test_despatch_savings_credit(self):
        """Despatch is earned at agreed fraction of demurrage when turnaround is fast."""
        inputs = VoyageCostInputs(
            cargo_quantity_t=40000.0,
            freight_rate_usd=18.0,
            vessel_speed_knots=12.0,
            vessel_daily_fuel_consumption_tpd=22.0,
            vessel_daily_hire_cost_usd=14000.0,
            route_distance_nm=2400.0,
            port_handling_rate_tpd=20000.0,
            discharge_port_handling_rate_tpd=20000.0,
            expected_waiting_days=0.0,
            laytime_allowed_days=6.0,
            daily_demurrage_rate_usd=24000.0,
            despatch_rate_fraction=0.50,
        )
        res = calculate_voyage_cost(inputs)
        assert res.demurrage_exposure == 0.0
        assert res.despatch_savings == pytest.approx(2.0 * 12000.0, abs=1e-2)
