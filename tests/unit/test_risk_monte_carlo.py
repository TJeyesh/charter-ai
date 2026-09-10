"""
Unit Tests: Maritime Risk Engine and Probabilistic Monte Carlo Simulation.

Verifies the 8 maritime risk categories, dynamic sensitivity scaling,
composite risk index calculation, and Monte Carlo quantile reproducibility.
"""

import pytest
import numpy as np

from src.risk.risk_engine import MaritimeRiskEngine, RiskSeverity
from src.risk.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloSimulationConfig,
    CharterPlanInputs,
)


class TestMaritimeRiskEngine:
    """Test 8-dimension maritime risk model and composite scoring."""

    def test_eight_risk_dimensions_present(self):
        engine = MaritimeRiskEngine()
        inputs = {
            "market_data": {"price_volatility_pct": 15.0, "p10": 18.0, "p90": 25.0, "p50": 21.0},
            "port_data": {"expected_wait_days": 3.0, "vessels_waiting": 5, "delay_probability": 0.35},
            "weather_data": {"wave_height_m": 2.0, "wind_speed_kmh": 35.0, "storm_warning": False},
            "vessel_data": {"available_vessels_in_region": 8, "lead_time_days": 5.0},
            "operational_data": {"vessel_age_years": 10.0, "maintenance_due": False, "cargo_type": "coal"},
            "geopolitical_data": {"route_conflict_level": 1.0, "chokepoints": ["malacca"]},
            "schedule_data": {"total_duration": 15.0, "delivery_deadline_days": 25.0},
            "demurrage_data": {"expected_demurrage": 15000.0, "freight_cost": 300000.0},
        }
        res = engine.evaluate_total_risk(inputs)

        assert 0.0 <= res.overall_score <= 100.0
        assert res.overall_severity in [RiskSeverity.LOW, RiskSeverity.MODERATE, RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        assert "Market" in res.categories
        assert "Port Congestion" in res.categories
        assert "Weather" in res.categories
        assert "Vessel Availability" in res.categories
        assert "Operational" in res.categories
        assert "Geopolitical" in res.categories
        assert "Schedule" in res.categories
        assert "Demurrage" in res.categories

    def test_dynamic_risk_sensitivity(self):
        """Increasing congestion wait days must monotonically increase port congestion risk."""
        engine = MaritimeRiskEngine()
        low_cong = {
            "market_data": {}, "port_data": {"expected_wait_days": 1.0, "delay_probability": 0.1},
            "weather_data": {}, "vessel_data": {}, "operational_data": {},
            "geopolitical_data": {}, "schedule_data": {}, "demurrage_data": {},
        }
        high_cong = {
            "market_data": {}, "port_data": {"expected_wait_days": 12.0, "delay_probability": 0.9},
            "weather_data": {}, "vessel_data": {}, "operational_data": {},
            "geopolitical_data": {}, "schedule_data": {}, "demurrage_data": {},
        }

        res_low = engine.evaluate_total_risk(low_cong)
        res_high = engine.evaluate_total_risk(high_cong)

        port_low = res_low.categories["Port Congestion"].score
        port_high = res_high.categories["Port Congestion"].score

        assert port_high > port_low
        assert res_high.overall_score > res_low.overall_score


class TestMonteCarloSimulation:
    """Test probabilistic Monte Carlo simulation and distribution quantiles."""

    def test_simulation_reproducibility_with_seed(self):
        """Fixed random seeds must produce byte-for-byte identical quantiles."""
        plan_inputs = CharterPlanInputs(
            cargo_quantity_t=82000.0,
            base_freight_rate=22.50,
            sea_distance_nm=3500.0,
            service_speed_knots=13.0,
            fuel_consumption_t_day=28.0,
            base_bunker_price=650.0,
            expected_wait_days=2.5,
            demurrage_rate_usd_day=25000.0,
            delivery_deadline_days=25.0,
        )

        sim1 = MonteCarloSimulator(MonteCarloSimulationConfig(n_simulations=1000, seed=42))
        res1 = sim1.run_simulation(plan_inputs)

        sim2 = MonteCarloSimulator(MonteCarloSimulationConfig(n_simulations=1000, seed=42))
        res2 = sim2.run_simulation(plan_inputs)

        assert res1.p50_cost == pytest.approx(res2.p50_cost, abs=1e-5)
        assert res1.p95_cost == pytest.approx(res2.p95_cost, abs=1e-5)
        assert res1.late_delivery_probability == pytest.approx(res2.late_delivery_probability, abs=1e-5)

    def test_quantile_ordering_and_delivery_prob(self):
        """P10 <= P50 <= P90 <= P95 cost must hold."""
        plan_inputs = CharterPlanInputs(
            cargo_quantity_t=160000.0,
            base_freight_rate=19.50,
            sea_distance_nm=4000.0,
            service_speed_knots=12.5,
            fuel_consumption_t_day=42.0,
            base_bunker_price=650.0,
            expected_wait_days=3.0,
            demurrage_rate_usd_day=35000.0,
            delivery_deadline_days=30.0,
        )
        sim = MonteCarloSimulator(MonteCarloSimulationConfig(n_simulations=1000, seed=123))
        res = sim.run_simulation(plan_inputs)

        assert res.p10_cost <= res.p50_cost <= res.p90_cost <= res.p95_cost
        assert 0.0 <= res.late_delivery_probability <= 1.0
        assert res.expected_cost > 0.0
