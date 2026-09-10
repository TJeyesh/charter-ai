"""
Unit Tests: Comprehensive Edge Cases Suite.

Verifies system resilience against 13 critical edge conditions:
1. cargo_quantity = 0
2. negative cargo quantity
3. cargo exceeds all vessels
4. vessel draft exceeds port limit
5. missing port
6. missing route
7. missing freight history
8. missing congestion history
9. extreme bunker price ($10,000/t and $0.01/t)
10. extreme congestion (30 days delay)
11. unavailable vessel
12. impossible delivery deadline
13. no feasible vessel plan
"""

from datetime import date, datetime, timedelta
import math
import pytest
from pydantic import ValidationError

from src.api.serializers import (
    AnalyzeVoyageRequest,
    CargoRequest,
    RecommendationRequest,
    VesselOptimizationRequest,
)
from src.economics.voyage_cost import (
    VoyageCostInputs,
    calculate_voyage_cost,
    calculate_delivery_probability,
)
from src.models.congestion_predictor import CongestionPredictor
from src.models.freight_forecaster import FreightForecaster
from src.models.market_timing import (
    MarketTimingEngine,
    MarketTimingInputs,
    TimingAction,
)
from src.optimization.vessel_selector import (
    PortConstraints,
    VesselSelector,
    VesselOptimizer,
    VesselSpecs,
)
from src.risk.risk_engine import MaritimeRiskEngine


@pytest.fixture
def standard_vessels():
    return [
        VesselSpecs("Handysize", 10_000, 40_000, 32_000, 11.0, 190.0, 30.0),
        VesselSpecs("Supramax", 40_000, 60_000, 56_000, 13.0, 200.0, 32.5),
        VesselSpecs("Panamax", 60_000, 100_000, 82_000, 14.5, 240.0, 36.0),
        VesselSpecs("Capesize", 120_000, 220_000, 180_000, 18.5, 300.0, 50.0),
    ]


class TestCriticalEdgeCases:
    """Test the 13 required edge cases across components."""

    # 1. cargo_quantity = 0
    def test_edge_case_1_cargo_quantity_zero_validation(self):
        """Pydantic schemas must reject cargo quantity == 0."""
        with pytest.raises(ValidationError):
            RecommendationRequest(
                origin_port_id="AUS_NEW",
                destination_port_id="IND_GVM",
                cargo_type="coal",
                cargo_tonnage=0,
                earliest_date=date(2026, 4, 1),
                latest_date=date(2026, 4, 25),
            )

        with pytest.raises(ValidationError):
            AnalyzeVoyageRequest(
                cargo_type="coal",
                cargo_quantity=0.0,
                origin="AUS_NEW",
                destination="IND_GVM",
                required_delivery_date=date(2026, 4, 25),
                number_of_voyages=1,
            )

    def test_edge_case_1_cargo_quantity_zero_economics(self):
        """Voyage economics must handle zero cargo without ZeroDivisionError."""
        inputs = VoyageCostInputs(
            cargo_quantity_t=0.0,
            freight_rate_usd=20.0,
            vessel_speed_knots=12.0,
            vessel_daily_fuel_consumption_tpd=25.0,
            vessel_daily_hire_cost_usd=15000.0,
            route_distance_nm=2400.0,
        )
        res = calculate_voyage_cost(inputs)
        assert res.cost_per_tonne == 0.0
        assert res.total_cost >= 0.0

    # 2. negative cargo quantity
    def test_edge_case_2_negative_cargo_quantity(self):
        """Negative cargo quantities must be rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RecommendationRequest(
                origin_port_id="AUS_NEW",
                destination_port_id="IND_GVM",
                cargo_type="coal",
                cargo_tonnage=-50000,
                earliest_date=date(2026, 4, 1),
                latest_date=date(2026, 4, 25),
            )

        with pytest.raises(ValidationError):
            AnalyzeVoyageRequest(
                cargo_type="coal",
                cargo_quantity=-25000.0,
                origin="AUS_NEW",
                destination="IND_GVM",
                required_delivery_date=date(2026, 4, 25),
                number_of_voyages=1,
            )

    # 3. cargo exceeds all vessels
    def test_edge_case_3_cargo_exceeds_all_vessels(self, standard_vessels):
        """Single cargo quantity exceeding maximum bulk carrier capacity (e.g. 500k MT)."""
        selector = VesselSelector()
        port_origin = PortConstraints("AUS_NEW", "Newcastle", 20.0, 300.0, 50.0, 200_000)
        port_dest = PortConstraints("IND_GVM", "Gangavaram", 21.0, 300.0, 50.0, 200_000)

        res = selector.select_vessels(
            port_origin, port_dest, standard_vessels, cargo_tonnage=500_000.0
        )
        assert len(res.feasible) == 0
        assert len(res.excluded) == 4

    # 4. vessel draft exceeds port limit
    def test_edge_case_4_vessel_draft_exceeds_port_limit(self, standard_vessels):
        """Capesize with 18.5m draft entering 8.5m shallow port must fail with Draft violation."""
        selector = VesselSelector()
        capesize = standard_vessels[3]
        shallow_port = PortConstraints("IND_HLD", "Haldia", 8.5, 230.0, 32.2, 55_000)

        res = selector.check_compatibility(capesize, shallow_port)
        assert res.is_compatible is False
        assert any("Draft" in v for v in res.violations)

    # 5. missing port
    def test_edge_case_5_missing_port_graceful_handling(self):
        """Predicting congestion for an unregistered or unknown port ID must fallback safely."""
        predictor = CongestionPredictor()
        res = predictor.predict_congestion(
            port_id="UNKNOWN_PORT_XYZ",
            target_date=date(2026, 4, 15),
            vessel_class="Panamax",
        )
        assert res.expected_wait_days >= 0.0
        assert res.congestion_level in ["LOW", "MODERATE", "HIGH", "SEVERE"]

    # 6. missing route
    def test_edge_case_6_missing_route_fallback(self):
        """Forecaster for an unmapped route must provide safe baseline fallback."""
        forecaster = FreightForecaster()
        res = forecaster.predict_freight(
            origin="UNKNOWN_ORIGIN",
            destination="UNKNOWN_DEST",
            vessel_class="Panamax",
            horizon_days=7,
        )
        assert res["forecast_rate"] > 0.0
        assert res["lower_bound"] <= res["upper_bound"]

    # 7. missing freight history
    def test_edge_case_7_missing_freight_history(self):
        """Forecaster with non-standard route returns fallback rates with uncertainty spread."""
        forecaster = FreightForecaster()
        res = forecaster.predict_freight(
            origin="NO_PORT_A",
            destination="NO_PORT_B",
            vessel_class="Handysize",
            horizon_days=14,
        )
        assert res["forecast_rate"] > 0.0
        assert (res["upper_bound"] - res["lower_bound"]) > 0.0

    # 8. missing congestion history
    def test_edge_case_8_missing_congestion_history(self):
        """Port with no prior AIS or queue history returns baseline estimate."""
        predictor = CongestionPredictor()
        res = predictor.predict_congestion(
            port_id="NEW_PRIVATE_JETTY",
            target_date=None,
            vessel_class="Supramax",
        )
        assert res.expected_wait_days > 0.0
        assert 0.0 <= res.delay_probability <= 1.0

    # 9. extreme bunker price
    def test_edge_case_9_extreme_bunker_prices(self):
        """Bunker prices at $10,000/MT (extreme spike) and $0.01/MT (near zero)."""
        base_inputs = dict(
            cargo_quantity_t=75000.0,
            freight_rate_usd=22.0,
            vessel_speed_knots=13.0,
            vessel_daily_fuel_consumption_tpd=30.0,
            vessel_daily_hire_cost_usd=16000.0,
            route_distance_nm=3120.0,  # 10 days
        )

        # High extreme: $10,000 / t
        inputs_high = VoyageCostInputs(**base_inputs, fuel_price_usd_per_t=10000.0)
        res_high = calculate_voyage_cost(inputs_high)
        # 10 days * 30 tpd * 10,000 $/t = $3,000,000 bunker cost
        assert res_high.bunker_cost == pytest.approx(3000000.0, abs=100.0)
        assert not math.isnan(res_high.total_cost)
        assert not math.isinf(res_high.total_cost)

        # Low extreme: $0.01 / t
        inputs_low = VoyageCostInputs(**base_inputs, fuel_price_usd_per_t=0.01)
        res_low = calculate_voyage_cost(inputs_low)
        assert res_low.bunker_cost == pytest.approx(3.0, abs=1.0)
        assert res_low.total_cost > 0.0

    # 10. extreme congestion
    def test_edge_case_10_extreme_congestion_delay(self):
        """30 days port congestion waiting time should scale demurrage without crash."""
        inputs = VoyageCostInputs(
            cargo_quantity_t=80000.0,
            freight_rate_usd=20.0,
            vessel_speed_knots=12.5,
            vessel_daily_fuel_consumption_tpd=28.0,
            vessel_daily_hire_cost_usd=18000.0,
            route_distance_nm=3000.0,
            expected_waiting_days=30.0,
            laytime_allowed_days=6.0,
            daily_demurrage_rate_usd=25000.0,
        )
        res = calculate_voyage_cost(inputs)
        assert res.waiting_days == 30.0
        assert res.waiting_cost == pytest.approx(30.0 * 18000.0, abs=1e-2)
        assert res.excess_time_days >= 24.0
        assert res.demurrage_exposure >= 24.0 * 25000.0

    # 11. unavailable vessel
    def test_edge_case_11_unavailable_vessel_handling(self):
        """When regional availability is 0, availability risk score is elevated."""
        engine = MaritimeRiskEngine()
        inputs = {
            "vessel_data": {"available_vessels_in_region": 0, "lead_time_days": 20.0},
        }
        res = engine.evaluate_total_risk(inputs)
        assert "Vessel Availability" in res.categories
        assert res.categories["Vessel Availability"].score >= 70.0

    # 12. impossible delivery deadline
    def test_edge_case_12_impossible_delivery_deadline(self, standard_vessels):
        """Delivery deadline 2 days from now for a 14-day voyage."""
        prob = calculate_delivery_probability(total_elapsed_days=14.0, delivery_deadline_days=2.0)
        assert prob <= 0.05

        optimizer = VesselOptimizer()
        origin = PortConstraints("AUS_NEW", "Newcastle", 20.0, 300.0, 50.0, 200_000)
        dest = PortConstraints("IND_GVM", "Gangavaram", 21.0, 300.0, 50.0, 200_000)
        now = datetime.now()

        res = optimizer.optimize(
            cargo_type="Coal",
            cargo_quantity=82000.0,
            origin_port=origin,
            destination_port=dest,
            expected_loading_date=now,
            required_delivery_date=now + timedelta(days=2),
            vessel_specs=standard_vessels,
        )
        assert res.recommended_vessel == "None"
        assert res.score == 0.0

    # 13. no feasible vessel plan
    def test_edge_case_13_no_feasible_vessel_plan(self, standard_vessels):
        """180,000 MT Capesize parcel sent to a shallow port where Capesize cannot berth."""
        optimizer = VesselOptimizer()
        origin = PortConstraints("AUS_NEW", "Newcastle", 20.0, 300.0, 50.0, 200_000)
        shallow_dest = PortConstraints("IND_HLD", "Haldia", 8.5, 230.0, 32.2, 55_000)
        now = datetime.now()

        res = optimizer.optimize(
            cargo_type="Coal",
            cargo_quantity=180000.0,
            origin_port=origin,
            destination_port=shallow_dest,
            expected_loading_date=now,
            required_delivery_date=now + timedelta(days=30),
            vessel_specs=standard_vessels,
        )
        assert res.recommended_vessel == "None"
        assert res.score == 0.0
        assert len(res.alternatives) > 0
