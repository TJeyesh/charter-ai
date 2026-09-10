"""
Unit Tests: Voyage Economics and Multi-Voyage Optimization.

Verifies the 9-component delivered cost model, cost per tonne metric,
candidate plan parcel generation, and fleet multi-voyage optimization.
"""

from datetime import datetime, timedelta
import pytest

from src.economics.voyage_cost import (
    VoyageCostInputs,
    VoyageCostBreakdown,
    calculate_voyage_cost,
)
from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.multi_voyage_optimizer import (
    MultiVoyageOptimizer,
    FleetOptimizationRequest,
)


class TestDeliveredVoyageEconomics:
    """Test full 9-component voyage economics calculation."""

    def test_complete_9_cost_components(self):
        inputs = VoyageCostInputs(
            cargo_quantity_t=75000.0,
            freight_rate_usd=22.0,
            vessel_speed_knots=13.0,
            vessel_daily_fuel_consumption_tpd=28.0,
            vessel_daily_hire_cost_usd=17000.0,
            route_distance_nm=3120.0,  # 3120 / (13*24) = 10.0 days
            fuel_price_usd_per_t=650.0,
            load_port_cost_usd=40000.0,
            discharge_port_cost_usd=50000.0,
            canal_charges_usd=15000.0,
            expected_waiting_days=2.0,
            laytime_allowed_days=6.0,
            daily_demurrage_rate_usd=20000.0,
            other_costs_usd=5000.0,
        )
        res = calculate_voyage_cost(inputs)

        # Freight: 75,000 * 22 = $1,650,000
        assert res.freight_cost == 1650000.0

        # Bunker: 10.0 days * 28 tpd * 650 $/t = $182,000
        assert res.bunker_cost == pytest.approx(182000.0, abs=10.0)

        # Port costs: 40,000 + 50,000 = $90,000
        assert res.port_cost == 90000.0

        # Miscellaneous costs (including canal charges: 15,000 + other: 5,000 = 20,000)
        assert res.miscellaneous_cost == 20000.0

        # Waiting: 2.0 days * 17,000 = $34,000
        assert res.waiting_cost == 34000.0

        # Total cost is sum of components
        expected_total = (
            res.freight_cost
            + res.bunker_cost
            + res.port_cost
            + res.waiting_cost
            + res.demurrage_exposure
            + res.miscellaneous_cost
            + res.positioning_cost
            - res.despatch_savings
        )
        assert res.total_cost == pytest.approx(expected_total, abs=1e-2)

        # Cost per tonne (rounded to 2 decimal places)
        assert res.cost_per_tonne == pytest.approx(res.total_cost / 75000.0, abs=0.01)


class TestMultiVoyageOptimization:
    """Test candidate parcel splitting and fleet optimization."""

    def test_parcel_generation_sums_to_total_cargo(self):
        generator = CandidateGenerator()
        total_cargo = 120_000.0
        plans = generator.generate_candidate_plans(total_cargo_t=total_cargo, max_voyages=4)

        assert len(plans) >= 3
        for plan in plans:
            # Each plan's allocated parcels must sum precisely to the total requested cargo
            assert sum(plan.cargo_per_voyage) == pytest.approx(total_cargo, abs=1.0)
            assert len(plan.cargo_per_voyage) == plan.number_of_voyages

    def test_multi_voyage_optimizer_solves_valid_request(self):
        optimizer = MultiVoyageOptimizer()
        req = FleetOptimizationRequest(
            cargo_quantity_t=100000.0,
            cargo_type="Coal",
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            expected_loading_date=datetime(2026, 4, 1),
            required_delivery_date=datetime(2026, 4, 30),
            fuel_price_usd_per_t=650.0,
        )
        plans = optimizer.optimize(req)

        assert len(plans) >= 1
        best_plan = plans[0]
        assert best_plan.total_cargo_t == pytest.approx(100000.0, abs=1.0)
        assert best_plan.cost_per_tonne > 0.0
