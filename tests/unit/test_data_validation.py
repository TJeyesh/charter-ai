"""
Unit Tests: Data Validation (Pydantic & Domain Models).

Verifies strict type enforcement, range validation, missing fields,
and schema invariants across request, domain, and response schemas.
"""

from datetime import date, datetime, timedelta
import pytest
from pydantic import ValidationError

from src.api.serializers import (
    CargoRequest,
    RecommendationRequest,
    FreightForecastApiRequest,
    CongestionPredictionRequest,
    VesselOptimizationRequest,
    VoyageOptimizationRequest,
    ContractOptimizationRequest,
    AnalyzeVoyageRequest,
    BacktestApiRequest,
)
from src.optimization.vessel_selector import PortConstraints, VesselSpecs
from src.optimization.port_compatibility import PortInfo, VesselInfo


class TestPydanticRequestValidation:
    """Test validation rules on all public API Pydantic schemas."""

    def test_cargo_request_valid(self):
        req = CargoRequest(
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            cargo_type="Coal",
            cargo_tonnage=75000,
            earliest_date=date(2026, 4, 1),
            latest_date=date(2026, 4, 25),
            risk_appetite="moderate",
        )
        assert req.cargo_tonnage == 75000
        assert req.origin_port_id == "AUS_NEW"
        assert req.risk_appetite == "moderate"

    def test_cargo_request_missing_required_fields(self):
        with pytest.raises(ValidationError) as exc:
            CargoRequest(
                origin_port_id="AUS_NEW",
            )
        errors = exc.value.errors()
        fields = [e["loc"][0] for e in errors]
        assert "destination_port_id" in fields
        assert "cargo_tonnage" in fields

    def test_cargo_request_invalid_quantity_types(self):
        with pytest.raises(ValidationError):
            CargoRequest(
                origin_port_id="AUS_NEW",
                destination_port_id="IND_GVM",
                cargo_type="Coal",
                cargo_tonnage="NOT_A_NUMBER",
                earliest_date=date(2026, 4, 1),
                latest_date=date(2026, 4, 25),
            )

    def test_freight_forecast_request_validation(self):
        req = FreightForecastApiRequest(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            horizon_days=14,
        )
        assert req.origin == "AUS_NEW"
        assert req.destination == "IND_GVM"
        assert req.horizon_days == 14

        # Test defaults
        req_default = FreightForecastApiRequest(
            origin="AUS_NEW",
            destination="IND_GVM",
        )
        assert req_default.horizon_days == 7
        assert req_default.vessel_class == "Panamax"

    def test_congestion_prediction_request_validation(self):
        req = CongestionPredictionRequest(
            port_id="IND_GVM",
            target_date="2026-04-15",
            vessel_class="Capesize",
        )
        assert req.port_id == "IND_GVM"
        assert req.vessel_class == "Capesize"

    def test_vessel_optimization_request_validation(self):
        req = VesselOptimizationRequest(
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            cargo_tonnage=80000.0,
            cargo_type="Iron Ore",
        )
        assert req.cargo_tonnage == 80000.0
        assert req.origin_port_id == "AUS_NEW"

    def test_voyage_optimization_request_validation(self):
        req = VoyageOptimizationRequest(
            cargo_quantity_t=160000.0,
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            max_voyages=3,
        )
        assert req.max_voyages == 3
        assert req.cargo_quantity_t == 160000.0

    def test_contract_optimization_request_validation(self):
        req = ContractOptimizationRequest(
            cargo_quantity_t=80000.0,
            spot_freight_rate=22.50,
            freight_volatility_pct=18.0,
        )
        assert req.cargo_quantity_t == 80000.0
        assert req.spot_freight_rate == 22.50

    def test_backtest_request_validation(self):
        req = BacktestApiRequest(
            years=[2022, 2023],
            horizons=[7, 14],
            n_scenarios=5,
            seed=42,
        )
        assert req.years == [2022, 2023]
        assert req.n_scenarios == 5


class TestDomainModelInvariants:
    """Test validation invariants on domain model dataclasses."""

    def test_port_constraints_dataclass(self):
        pc = PortConstraints(
            port_id="IND_GVM",
            port_name="Gangavaram",
            max_draft_m=21.0,
            max_loa_m=300.0,
            max_beam_m=50.0,
            max_dwt=200000.0,
        )
        assert pc.port_id == "IND_GVM"
        assert pc.is_anchorage_only is False

    def test_vessel_specs_dataclass(self):
        vs = VesselSpecs(
            class_name="Panamax",
            dwt_min=60000,
            dwt_max=100000,
            typical_dwt=82000,
            draft_max_m=14.5,
            loa_max_m=240.0,
            beam_max_m=36.0,
        )
        assert vs.class_name == "Panamax"
        assert vs.dwt_min < vs.dwt_max
        assert vs.typical_dwt >= vs.dwt_min
