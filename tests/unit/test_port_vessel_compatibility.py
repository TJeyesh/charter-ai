"""
Unit Tests: Port and Vessel Compatibility Engine.

Verifies hard physical constraints (draft, LOA, beam, DWT, air draft, anchorage),
operational warnings, and vessel class suitability scoring.
"""

import pytest

from src.optimization.port_compatibility import (
    PortInfo,
    VesselInfo,
    check_vessel_port_compatibility,
)
from src.optimization.vessel_selector import (
    PortConstraints,
    VesselSelector,
    VesselSpecs,
)


@pytest.fixture
def standard_vessel_specs():
    return [
        VesselSpecs("Handysize", 10_000, 40_000, 32_000, 11.0, 190.0, 30.0),
        VesselSpecs("Supramax", 40_000, 60_000, 56_000, 13.0, 200.0, 32.5),
        VesselSpecs("Panamax", 60_000, 100_000, 82_000, 14.5, 240.0, 36.0),
        VesselSpecs("Capesize", 120_000, 220_000, 180_000, 18.5, 300.0, 50.0),
    ]


@pytest.fixture
def deep_port():
    return PortConstraints(
        port_id="IND_GVM",
        port_name="Gangavaram",
        max_draft_m=21.0,
        max_loa_m=300.0,
        max_beam_m=50.0,
        max_dwt=200_000,
    )


@pytest.fixture
def shallow_port():
    return PortConstraints(
        port_id="IND_HLD",
        port_name="Haldia",
        max_draft_m=8.5,
        max_loa_m=230.0,
        max_beam_m=32.2,
        max_dwt=55_000,
    )


@pytest.fixture
def anchorage_only_port():
    return PortConstraints(
        port_id="IND_SAG",
        port_name="Sagar Anchorage",
        max_draft_m=-9999,
        max_loa_m=-9999,
        max_beam_m=-9999,
        max_dwt=-9999,
        is_anchorage_only=True,
    )


class TestPhysicalPortConstraints:
    """Test deterministic check_compatibility against physical limits."""

    def test_capesize_draft_exceeds_shallow_port(self, standard_vessel_specs, shallow_port):
        selector = VesselSelector()
        capesize = standard_vessel_specs[3]
        res = selector.check_compatibility(capesize, shallow_port)

        assert res.is_compatible is False
        assert any("Draft" in v for v in res.violations)

    def test_panamax_fits_deep_port(self, standard_vessel_specs, deep_port):
        selector = VesselSelector()
        panamax = standard_vessel_specs[2]
        res = selector.check_compatibility(panamax, deep_port)

        assert res.is_compatible is True
        assert len(res.violations) == 0

    def test_anchorage_rejects_berthing(self, standard_vessel_specs, anchorage_only_port):
        selector = VesselSelector()
        for v in standard_vessel_specs:
            res = selector.check_compatibility(v, anchorage_only_port)
            assert res.is_compatible is False
            assert "anchorage" in res.violations[0].lower()

    def test_beam_constraint_violation(self, standard_vessel_specs):
        selector = VesselSelector()
        narrow_lock_port = PortConstraints(
            port_id="PAN_LOCK",
            port_name="Panama Canal Lock Old",
            max_draft_m=15.0,
            max_loa_m=290.0,
            max_beam_m=32.31,  # Panamax beam max
            max_dwt=80_000,
        )
        capesize = standard_vessel_specs[3]  # beam 50m
        res = selector.check_compatibility(capesize, narrow_lock_port)
        assert res.is_compatible is False
        assert any("Beam" in v for v in res.violations)


class TestPortCompatibilityScoring:
    """Test compatibility scoring and operational warnings."""

    def test_detailed_port_compatibility_scoring(self):
        port = PortInfo(
            port_id="IND_GVM",
            port_name="Gangavaram",
            max_draft_m=20.0,
            max_loa_m=300.0,
            max_beam_m=50.0,
            cargo_handling_rate_tpd=40000.0,
            berthing_capacity=5,
            current_congestion_factor=1.1,
        )
        vessel = VesselInfo(
            class_name="Panamax",
            draft_m=14.0,
            loa_m=235.0,
            beam_m=32.2,
            cargo_to_handle_t=75000.0,
        )
        res = check_vessel_port_compatibility(vessel, port)
        assert res["compatible"] is True
        assert res["score"] >= 85.0
        assert len(res["failed_constraints"]) == 0

    def test_slow_turnaround_warning_generated(self):
        port = PortInfo(
            port_id="IND_SLOW",
            port_name="SlowPort",
            max_draft_m=18.0,
            max_loa_m=300.0,
            max_beam_m=50.0,
            cargo_handling_rate_tpd=5000.0,  # Very slow
            berthing_capacity=2,
            current_congestion_factor=1.0,
        )
        vessel = VesselInfo(
            class_name="Capesize",
            draft_m=17.0,
            loa_m=290.0,
            beam_m=45.0,
            cargo_to_handle_t=160000.0,  # 32 days handling!
        )
        res = check_vessel_port_compatibility(vessel, port)
        assert res["compatible"] is True
        assert res["score"] < 100.0
        assert any("Slow turnaround warning" in w for w in res["warnings"])
