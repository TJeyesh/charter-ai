"""
Charter-AI — Mock Database
Synthetic data used for SIH Demo Mode.
"""

from src.optimization.vessel_selector import VesselSpecs
from src.optimization.port_compatibility import PortInfo
from src.utils.config import get_settings

def get_mock_vessel_db() -> list[VesselSpecs]:
    """Returns a synthetic list of vessel classes."""
    return [
        VesselSpecs(class_name="Handysize", dwt_max=40000, dwt_min=15000, typical_dwt=30000, draft_max_m=10.0, loa_max_m=180.0, beam_max_m=28.0),
        VesselSpecs(class_name="Supramax", dwt_max=60000, dwt_min=40000, typical_dwt=55000, draft_max_m=12.0, loa_max_m=200.0, beam_max_m=32.0),
        VesselSpecs(class_name="Panamax", dwt_max=80000, dwt_min=60000, typical_dwt=75000, draft_max_m=14.0, loa_max_m=225.0, beam_max_m=32.2),
        VesselSpecs(class_name="Capesize", dwt_max=200000, dwt_min=100000, typical_dwt=150000, draft_max_m=18.0, loa_max_m=300.0, beam_max_m=45.0)
    ]

def get_mock_port_info(port_id: str) -> PortInfo:
    """Returns synthetic port infrastructure constraints."""
    return PortInfo(
        port_id=port_id, 
        port_name=port_id.title(), 
        max_draft_m=20.0, 
        max_loa_m=350.0, 
        max_beam_m=50.0, 
        cargo_handling_rate_tpd=20000, 
        berthing_capacity=5, 
        current_congestion_factor=1.0
    )

def require_sih_demo_mode():
    """Raises an error if trying to access mock data when not in demo mode."""
    if not get_settings().sih_demo_mode:
        raise ValueError("System is not in SIH Demo Mode. Live database connection required.")
