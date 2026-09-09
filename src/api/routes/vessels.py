"""
Charter-AI — Vessels Endpoint.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import VesselClassResponse, VesselSelectionResponse, VesselCompatibilityResponse
from src.data.repository import PortRepository, VesselClassRepository
from src.optimization.vessel_selector import PortConstraints, VesselSelector, VesselSpecs

router = APIRouter(prefix="/vessels", tags=["Vessels"])

_selector = VesselSelector()


@router.get("", response_model=List[VesselClassResponse])
async def list_vessel_classes(
    db: AsyncSession = Depends(get_session),
):
    """List all vessel classes with their specifications."""
    repo = VesselClassRepository(db)
    classes = await repo.get_all()
    return [
        VesselClassResponse(
            class_name=vc.class_name,
            dwt_min=vc.dwt_min,
            dwt_max=vc.dwt_max,
            typical_dwt=vc.typical_dwt,
            draft_max_m=vc.draft_max_m,
            loa_max_m=vc.loa_max_m,
            beam_max_m=vc.beam_max_m,
        )
        for vc in classes
    ]


@router.get("/compatibility", response_model=VesselSelectionResponse)
async def check_vessel_compatibility(
    origin_port_id: str,
    destination_port_id: str,
    cargo_tonnage: Optional[int] = None,
    db: AsyncSession = Depends(get_session),
):
    """
    Check which vessel classes are compatible with an origin-destination pair.

    Applies hard physical constraints: draft, LOA, beam.
    """
    port_repo = PortRepository(db)
    vessel_repo = VesselClassRepository(db)

    origin = await port_repo.get_by_id(origin_port_id)
    destination = await port_repo.get_by_id(destination_port_id)
    all_vessels = await vessel_repo.get_all()

    if not origin or not destination:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Origin or destination port not found")

    origin_constraints = PortConstraints(
        port_id=origin.port_id,
        port_name=origin.port_name,
        max_draft_m=origin.max_draft_m or 0,
        max_loa_m=origin.max_loa_m or 0,
        max_beam_m=origin.max_beam_m or 0,
        max_dwt=origin.max_dwt or 0,
        is_anchorage_only=(origin.max_draft_m == -9999),
    )
    dest_constraints = PortConstraints(
        port_id=destination.port_id,
        port_name=destination.port_name,
        max_draft_m=destination.max_draft_m or 0,
        max_loa_m=destination.max_loa_m or 0,
        max_beam_m=destination.max_beam_m or 0,
        max_dwt=destination.max_dwt or 0,
        is_anchorage_only=(destination.max_draft_m == -9999),
    )

    vessel_specs = [
        VesselSpecs(
            class_name=vc.class_name,
            dwt_min=vc.dwt_min,
            dwt_max=vc.dwt_max,
            typical_dwt=vc.typical_dwt,
            draft_max_m=vc.draft_max_m,
            loa_max_m=vc.loa_max_m,
            beam_max_m=vc.beam_max_m,
        )
        for vc in all_vessels
    ]

    result = _selector.select_vessels(
        origin_port=origin_constraints,
        destination_port=dest_constraints,
        vessel_specs=vessel_specs,
        cargo_tonnage=cargo_tonnage,
    )

    return VesselSelectionResponse(
        origin_port_id=result.origin_port_id,
        destination_port_id=result.destination_port_id,
        feasible=[
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.feasible
        ],
        excluded=[
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.excluded
        ],
    )
