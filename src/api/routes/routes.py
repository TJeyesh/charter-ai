"""
Charter-AI — Routes (Shipping Routes) Endpoint (Phase 14).

Provides catalog of dry-bulk shipping routes, nautical distances, and transit durations
with resilient fallback to processed CSV data.
"""

from typing import List, Optional
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import RouteResponse, RoutesListResponse
from src.data.repository import RouteRepository
from src.data.mock_db import get_mock_port_info
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/routes", tags=["Routes"])

_ROUTES_CSV = Path("data/processed/routes.csv")


def _load_routes_from_csv(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> List[RouteResponse]:
    """Load shipping routes from processed CSV dataset."""
    routes = []
    if _ROUTES_CSV.exists():
        try:
            df = pd.read_csv(_ROUTES_CSV)
            if origin:
                df = df[df["origin_port"].str.upper() == origin.upper()]
            if destination:
                df = df[df["destination_port"].str.upper() == destination.upper()]

            for _, row in df.iterrows():
                orig_id = str(row["origin_port"])
                dest_id = str(row["destination_port"])
                orig_info = get_mock_port_info(orig_id)
                dest_info = get_mock_port_info(dest_id)
                dist = float(row["distance_nm"])
                routes.append(
                    RouteResponse(
                        origin_port_id=orig_id,
                        origin_port_name=orig_info.port_name if orig_info else orig_id,
                        origin_country=orig_id.split("_")[0],
                        destination_port_id=dest_id,
                        destination_port_name=dest_info.port_name if dest_info else dest_id,
                        great_circle_nm=dist * 0.95,
                        est_sailing_distance_nm=dist,
                        typical_cargo="Thermal Coal",
                        routing_note=f"Type: {row.get('route_type', 'direct')}",
                    )
                )
            return routes
        except Exception as e:
            logger.warning("Error reading %s: %s", _ROUTES_CSV, e)

    # Synthetic fallback standard routes
    standard_routes = [
        ("AUS_NEW", "IND_VZG", 5450.0),
        ("AUS_NEW", "IND_PAR", 5620.0),
        ("AUS_NEW", "IND_GVM", 5440.0),
        ("IDN_TAB", "IND_GVM", 2650.0),
        ("IDN_TAB", "IND_PAR", 2720.0),
        ("ZAF_RIC", "IND_VZG", 4650.0),
    ]
    for orig_id, dest_id, dist in standard_routes:
        if origin and orig_id.upper() != origin.upper():
            continue
        if destination and dest_id.upper() != destination.upper():
            continue
        orig_info = get_mock_port_info(orig_id)
        dest_info = get_mock_port_info(dest_id)
        routes.append(
            RouteResponse(
                origin_port_id=orig_id,
                origin_port_name=orig_info.port_name,
                origin_country=orig_id.split("_")[0],
                destination_port_id=dest_id,
                destination_port_name=dest_info.port_name,
                great_circle_nm=dist * 0.95,
                est_sailing_distance_nm=dist,
                typical_cargo="Thermal Coal",
                routing_note="Standard benchmark corridor",
            )
        )
    return routes


@router.get("", response_model=RoutesListResponse)
async def list_routes(
    origin: Optional[str] = Query(None, description="Filter by origin port ID, e.g. AUS_NEW"),
    destination: Optional[str] = Query(None, description="Filter by destination port ID, e.g. IND_GVM"),
    db: Optional[AsyncSession] = Depends(get_session),
) -> RoutesListResponse:
    """
    GET /api/v1/routes
    
    List shipping routes with nautical distances, optionally filtered by origin/destination.
    """
    routes = []
    if db is not None:
        try:
            repo = RouteRepository(db)
            db_routes = await repo.find(origin_port_id=origin, destination_port_id=destination)
            routes = [
                RouteResponse(
                    origin_port_id=r.origin_port_id,
                    origin_port_name=r.origin_port_name,
                    origin_country=r.origin_country,
                    destination_port_id=r.destination_port_id,
                    destination_port_name=r.destination_port_name,
                    great_circle_nm=r.great_circle_nm,
                    est_sailing_distance_nm=r.est_sailing_distance_nm,
                    typical_cargo=r.typical_cargo,
                    routing_note=r.routing_note,
                )
                for r in db_routes
            ]
        except Exception as e:
            logger.debug("Database error querying routes (%s), falling back to CSV", e)

    if not routes:
        routes = _load_routes_from_csv(origin=origin, destination=destination)

    return RoutesListResponse(
        total_count=len(routes),
        routes=routes,
    )
