"""
Charter-AI — Ports Routes (Phase 14).

Provides catalog of loading and discharge dry-bulk ports, infrastructure constraints,
and current port congestion status with resilient fallback to processed dataset.
"""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from src.api.deps import get_session
from src.api.serializers import (
    PortResponse,
    PortCongestionResponse,
    PortsListResponse,
)
from src.data.repository import CongestionRepository, PortRepository
from src.data.mock_db import get_mock_ports
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ports", tags=["Ports"])

_PORTS_CSV = Path("data/processed/ports.csv")


def _load_ports_from_csv(country: Optional[str] = None) -> List[PortResponse]:
    """Load port catalog from processed CSV file or mock DB."""
    if _PORTS_CSV.exists():
        try:
            df = pd.read_csv(_PORTS_CSV)
            if country:
                df = df[df["country"].str.upper() == country.upper()]
            ports = []
            for _, row in df.iterrows():
                ports.append(
                    PortResponse(
                        port_id=str(row["port_id"]),
                        port_name=str(row["port_name"]),
                        state=str(row.get("state", row["country"])),
                        country=str(row["country"]),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        port_type="Sea Port",
                        operator="Port Authority",
                        berths_total=int(row["berth_count"]) if pd.notna(row.get("berth_count")) else None,
                        max_draft_m=float(row["max_draft_m"]) if pd.notna(row.get("max_draft_m")) else None,
                        max_loa_m=float(row["max_loa_m"]) if pd.notna(row.get("max_loa_m")) else None,
                        max_beam_m=float(row["max_beam_m"]) if pd.notna(row.get("max_beam_m")) else None,
                        max_dwt=int(float(row["max_draft_m"]) * 10000) if pd.notna(row.get("max_draft_m")) else None,
                        primary_cargo="Thermal Coal" if row.get("coal_terminal") else "Dry Bulk",
                    )
                )
            return ports
        except Exception as e:
            logger.warning("Error reading %s: %s", _PORTS_CSV, e)

    # Fallback to mock ports
    mock_ports = get_mock_ports()
    if country:
        mock_ports = [p for p in mock_ports if p.get("country", "").upper() == country.upper()]
    return [
        PortResponse(
            port_id=p["port_id"],
            port_name=p["port_name"],
            state=p.get("state", p.get("country", "IND")),
            country=p.get("country", "IND"),
            latitude=p["latitude"],
            longitude=p["longitude"],
            port_type="Sea Port",
            operator="Port Authority",
            berths_total=p.get("berth_count"),
            max_draft_m=p.get("max_draft_m"),
            max_loa_m=p.get("max_loa_m"),
            max_beam_m=p.get("max_beam_m"),
            max_dwt=int(p.get("max_draft_m", 15.0) * 10000),
            primary_cargo="Thermal Coal" if p.get("coal_terminal") else "Dry Bulk",
        )
        for p in mock_ports
    ]


@router.get("", response_model=PortsListResponse)
async def list_ports(
    country: Optional[str] = Query(None, description="Optional filter by ISO country code, e.g. IND, AUS, IDN"),
    db: Optional[AsyncSession] = Depends(get_session),
) -> PortsListResponse:
    """
    GET /api/v1/ports
    
    List all ports with infrastructure dimensions (draft, LOA, beam, berths).
    Falls back gracefully to processed dataset when database session is disconnected.
    """
    ports = []
    if db is not None:
        try:
            repo = PortRepository(db)
            db_ports = await repo.get_all(country=country)
            ports = [
                PortResponse(
                    port_id=p.port_id,
                    port_name=p.port_name,
                    state=p.state or p.country,
                    country=p.country,
                    latitude=p.latitude,
                    longitude=p.longitude,
                    port_type=p.port_type or "Sea Port",
                    operator=p.operator or "Port Authority",
                    berths_total=p.berths_total,
                    max_draft_m=p.max_draft_m,
                    max_loa_m=p.max_loa_m,
                    max_beam_m=p.max_beam_m,
                    max_dwt=p.max_dwt,
                    annual_capacity_mtpa=p.annual_capacity_mtpa,
                    primary_cargo=p.primary_cargo or "Dry Bulk",
                )
                for p in db_ports
            ]
        except Exception as e:
            logger.debug("Database error querying ports (%s), falling back to CSV", e)

    if not ports:
        ports = _load_ports_from_csv(country=country)

    return PortsListResponse(
        total_count=len(ports),
        ports=ports,
    )


@router.get("/{port_id}", response_model=PortResponse)
async def get_port(
    port_id: str,
    db: Optional[AsyncSession] = Depends(get_session),
) -> PortResponse:
    """
    GET /api/v1/ports/{port_id}
    
    Get infrastructure constraints and specifications for a specific port.
    """
    if db is not None:
        try:
            repo = PortRepository(db)
            p = await repo.get_by_id(port_id)
            if p:
                return PortResponse(
                    port_id=p.port_id,
                    port_name=p.port_name,
                    state=p.state or p.country,
                    country=p.country,
                    latitude=p.latitude,
                    longitude=p.longitude,
                    port_type=p.port_type or "Sea Port",
                    operator=p.operator or "Port Authority",
                    berths_total=p.berths_total,
                    max_draft_m=p.max_draft_m,
                    max_loa_m=p.max_loa_m,
                    max_beam_m=p.max_beam_m,
                    max_dwt=p.max_dwt,
                    annual_capacity_mtpa=p.annual_capacity_mtpa,
                    primary_cargo=p.primary_cargo or "Dry Bulk",
                )
        except Exception as e:
            logger.debug("Database error querying port %s: %s", port_id, e)

    # Fallback lookup from CSV
    all_ports = _load_ports_from_csv()
    for port in all_ports:
        if port.port_id.upper() == port_id.upper():
            return port

    raise HTTPException(status_code=404, detail=f"Port {port_id} not found")


@router.get("/{port_id}/congestion", response_model=Optional[PortCongestionResponse])
async def get_port_congestion(
    port_id: str,
    db: Optional[AsyncSession] = Depends(get_session),
) -> PortCongestionResponse:
    """
    GET /api/v1/ports/{port_id}/congestion
    
    Get the latest observed or predicted congestion data for a port.
    """
    if db is not None:
        try:
            repo = CongestionRepository(db)
            congestion = await repo.get_latest(port_id)
            if congestion:
                return PortCongestionResponse(
                    port_id=congestion.port_id,
                    date=congestion.date,
                    vessels_waiting=congestion.vessels_waiting,
                    avg_waiting_time_days=congestion.avg_waiting_time_days,
                    berth_occupancy_pct=congestion.berth_occupancy_pct,
                )
        except Exception as e:
            logger.debug("Database error querying congestion for %s: %s", port_id, e)

    # Fallback from congestion service or realistic estimate
    from src.services.congestion_service import CongestionService
    from datetime import date
    try:
        service = CongestionService()
        pred = service.predict_congestion(port_id=port_id)
        return PortCongestionResponse(
            port_id=port_id,
            date=date.today(),
            vessels_waiting=int(pred.get("expected_wait_days", 2.0) * 2.5),
            avg_waiting_time_days=pred.get("expected_wait_days", 2.0),
            berth_occupancy_pct=75.0,
        )
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"No congestion data available for port {port_id}"
        )
