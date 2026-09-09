"""
Charter-AI — Data Ingestion Pipeline.

Loads CSV datasets into PostgreSQL. Designed to be idempotent —
safe to re-run with updated data files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.data.db import Base, get_sync_engine, get_sync_session_factory
from src.data.models import (
    Commodity,
    Congestion,
    EconomicIndicator,
    Event,
    FreightRate,
    Port,
    Route,
    VesselClassModel,
    Weather,
)
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _ensure_postgis(engine) -> None:
    """Enable PostGIS extension if not already enabled."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()


def _load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV with basic cleaning."""
    logger.info(f"Loading {filepath.name} ({filepath.stat().st_size:,} bytes)")
    df = pd.read_csv(filepath, encoding="utf-8")
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


def ingest_ports(session: Session, data_dir: Path) -> int:
    """Load ports.csv into the ports table."""
    df = _load_csv(data_dir / "ports.csv")
    count = 0
    for _, row in df.iterrows():
        port = Port(
            port_id=row["port_id"],
            port_name=row["port_name"],
            state=row["state"],
            country=row["country"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            port_type=row["port_type"],
            operator=row["operator"],
            berths_total=int(row["berths_total"]) if pd.notna(row["berths_total"]) else None,
            max_draft_m=float(row["max_draft_m"]) if pd.notna(row["max_draft_m"]) else None,
            max_loa_m=float(row["max_loa_m"]) if pd.notna(row["max_loa_m"]) else None,
            max_beam_m=float(row["max_beam_m"]) if pd.notna(row["max_beam_m"]) else None,
            max_dwt=int(row["max_dwt_capesize_capable"]) if pd.notna(row.get("max_dwt_capesize_capable")) else None,
            annual_capacity_mtpa=str(row.get("annual_capacity_mtpa", "")),
            primary_cargo=str(row.get("primary_cargo", "")),
            data_type=row["data_type"],
            source_note=str(row.get("source_note", "")),
        )
        session.merge(port)  # Upsert by primary key
        count += 1

    # Set PostGIS geometry from lat/lon
    session.flush()
    session.execute(
        text(
            "UPDATE ports SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL"
        )
    )
    session.commit()
    logger.info(f"Ingested {count} ports")
    return count


def ingest_vessel_classes(session: Session, data_dir: Path) -> int:
    """Load vessels.csv into the vessel_classes table."""
    df = _load_csv(data_dir / "vessels.csv")
    count = 0
    for _, row in df.iterrows():
        vc = VesselClassModel(
            class_id=row["vessel_class"].replace(" ", "_").replace("/", "_"),
            class_name=row["vessel_class"],
            dwt_min=int(row["dwt_min"]),
            dwt_max=int(row["dwt_max"]),
            typical_dwt=int(row["typical_dwt"]),
            loa_min_m=float(row["loa_min_m"]),
            loa_max_m=float(row["loa_max_m"]),
            beam_min_m=float(row["beam_min_m"]),
            beam_max_m=float(row["beam_max_m"]),
            draft_min_m=float(row["draft_min_m"]),
            draft_max_m=float(row["draft_max_m"]),
            typical_cargo=str(row.get("typical_cargo", "")),
            data_type=row["data_type"],
            source_note=str(row.get("source_note", "")),
        )
        session.merge(vc)
        count += 1
    session.commit()
    logger.info(f"Ingested {count} vessel classes")
    return count


def ingest_freight_rates(session: Session, data_dir: Path) -> int:
    """Bulk-load freight_rates.csv using pandas to_sql for performance."""
    df = _load_csv(data_dir / "freight_rates.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    engine = session.get_bind()
    # Use pandas to_sql with "append" for bulk insert
    df.to_sql("freight_rates", engine, if_exists="append", index=False, method="multi")
    count = len(df)
    logger.info(f"Ingested {count:,} freight rate records")
    return count


def ingest_all(data_dir: Optional[str] = None) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        data_dir: Path to the raw data directory. Defaults to settings.raw_data_dir.

    Returns:
        Dictionary of {table_name: records_loaded} counts.
    """
    settings = get_settings()
    data_path = Path(data_dir) if data_dir else Path(settings.raw_data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    engine = get_sync_engine()
    _ensure_postgis(engine)
    Base.metadata.create_all(engine)

    factory = get_sync_session_factory()
    results = {}

    with factory() as session:
        # Order matters: reference tables first, then transactional data
        if (data_path / "ports.csv").exists():
            results["ports"] = ingest_ports(session, data_path)
        if (data_path / "vessels.csv").exists():
            results["vessel_classes"] = ingest_vessel_classes(session, data_path)
        if (data_path / "freight_rates.csv").exists():
            results["freight_rates"] = ingest_freight_rates(session, data_path)

        # Remaining tables follow the same pattern (stubs for now)
        logger.info(f"Ingestion complete: {results}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging(level="INFO", fmt="text")
    results = ingest_all()
    print(f"Ingestion results: {results}")
