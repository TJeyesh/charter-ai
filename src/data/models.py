"""
Charter-AI — SQLAlchemy ORM Models.

Maps to the database entities defined in the architecture (Section 5).
Uses PostGIS geometry for port locations.
"""

from datetime import date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import BigInteger

from src.data.db import Base


class Port(Base):
    """Port entity with PostGIS geometry for map queries."""

    __tablename__ = "ports"

    port_id = Column(String(20), primary_key=True)
    port_name = Column(String(100), nullable=False)
    state = Column(String(100))
    country = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geom = Column(Geometry("POINT", srid=4326), nullable=True)
    port_type = Column(String(100))
    operator = Column(String(200))
    berths_total = Column(Integer)
    max_draft_m = Column(Float)
    max_loa_m = Column(Float)
    max_beam_m = Column(Float)
    max_dwt = Column(Integer)
    annual_capacity_mtpa = Column(String(50))
    primary_cargo = Column(Text)
    data_type = Column(String(50))
    source_note = Column(Text)


class VesselClassModel(Base):
    """Vessel class specifications (Handysize through VLOC)."""

    __tablename__ = "vessel_classes"

    class_id = Column(String(30), primary_key=True)
    class_name = Column(String(50), nullable=False)
    dwt_min = Column(Integer, nullable=False)
    dwt_max = Column(Integer, nullable=False)
    typical_dwt = Column(Integer, nullable=False)
    loa_min_m = Column(Float)
    loa_max_m = Column(Float)
    beam_min_m = Column(Float)
    beam_max_m = Column(Float)
    draft_min_m = Column(Float)
    draft_max_m = Column(Float)
    typical_cargo = Column(Text)
    data_type = Column(String(50))
    source_note = Column(Text)


class Route(Base):
    """Shipping route between origin and destination port."""

    __tablename__ = "routes"

    route_id = Column(Integer, primary_key=True, autoincrement=True)
    origin_port_id = Column(String(20), nullable=False, index=True)
    destination_port_id = Column(String(20), nullable=False, index=True)
    origin_port_name = Column(String(100))
    destination_port_name = Column(String(100))
    origin_country = Column(String(50))
    great_circle_nm = Column(Float)
    est_sailing_distance_nm = Column(Float)
    typical_cargo = Column(Text)
    routing_note = Column(Text)
    data_type = Column(String(50))


class FreightRate(Base):
    """Historical freight rate observation."""

    __tablename__ = "freight_rates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    origin_port_id = Column(String(20), nullable=False, index=True)
    destination_port_id = Column(String(20), nullable=False, index=True)
    vessel_class = Column(String(30), nullable=False, index=True)
    freight_rate_usd_per_day = Column(Float)
    freight_rate_usd_per_tonne = Column(Float)
    bdi_proxy_index = Column(Float)
    data_type = Column(String(50))


class Congestion(Base):
    """Port congestion snapshot."""

    __tablename__ = "congestion"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    port_id = Column(String(20), nullable=False, index=True)
    vessels_waiting = Column(Integer)
    avg_waiting_time_days = Column(Float)
    berth_occupancy_pct = Column(Float)
    data_type = Column(String(50))


class Weather(Base):
    """Weather observation at a port."""

    __tablename__ = "weather"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    port_id = Column(String(20), nullable=False, index=True)
    wind_speed_kmh = Column(Float)
    wave_height_m = Column(Float)
    rainfall_mm = Column(Float)
    cyclone_alert_level = Column(String(30))
    sea_state = Column(String(30))
    data_type = Column(String(50))


class Commodity(Base):
    """Commodity reference data."""

    __tablename__ = "commodities"

    commodity_id = Column(String(30), primary_key=True)
    commodity_name = Column(String(100), nullable=False)
    category = Column(String(50))
    typical_origin_countries = Column(Text)
    calorific_value_kcal_kg = Column(String(30))
    ash_content_pct = Column(String(30))
    typical_parcel_size_tonnes = Column(String(50))
    typical_vessel_class = Column(String(50))
    primary_indian_use = Column(String(100))
    data_type = Column(String(50))
    source_note = Column(Text)


class EconomicIndicator(Base):
    """Macroeconomic and commodity price indicators."""

    __tablename__ = "economic_indicators"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    brent_crude_usd_bbl = Column(Float)
    bunker_fuel_vlsfo_usd_tonne = Column(Float)
    newcastle_coal_price_usd_tonne = Column(Float)
    india_coal_import_demand_index = Column(Float)
    usd_inr_rate = Column(Float)
    china_pmi_manufacturing = Column(Float)
    data_type = Column(String(50))


class Event(Base):
    """Disruption events (cyclones, strikes, etc.)."""

    __tablename__ = "events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(Date, nullable=False, index=True)
    event_name = Column(String(100), nullable=False)
    event_category = Column(String(50))
    affected_region = Column(String(200))
    affected_ports_or_origins = Column(Text)
    severity = Column(String(30))
    description = Column(Text)
    data_type = Column(String(50))


class ModelRegistryEntry(Base):
    """ML model version tracking."""

    __tablename__ = "model_registry"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(30), nullable=False)
    model_type = Column(String(50))
    hyperparameters = Column(JSONB)
    metrics = Column(JSONB)
    artifact_path = Column(String(500))
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)


class RecommendationLog(Base):
    """Audit log of recommendations generated."""

    __tablename__ = "recommendation_logs"

    request_id = Column(String(36), primary_key=True)  # UUID
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    request_params = Column(JSONB)
    recommendation = Column(JSONB)
    model_version = Column(String(30))
