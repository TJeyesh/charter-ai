"""
Charter-AI — Pydantic Data Schemas.

Validation schemas for all ingested datasets and API request/response shapes.
These mirror the CSV structures discovered in the existing datasets/ folder.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Port
# =============================================================================

class PortSchema(BaseModel):
    """Schema matching ports.csv."""

    port_id: str = Field(..., description="Unique port identifier, e.g. IND_VZG")
    port_name: str
    state: str
    country: str
    latitude: float
    longitude: float
    port_type: str
    operator: str
    berths_total: int
    max_draft_m: float
    max_loa_m: float
    max_beam_m: float
    max_dwt: int = Field(..., alias="max_dwt_capesize_capable")
    annual_capacity_mtpa: str
    primary_cargo: str
    data_type: str
    source_note: Optional[str] = None


# =============================================================================
# Vessel Class
# =============================================================================

class VesselClassSchema(BaseModel):
    """Schema matching vessels.csv."""

    vessel_class: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    loa_min_m: float
    loa_max_m: float
    beam_min_m: float
    beam_max_m: float
    draft_min_m: float
    draft_max_m: float
    typical_cargo: str
    data_type: str
    source_note: Optional[str] = None


# =============================================================================
# Route
# =============================================================================

class RouteSchema(BaseModel):
    """Schema matching routes.csv."""

    origin_port_id: str
    origin_port_name: str
    origin_country: str
    destination_port_id: str
    destination_port_name: str
    great_circle_nm: float
    est_sailing_distance_nm: float
    typical_cargo: str
    routing_note: Optional[str] = None
    data_type: str


# =============================================================================
# Freight Rate
# =============================================================================

class FreightRateSchema(BaseModel):
    """Schema matching freight_rates.csv."""

    date: date
    origin_port_id: str
    destination_port_id: str
    vessel_class: str
    freight_rate_usd_per_day: float
    freight_rate_usd_per_tonne: float
    bdi_proxy_index: float
    data_type: str


# =============================================================================
# Port Congestion
# =============================================================================

class CongestionSchema(BaseModel):
    """Schema matching congestion.csv."""

    date: date
    port_id: str
    vessels_waiting: int
    avg_waiting_time_days: float
    berth_occupancy_pct: float
    data_type: str


# =============================================================================
# Weather
# =============================================================================

class WeatherSchema(BaseModel):
    """Schema matching weather.csv."""

    date: date
    port_id: str
    wind_speed_kmh: float
    wave_height_m: float
    rainfall_mm: float
    cyclone_alert_level: str
    sea_state: str
    data_type: str


# =============================================================================
# Commodity
# =============================================================================

class CommoditySchema(BaseModel):
    """Schema matching commodities.csv."""

    commodity_id: str
    commodity_name: str
    category: str
    typical_origin_countries: str
    calorific_value_kcal_kg: Optional[str] = None
    ash_content_pct: Optional[str] = None
    typical_parcel_size_tonnes: Optional[str] = None
    typical_vessel_class: Optional[str] = None
    primary_indian_use: str
    data_type: str
    source_note: Optional[str] = None


# =============================================================================
# Economic Indicator
# =============================================================================

class EconomicIndicatorSchema(BaseModel):
    """Schema matching economic_indicators.csv."""

    date: date
    brent_crude_usd_bbl: float
    bunker_fuel_vlsfo_usd_tonne: float
    newcastle_coal_price_usd_tonne: float
    india_coal_import_demand_index: float
    usd_inr_rate: float
    china_pmi_manufacturing: float
    data_type: str


# =============================================================================
# Disruption Event
# =============================================================================

class EventSchema(BaseModel):
    """Schema matching events.csv."""

    event_date: date
    event_name: str
    event_category: str
    affected_region: str
    affected_ports_or_origins: str
    severity: str
    description: str
    data_type: str
