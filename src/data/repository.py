"""
Charter-AI — Data Access Repository.

All database queries are centralized here behind a clean interface.
Route handlers and business logic should never write raw SQL.
"""

from datetime import date
from typing import List, Optional

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import (
    Congestion,
    EconomicIndicator,
    Event,
    FreightRate,
    Port,
    Route,
    VesselClassModel,
    Weather,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PortRepository:
    """Read operations for ports."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, country: Optional[str] = None) -> List[Port]:
        stmt = select(Port)
        if country:
            stmt = stmt.where(Port.country == country)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, port_id: str) -> Optional[Port]:
        result = await self.session.execute(
            select(Port).where(Port.port_id == port_id)
        )
        return result.scalar_one_or_none()


class VesselClassRepository:
    """Read operations for vessel classes."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> List[VesselClassModel]:
        result = await self.session.execute(select(VesselClassModel))
        return list(result.scalars().all())

    async def get_by_id(self, class_id: str) -> Optional[VesselClassModel]:
        result = await self.session.execute(
            select(VesselClassModel).where(VesselClassModel.class_id == class_id)
        )
        return result.scalar_one_or_none()


class RouteRepository:
    """Read operations for routes."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> List[Route]:
        result = await self.session.execute(select(Route))
        return list(result.scalars().all())

    async def find(
        self,
        origin_port_id: Optional[str] = None,
        destination_port_id: Optional[str] = None,
    ) -> List[Route]:
        stmt = select(Route)
        if origin_port_id:
            stmt = stmt.where(Route.origin_port_id == origin_port_id)
        if destination_port_id:
            stmt = stmt.where(Route.destination_port_id == destination_port_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class FreightRateRepository:
    """Read operations for freight rates."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_history(
        self,
        origin_port_id: str,
        destination_port_id: str,
        vessel_class: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[FreightRate]:
        """Fetch freight rate time series for a specific route and vessel class."""
        stmt = (
            select(FreightRate)
            .where(FreightRate.origin_port_id == origin_port_id)
            .where(FreightRate.destination_port_id == destination_port_id)
            .where(FreightRate.vessel_class == vessel_class)
            .order_by(FreightRate.date)
        )
        if start_date:
            stmt = stmt.where(FreightRate.date >= start_date)
        if end_date:
            stmt = stmt.where(FreightRate.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(
        self,
        origin_port_id: str,
        destination_port_id: str,
        vessel_class: str,
    ) -> Optional[FreightRate]:
        """Fetch the most recent freight rate for a route/vessel."""
        stmt = (
            select(FreightRate)
            .where(FreightRate.origin_port_id == origin_port_id)
            .where(FreightRate.destination_port_id == destination_port_id)
            .where(FreightRate.vessel_class == vessel_class)
            .order_by(FreightRate.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class CongestionRepository:
    """Read operations for port congestion."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_history(
        self,
        port_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Congestion]:
        stmt = (
            select(Congestion)
            .where(Congestion.port_id == port_id)
            .order_by(Congestion.date)
        )
        if start_date:
            stmt = stmt.where(Congestion.date >= start_date)
        if end_date:
            stmt = stmt.where(Congestion.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(self, port_id: str) -> Optional[Congestion]:
        stmt = (
            select(Congestion)
            .where(Congestion.port_id == port_id)
            .order_by(Congestion.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class WeatherRepository:
    """Read operations for weather data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_history(
        self,
        port_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Weather]:
        stmt = (
            select(Weather)
            .where(Weather.port_id == port_id)
            .order_by(Weather.date)
        )
        if start_date:
            stmt = stmt.where(Weather.date >= start_date)
        if end_date:
            stmt = stmt.where(Weather.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class EconomicIndicatorRepository:
    """Read operations for macroeconomic indicators."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_range(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[EconomicIndicator]:
        stmt = select(EconomicIndicator).order_by(EconomicIndicator.date)
        if start_date:
            stmt = stmt.where(EconomicIndicator.date >= start_date)
        if end_date:
            stmt = stmt.where(EconomicIndicator.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(self) -> Optional[EconomicIndicator]:
        stmt = (
            select(EconomicIndicator)
            .order_by(EconomicIndicator.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
