"""
Charter-AI — Demurrage Cost Calculator.

Estimates demurrage exposure based on predicted idle time,
contractual laytime, and demurrage rate.

DETERMINISTIC formula, but uses ML-predicted idle time as input.
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default demurrage rates by vessel class (USD/day)
DEFAULT_DEMURRAGE_RATES = {
    "Handysize": 12_000,
    "Supramax": 15_000,
    "Panamax": 20_000,
    "Post-Panamax": 25_000,
    "Capesize": 35_000,
    "VLOC": 45_000,
}

# Default allowed laytime (days) by vessel class
DEFAULT_LAYTIME_DAYS = {
    "Handysize": 5.0,
    "Supramax": 5.0,
    "Panamax": 6.0,
    "Post-Panamax": 6.0,
    "Capesize": 7.0,
    "VLOC": 8.0,
}


@dataclass
class DemurrageEstimate:
    """Demurrage cost estimate."""
    predicted_idle_days: float
    allowed_laytime_days: float
    demurrage_days: float  # max(0, idle - laytime)
    demurrage_rate_usd_per_day: float
    total_demurrage_usd: float
    despatch_days: float  # max(0, laytime - idle) — potential savings
    despatch_rate_usd_per_day: float  # Typically 50% of demurrage rate
    potential_despatch_usd: float
    net_demurrage_exposure_usd: float


class DemurrageCalculator:
    """
    Calculates demurrage cost and despatch savings.

    Demurrage: Penalty paid to shipowner when vessel stays beyond allowed laytime.
    Despatch: Reward to charterer if vessel completes loading/discharge early.

    Formula:
        demurrage = max(0, predicted_idle_days - allowed_laytime) × demurrage_rate
        despatch  = max(0, allowed_laytime - predicted_idle_days) × despatch_rate
        net_exposure = demurrage - despatch
    """

    def calculate(
        self,
        vessel_class: str,
        predicted_idle_days: float,
        allowed_laytime_days: Optional[float] = None,
        demurrage_rate: Optional[float] = None,
        despatch_rate_fraction: float = 0.50,
    ) -> DemurrageEstimate:
        """
        Calculate demurrage exposure.

        Args:
            vessel_class: Vessel class name.
            predicted_idle_days: ML-predicted waiting/idle time in days.
            allowed_laytime_days: Contractual laytime. If None, uses default.
            demurrage_rate: USD/day rate. If None, uses class default.
            despatch_rate_fraction: Despatch as fraction of demurrage rate.

        Returns:
            DemurrageEstimate with costs and potential savings.
        """
        laytime = allowed_laytime_days or DEFAULT_LAYTIME_DAYS.get(vessel_class, 6.0)
        dem_rate = demurrage_rate or DEFAULT_DEMURRAGE_RATES.get(vessel_class, 20_000)
        desp_rate = dem_rate * despatch_rate_fraction

        dem_days = max(0.0, predicted_idle_days - laytime)
        desp_days = max(0.0, laytime - predicted_idle_days)

        total_dem = dem_days * dem_rate
        potential_desp = desp_days * desp_rate
        net_exposure = total_dem - potential_desp

        return DemurrageEstimate(
            predicted_idle_days=round(predicted_idle_days, 2),
            allowed_laytime_days=laytime,
            demurrage_days=round(dem_days, 2),
            demurrage_rate_usd_per_day=dem_rate,
            total_demurrage_usd=round(total_dem, 0),
            despatch_days=round(desp_days, 2),
            despatch_rate_usd_per_day=desp_rate,
            potential_despatch_usd=round(potential_desp, 0),
            net_demurrage_exposure_usd=round(net_exposure, 0),
        )
