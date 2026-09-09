"""
Charter-AI — Voyage Economics Engine

A transparent, formula-based engine to estimate the total economic cost 
of a proposed voyage, accounting for positioning, bunker, port charges, 
waiting time, and expected demurrage.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VoyageCostInputs:
    cargo_quantity_t: float
    freight_rate_usd: float # Assumed USD per tonne
    vessel_speed_knots: float
    vessel_daily_fuel_consumption_tpd: float
    vessel_daily_hire_cost_usd: float
    route_distance_nm: float
    positioning_distance_nm: float
    fuel_price_usd_per_t: float
    load_port_cost_usd: float
    discharge_port_cost_usd: float
    expected_waiting_days: float
    daily_demurrage_rate_usd: float
    other_costs_usd: float

@dataclass
class VoyageCostBreakdown:
    freight_cost_usd: float
    bunker_cost_usd: float
    port_costs_usd: float
    waiting_cost_usd: float
    expected_demurrage_usd: float
    positioning_cost_usd: float
    other_costs_usd: float
    total_voyage_cost_usd: float

def calculate_voyage_cost(inputs: VoyageCostInputs) -> VoyageCostBreakdown:
    """
    Transparently calculates the total economic cost of a voyage.
    """
    
    # 1. Freight Cost (Rate * Quantity)
    freight_cost = inputs.freight_rate_usd * inputs.cargo_quantity_t
    
    # Calculate days
    # Speed in knots = nautical miles per hour
    # 1 day = 24 hours
    sailing_days = inputs.route_distance_nm / (inputs.vessel_speed_knots * 24.0) if inputs.vessel_speed_knots > 0 else 0
    positioning_days = inputs.positioning_distance_nm / (inputs.vessel_speed_knots * 24.0) if inputs.vessel_speed_knots > 0 else 0
    
    # 2. Bunker / Fuel Cost
    # Assuming vessel consumes fuel only while sailing and positioning at the given rate
    # Port fuel consumption is assumed negligible or bundled into other port costs for this formula
    total_sailing_days = sailing_days + positioning_days
    bunker_cost = total_sailing_days * inputs.vessel_daily_fuel_consumption_tpd * inputs.fuel_price_usd_per_t
    
    # 3. Port Costs
    port_costs = inputs.load_port_cost_usd + inputs.discharge_port_cost_usd
    
    # 4. Waiting Cost
    # Represents the opportunity cost of the vessel idling
    waiting_cost = inputs.expected_waiting_days * inputs.vessel_daily_hire_cost_usd
    
    # 5. Demurrage Estimate
    # Penalties for exceeding allowable laytime (often overlaps with severe waiting)
    # We treat expected_waiting_days > 2 as demurrage triggers for this simple formula
    demurrage_days = max(0.0, inputs.expected_waiting_days - 2.0)
    expected_demurrage = demurrage_days * inputs.daily_demurrage_rate_usd
    
    # 6. Positioning / Deadheading Cost
    # The cost of time taken to reach the origin port
    positioning_time_cost = positioning_days * inputs.vessel_daily_hire_cost_usd
    
    # Total
    total_cost = (
        freight_cost +
        bunker_cost +
        port_costs +
        waiting_cost +
        expected_demurrage +
        positioning_time_cost +
        inputs.other_costs_usd
    )
    
    return VoyageCostBreakdown(
        freight_cost_usd=round(freight_cost, 2),
        bunker_cost_usd=round(bunker_cost, 2),
        port_costs_usd=round(port_costs, 2),
        waiting_cost_usd=round(waiting_cost, 2),
        expected_demurrage_usd=round(expected_demurrage, 2),
        positioning_cost_usd=round(positioning_time_cost, 2),
        other_costs_usd=round(inputs.other_costs_usd, 2),
        total_voyage_cost_usd=round(total_cost, 2)
    )
