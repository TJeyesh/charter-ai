"""
Charter-AI — Optimization API Router (Phase 14).

Provides unified optimization endpoints for:
1. POST /api/v1/optimize/vessels - Port physical constraint filtering and candidate vessel selection
2. POST /api/v1/optimize/voyage - Multi-voyage fleet allocation, schedule and cost optimization
3. POST /api/v1/optimize/contract - Monte Carlo contract allocation (spot vs term vs hybrid)
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Request

from src.api.serializers import (
    VesselOptimizationRequest,
    VesselOptimizationResponse,
    VesselCompatibilityResponse,
    VoyageOptimizationRequest,
    VoyageOptimizationResponse,
    ContractOptimizationApiRequest,
    ContractOptimizationApiResponse,
)
from src.optimization.vessel_selector import VesselSelector, PortConstraints
from src.optimization.multi_voyage_optimizer import MultiVoyageOptimizer, FleetOptimizationRequest
from src.optimization.vessel_scoring import OptimizationWeights
from src.services.contract_optimization_service import ContractOptimizationService
from src.data.mock_db import get_mock_vessel_db, get_mock_port_info
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/optimize", tags=["Optimization"])

_selector = VesselSelector()
_contract_service = ContractOptimizationService()


@router.post("/vessels", response_model=VesselOptimizationResponse)
async def optimize_vessels(
    payload: VesselOptimizationRequest,
    request: Request,
) -> VesselOptimizationResponse:
    """
    POST /api/v1/optimize/vessels
    
    Check physical compatibility (draft, LOA, beam, DWT) and rank feasible vessel classes
    for a given origin-destination corridor and cargo quantity.
    """
    try:
        orig_info = get_mock_port_info(payload.origin_port_id)
        dest_info = get_mock_port_info(payload.destination_port_id)

        origin_constraints = PortConstraints(
            port_id=orig_info.port_id,
            port_name=orig_info.port_name,
            max_draft_m=orig_info.max_draft_m,
            max_loa_m=orig_info.max_loa_m,
            max_beam_m=orig_info.max_beam_m,
            max_dwt=orig_info.max_draft_m * 10000,
        )
        dest_constraints = PortConstraints(
            port_id=dest_info.port_id,
            port_name=dest_info.port_name,
            max_draft_m=dest_info.max_draft_m,
            max_loa_m=dest_info.max_loa_m,
            max_beam_m=dest_info.max_beam_m,
            max_dwt=dest_info.max_draft_m * 10000,
        )

        vessel_specs = get_mock_vessel_db()

        result = _selector.select_vessels(
            origin_port=origin_constraints,
            destination_port=dest_constraints,
            vessel_specs=vessel_specs,
            cargo_tonnage=int(payload.cargo_tonnage) if payload.cargo_tonnage else None,
        )

        feasible_resp = [
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.feasible
        ]

        excluded_resp = [
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.excluded
        ]

        return VesselOptimizationResponse(
            origin_port_id=payload.origin_port_id,
            destination_port_id=payload.destination_port_id,
            cargo_tonnage=payload.cargo_tonnage,
            cargo_type=payload.cargo_type,
            feasible=feasible_resp,
            excluded=excluded_resp,
            summary={
                "feasible_count": len(feasible_resp),
                "excluded_count": len(excluded_resp),
                "recommended_class": feasible_resp[0].vessel_class if feasible_resp else None,
            },
        )
    except Exception as e:
        logger.exception("Vessel optimization error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Vessel optimization error: {str(e)}",
        )


@router.post("/voyage", response_model=VoyageOptimizationResponse)
async def optimize_voyage(
    payload: VoyageOptimizationRequest,
    request: Request,
) -> VoyageOptimizationResponse:
    """
    POST /api/v1/optimize/voyage
    
    Multi-voyage fleet and schedule optimization. Compares single vs multi-vessel plans,
    evaluates delivered costs, risk scores, and schedule deadlines.
    """
    try:
        weights = OptimizationWeights(
            cost_weight=payload.cost_weight,
            schedule_weight=payload.schedule_weight,
            risk_weight=payload.risk_weight,
            utilization_weight=payload.utilization_weight,
            demurrage_weight=payload.demurrage_weight,
        )

        opt_request = FleetOptimizationRequest(
            cargo_quantity_t=payload.cargo_quantity_t,
            origin_port_id=payload.origin_port_id,
            destination_port_id=payload.destination_port_id,
            cargo_type=payload.cargo_type,
            route_distance_nm=payload.route_distance_nm,
            freight_rate_usd=payload.freight_rate_usd,
            delivery_deadline_days=payload.delivery_deadline_days,
            max_voyages=payload.max_voyages,
            allow_mixed_classes=payload.allow_mixed_classes,
            weights=weights,
        )

        optimizer = MultiVoyageOptimizer()
        ranked_plans = optimizer.optimize(opt_request)
        serialized_plans = [plan.to_dict() for plan in ranked_plans]

        return VoyageOptimizationResponse(
            total_plans_evaluated=len(serialized_plans),
            ranked_plans=serialized_plans,
            best_plan=serialized_plans[0] if serialized_plans else None,
        )
    except Exception as e:
        logger.exception("Voyage fleet optimization error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Voyage fleet optimization error: {str(e)}",
        )


@router.post("/contract", response_model=ContractOptimizationApiResponse)
async def optimize_contract(
    payload: ContractOptimizationApiRequest,
    request: Request,
) -> ContractOptimizationApiResponse:
    """
    POST /api/v1/optimize/contract
    
    Quantitatively optimize chartering contract strategy across Spot, Term, and Hybrid allocations
    using Monte Carlo simulation and multi-objective utility functions.
    """
    try:
        service: ContractOptimizationService = getattr(
            request.app.state, "contract_service", None
        )
        if service is None:
            service = _contract_service

        result = service.optimize_risk_aware(payload.model_dump())

        return ContractOptimizationApiResponse(
            recommended_strategy=result.recommended_strategy,
            spot_percentage=result.spot_percentage,
            short_term_percentage=result.short_term_percentage,
            medium_term_percentage=result.medium_term_percentage,
            expected_cost=result.expected_cost,
            p90_cost=result.p90_cost,
            risk_score=result.risk_score,
            flexibility_score=result.flexibility_score,
            reasons=result.reasons,
            evaluated_strategies=[s.to_dict() for s in result.evaluated_strategies],
        )
    except Exception as e:
        logger.exception("Contract strategy optimization error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Contract strategy optimization error: {str(e)}",
        )
