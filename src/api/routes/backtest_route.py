"""
Charter-AI — Historical Backtesting API Router (Phase 14).

Provides endpoints for executing leak-free walk-forward historical evaluations:
POST /api/v1/backtest
"""

from fastapi import APIRouter, HTTPException, Request

from src.api.serializers import BacktestApiRequest, BacktestApiResponse
from src.backtesting.backtest_engine import BacktestEngine
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtesting"])


@router.post("", response_model=BacktestApiResponse)
async def run_backtest(
    payload: BacktestApiRequest,
    request: Request,
) -> BacktestApiResponse:
    """
    POST /api/v1/backtest
    
    Execute walk-forward historical backtesting across specified test years and horizons.
    Replays tender decisions against 5 baselines and returns comparative performance metrics.
    """
    try:
        years = payload.years or [2024]
        horizons = payload.horizons or [7]
        engine = BacktestEngine(seed=payload.seed)

        report = engine.run_walk_forward_backtest(
            years=years,
            horizons=horizons,
            max_scenarios=payload.n_scenarios,
        )


        return BacktestApiResponse(
            metadata=report.get("metadata", {}),
            forecast_evaluation=report.get("forecast_evaluation", {}),
            optimization_evaluation=report.get("optimization_evaluation", {}),
            comparative_analysis=report.get("comparative_analysis", {}),
            savings_summary=report.get("comparative_analysis", {}),
        )
    except Exception as e:
        logger.exception("Historical backtesting execution error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Backtesting engine error: {str(e)}",
        )
