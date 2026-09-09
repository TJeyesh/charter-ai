"""
Charter-AI — Recommendation Endpoint.

The primary endpoint: combines vessel selection, forecasting, risk assessment,
voyage economics, and contract optimization into a single recommendation.
"""

from fastapi import APIRouter, HTTPException

from src.api.serializers import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/recommend", tags=["Recommendation"])


@router.post("", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """
    Generate a full chartering recommendation.

    This endpoint orchestrates:
    1. Vessel-port compatibility check
    2. Freight rate forecast
    3. Market timing analysis
    4. Risk assessment (market + port + weather + operational)
    5. Voyage economics calculation
    6. Contract strategy optimization
    7. SHAP explainability

    NOTE: ML models are not yet trained. This endpoint will return
    an error until the forecasting and prediction models are implemented.
    """
    # TODO: Implement full recommendation pipeline
    raise HTTPException(
        status_code=501,
        detail=(
            "Recommendation engine not yet fully implemented. "
            "ML models must be trained first. Deterministic components "
            "(vessel selection, economics, risk) are available via their "
            "individual endpoints."
        ),
    )
