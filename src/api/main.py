"""
Charter-AI — FastAPI Application Factory (Phase 14: Production API).

Creates and configures the FastAPI app with:
- Strict Pydantic validation on all requests
- Typed response schemas with standard metadata envelope
- Global exception handlers preventing internal stack trace exposure
- Comprehensive OpenAPI / Swagger documentation
- Singletons for ML models and optimization engines
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import (
    economics,
    forecast,
    health,
    ports,
    recommend,
    risk,
    routes,
    vessels,
    analyze_voyage,
    congestion,
    contracts,
    optimize,
    market,
    models_route,
    backtest_route,
)
from src.api.serializers import (
    generate_request_id,
    get_current_iso_timestamp,
    DEFAULT_API_VERSION,
)
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger

# Import services
from src.services.freight_forecast_service import FreightForecastService
from src.services.risk_service import RiskService
from src.services.voyage_economics_service import VoyageEconomicsService
from src.services.vessel_optimization_service import VesselOptimizationService
from src.services.contract_optimization_service import ContractOptimizationService
from src.services.congestion_service import CongestionService


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    logger.info(
        f"Charter-AI starting | version={settings.model_version} | "
        f"db={settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # Initialize Core Services (Singletons for the app lifecycle)
    app.state.congestion_service = CongestionService()
    app.state.forecast_service = FreightForecastService()
    app.state.risk_service = RiskService(congestion_service=app.state.congestion_service)
    app.state.economics_service = VoyageEconomicsService(congestion_service=app.state.congestion_service)
    app.state.vessel_service = VesselOptimizationService()
    app.state.contract_service = ContractOptimizationService()

    logger.info("Core services initialized.")

    yield

    logger.info("Charter-AI shutting down")


OPENAPI_TAGS = [
    {
        "name": "Recommendation",
        "description": "Unified 9-step decision engine orchestrating freight forecasting, port congestion, market timing, vessel optimization, voyage economics, 8-category risk, Monte Carlo simulation, and contract strategy.",
    },
    {
        "name": "Forecast",
        "description": "Multi-horizon machine learning (XGBoost, ARIMA, Seasonal, Ensemble) freight rate forecasting with P10/P90 quantile confidence intervals.",
    },
    {
        "name": "Prediction",
        "description": "Quantile gradient boosted regression for port congestion, vessel waiting days, and delay probabilities.",
    },
    {
        "name": "Optimization",
        "description": "Constrained fleet selection, multi-voyage schedule & cost optimization, and Monte Carlo contract allocation.",
    },
    {
        "name": "Market",
        "description": "Real-time dry bulk freight benchmarks, Baltic Exchange index levels (BDI, BCI, BPI, BSI), and bunker fuel pricing.",
    },
    {
        "name": "Models",
        "description": "CharterAI machine learning model registry, active versions, hyperparameters, and benchmark accuracy metrics.",
    },
    {
        "name": "Backtesting",
        "description": "Leak-free historical walk-forward backtesting framework evaluating forecast accuracy and chartering decisions against 5 empirical baselines.",
    },
    {
        "name": "Ports",
        "description": "Dry bulk port directory, physical constraints (draft, LOA, beam), cargo throughput, and congestion monitoring.",
    },
    {
        "name": "Vessels",
        "description": "Vessel class specifications, capacity boundaries, and commercial fleet registry.",
    },
    {
        "name": "Routes",
        "description": "Standard maritime shipping routes, great circle and sailing distances in nautical miles.",
    },
    {
        "name": "Contracts",
        "description": "Quantitative risk-aware chartering contract allocation between Spot, Short-term, and Medium-term commitments.",
    },
    {
        "name": "Economics",
        "description": "Voyage economics, TCE calculations, bunker consumption, and 9-component delivered cost breakdowns.",
    },
    {
        "name": "Risk",
        "description": "8-category probabilistic risk assessment and Monte Carlo simulation engine.",
    },
    {
        "name": "Analysis",
        "description": "Comprehensive single-request voyage analysis orchestrating the full decision stack.",
    },
    {
        "name": "Health",
        "description": "API status, database connectivity, and environment mode verification.",
    },
]


def create_app() -> FastAPI:
    """Create and configure the production FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Charter-AI Enterprise API",
        description=(
            "### AI-Powered Bulk Vessel Chartering & Freight Intelligence Platform\n\n"
            "Production-grade decision-support system for dry-bulk cargo chartering to India's East Coast ports.\n\n"
            "**Core Capabilities:**\n"
            "- Multi-horizon probabilistic freight rate forecasting (Ensemble, XGBoost, ARIMA)\n"
            "- Quantile port congestion and waiting time predictions\n"
            "- Constrained multi-voyage vessel optimization with physical port limits\n"
            "- 8-category maritime risk engine with 10,000-run Monte Carlo simulation\n"
            "- Risk-aware contract allocation (Spot vs Short-Term vs Medium-Term)\n"
            "- Structured XAI explainability and walk-forward historical backtesting\n"
        ),
        version=DEFAULT_API_VERSION,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # CORS Configuration
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Global Exception Handlers (Prevent raw stack trace leakage to clients)
    # -------------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = generate_request_id()
        logger.warning("Request validation error [%s] on %s: %s", req_id, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "status_code": 422,
                "request_id": req_id,
                "timestamp": get_current_iso_timestamp(),
                "message": "Input validation failed. Please verify request parameters and types.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = generate_request_id()
        logger.info("HTTPException %d [%s] on %s: %s", exc.status_code, req_id, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "status_code": exc.status_code,
                "request_id": req_id,
                "timestamp": get_current_iso_timestamp(),
                "message": str(exc.detail),
                "details": exc.detail if isinstance(exc.detail, (dict, list)) else None,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = generate_request_id()
        logger.exception("Unhandled server exception [%s] on %s: %s", req_id, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "status_code": 500,
                "request_id": req_id,
                "timestamp": get_current_iso_timestamp(),
                "message": f"An unexpected server error occurred. Quote reference: {req_id}",
                "details": None,
            },
        )

    # -------------------------------------------------------------------------
    # Router Mounts (All under /api/v1 prefix)
    # -------------------------------------------------------------------------
    api_prefix = "/api/v1"

    # Core System Endpoints
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(recommend.router, prefix=api_prefix)
    app.include_router(forecast.router, prefix=api_prefix)
    app.include_router(congestion.predict_router, prefix=api_prefix)
    app.include_router(congestion.router, prefix=api_prefix)
    app.include_router(optimize.router, prefix=api_prefix)
    app.include_router(ports.router, prefix=api_prefix)
    app.include_router(vessels.router, prefix=api_prefix)
    app.include_router(routes.router, prefix=api_prefix)
    app.include_router(market.router, prefix=api_prefix)
    app.include_router(models_route.router, prefix=api_prefix)
    app.include_router(backtest_route.router, prefix=api_prefix)

    # Auxiliary & Compatibility Endpoints
    app.include_router(analyze_voyage.router, prefix=api_prefix)
    app.include_router(contracts.router, prefix=api_prefix)
    app.include_router(economics.router, prefix=api_prefix)
    app.include_router(risk.router, prefix=api_prefix)

    return app


# Application instance (used by uvicorn)
app = create_app()
