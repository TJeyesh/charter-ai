"""
Charter-AI — FastAPI Application Factory.

Creates and configures the FastAPI app with CORS, routes, and lifespan events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import economics, forecast, health, ports, recommend, risk, routes, vessels
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger

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
    yield
    logger.info("Charter-AI shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Charter-AI",
        description=(
            "AI-Powered Bulk Vessel Chartering & Freight Intelligence Platform. "
            "Decision-support system for dry-bulk cargo chartering to India's East Coast ports."
        ),
        version=settings.model_version,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules under /api/v1 prefix
    api_prefix = "/api/v1"
    
    # We must import analyze_voyage lazily or at module level, I will add it to the top later,
    # Actually let's just do it cleanly.
    from src.api.routes import analyze_voyage
    
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(ports.router, prefix=api_prefix)
    app.include_router(routes.router, prefix=api_prefix)
    app.include_router(vessels.router, prefix=api_prefix)
    app.include_router(forecast.router, prefix=api_prefix)
    app.include_router(recommend.router, prefix=api_prefix)
    app.include_router(economics.router, prefix=api_prefix)
    app.include_router(risk.router, prefix=api_prefix)
    app.include_router(analyze_voyage.router, prefix=api_prefix)

    return app


# Application instance (used by uvicorn)
app = create_app()
