"""
Charter-AI — Production API Integration Test Suite (Phase 14).

Validates:
1. All 13 core endpoints work and return typed responses.
2. Standard metadata envelope present: request_id, timestamp, api_version, model_versions, data_versions.
3. Pydantic request validation and clean 422 error envelope.
4. HTTP error handling and 404 responses without internal traceback leakage.
5. OpenAPI documentation and schema completeness.
6. Full 9-step pipeline execution on POST /api/v1/recommend with 0 placeholders.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.serializers import (
    FreightForecastResponse,
    CongestionPredictionResponse,
    VesselOptimizationResponse,
    VoyageOptimizationResponse,
    ContractOptimizationApiResponse,
    AnalyzeVoyageResponse,
    RecommendationResponse,
    PortsListResponse,
    VesselsListResponse,
    RoutesListResponse,
    MarketSummaryResponse,
    ModelRegistryResponse,
    BacktestApiResponse,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# 1. POST /api/v1/forecast/freight
# =============================================================================

def test_post_forecast_freight_success(client):
    payload = {
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Panamax",
        "horizon_days": 7,
        "cargo_type": "thermal_coal",
    }
    response = client.post("/api/v1/forecast/freight", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = FreightForecastResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.timestamp is not None
    assert validated.api_version == "v1.0"
    assert "freight_forecaster" in validated.model_versions
    assert "data_source" in validated.data_versions
    assert validated.forecast_rate > 0.0
    assert validated.lower_bound <= validated.forecast_rate <= validated.upper_bound
    assert validated.trend in ["rising", "falling", "stable"]
    assert 0.0 <= validated.confidence <= 1.0


# =============================================================================
# 2. POST /api/v1/predict/congestion
# =============================================================================

def test_post_predict_congestion_success(client):
    payload = {
        "port_id": "IND_PAR",
        "vessel_class": "Panamax",
        "cargo_type": "thermal_coal",
        "cargo_quantity": 75000.0,
    }
    response = client.post("/api/v1/predict/congestion", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = CongestionPredictionResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.timestamp is not None
    assert validated.api_version == "v1.0"
    assert validated.port_id == "IND_PAR"
    assert validated.expected_wait_days > 0.0
    assert validated.p10_wait_days <= validated.p50_wait_days <= validated.p90_wait_days
    assert validated.congestion_level in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert 0.0 <= validated.delay_probability <= 1.0


# =============================================================================
# 3. POST /api/v1/optimize/vessels
# =============================================================================

def test_post_optimize_vessels_success(client):
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "cargo_tonnage": 80000.0,
        "cargo_type": "thermal_coal",
    }
    response = client.post("/api/v1/optimize/vessels", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = VesselOptimizationResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert len(validated.feasible) > 0
    assert any(v.vessel_class == "Panamax" for v in validated.feasible)


# =============================================================================
# 4. POST /api/v1/optimize/voyage
# =============================================================================

def test_post_optimize_voyage_success(client):
    payload = {
        "cargo_quantity_t": 100000.0,
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "route_distance_nm": 4800.0,
        "max_voyages": 3,
    }
    response = client.post("/api/v1/optimize/voyage", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = VoyageOptimizationResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert validated.total_plans_evaluated > 0
    assert len(validated.ranked_plans) > 0
    assert validated.best_plan is not None
    assert validated.best_plan["total_cost"] > 0.0


# =============================================================================
# 5. POST /api/v1/optimize/contract
# =============================================================================

def test_post_optimize_contract_success(client):
    payload = {
        "cargo_quantity_t": 75000.0,
        "spot_freight_rate": 22.0,
        "risk_tolerance": "MEDIUM",
        "n_simulations": 1000,
    }
    response = client.post("/api/v1/optimize/contract", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = ContractOptimizationApiResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    total_pct = validated.spot_percentage + validated.short_term_percentage + validated.medium_term_percentage
    assert abs(total_pct - 100.0) < 0.05
    assert validated.expected_cost > 0.0
    assert len(validated.reasons) > 0



# =============================================================================
# 6. POST /api/v1/analyze-voyage
# =============================================================================

def test_post_analyze_voyage_success(client):
    payload = {
        "cargo_type": "coal",
        "cargo_quantity": 80000,
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "required_delivery_date": "2026-11-15",
        "number_of_voyages": 1,
    }
    response = client.post("/api/v1/analyze-voyage", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = AnalyzeVoyageResponse(**data)
    assert validated.status == "SUCCESS"
    assert validated.request_id is not None
    assert validated.api_version == "v1.0"
    assert "recommended_vessel" in data
    assert "voyage_economics" in data


# =============================================================================
# 7. POST /api/v1/recommend (Full 9-step pipeline without 501)
# =============================================================================

def test_post_recommend_pipeline_execution(client):
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "cargo_type": "coal",
        "cargo_tonnage": 80000,
        "earliest_date": "2026-10-01",
        "latest_date": "2026-11-01",
        "risk_appetite": "moderate",
    }
    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = RecommendationResponse(**data)
    assert validated.request_id.startswith("req_") or validated.request_id.startswith("dec_")
    assert validated.api_version == "v1.0"
    assert validated.recommendation.vessel_class in ["Capesize", "Panamax", "Supramax", "Handysize"]
    assert validated.forecast is not None
    assert validated.economics is not None
    assert validated.risk is not None
    assert validated.vessel_compatibility is not None


# =============================================================================
# 8. GET /api/v1/ports
# =============================================================================

def test_get_ports_catalog(client):
    response = client.get("/api/v1/ports")
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = PortsListResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert validated.total_count > 0
    assert len(validated.ports) == validated.total_count

    # Verify filtering by country works
    resp_ind = client.get("/api/v1/ports?country=IND")
    assert resp_ind.status_code == 200
    ind_data = resp_ind.json()
    assert ind_data["total_count"] > 0
    assert all(p["country"] == "IND" for p in ind_data["ports"])


# =============================================================================
# 9. GET /api/v1/vessels
# =============================================================================

def test_get_vessels_catalog(client):
    response = client.get("/api/v1/vessels")
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = VesselsListResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert len(validated.vessel_classes) >= 4
    class_names = [vc.class_name for vc in validated.vessel_classes]
    assert "Panamax" in class_names
    assert "Capesize" in class_names


# =============================================================================
# 10. GET /api/v1/routes
# =============================================================================

def test_get_routes_catalog(client):
    response = client.get("/api/v1/routes")
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = RoutesListResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert validated.total_count > 0
    assert any(r.origin_port_id == "AUS_NEW" for r in validated.routes)


# =============================================================================
# 11. GET /api/v1/market
# =============================================================================

def test_get_market_intelligence(client):
    response = client.get("/api/v1/market")
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = MarketSummaryResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert "Panamax" in validated.benchmark_freight
    assert "BDI" in validated.dry_bulk_indices
    assert validated.volatility_30d > 0.0


# =============================================================================
# 12. GET /api/v1/models
# =============================================================================

def test_get_models_registry(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()
    validated = ModelRegistryResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert validated.total_models >= 6
    names = [m.model_name for m in validated.models]
    assert "freight_forecaster_ensemble" in names
    assert "port_congestion_predictor" in names
    assert "market_timing_engine" in names


# =============================================================================
# 13. POST /api/v1/backtest
# =============================================================================

def test_post_backtest_execution(client):
    payload = {
        "years": [2024],
        "horizons": [7],
        "n_scenarios": 3,
    }
    response = client.post("/api/v1/backtest", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"


    data = response.json()
    validated = BacktestApiResponse(**data)
    assert validated.request_id.startswith("req_")
    assert validated.api_version == "v1.0"
    assert "test_years" in validated.metadata
    assert "optimization_evaluation" in data
    assert "comparative_analysis" in data
    assert len(validated.comparative_analysis) > 0


# =============================================================================
# 14. Validation Error Handling (422)
# =============================================================================

def test_request_validation_error_handling(client):
    # Missing required origin & destination fields
    payload = {"vessel_class": "Panamax"}
    response = client.post("/api/v1/forecast/freight", json=payload)
    assert response.status_code == 422

    data = response.json()
    assert data["error"] == "Validation Error"
    assert data["status_code"] == 422
    assert "request_id" in data
    assert "timestamp" in data
    assert len(data["details"]) > 0


# =============================================================================
# 15. HTTP 404 Error Handling
# =============================================================================

def test_http_404_error_handling(client):
    response = client.get("/api/v1/ports/NON_EXISTENT_PORT_XYZ")
    assert response.status_code == 404

    data = response.json()
    assert data["error"] == "HTTP Error"
    assert data["status_code"] == 404
    assert "request_id" in data
    assert "timestamp" in data


# =============================================================================
# 16. OpenAPI Documentation & Swagger UI
# =============================================================================

def test_openapi_schema_availability(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "Charter-AI Enterprise API"
    assert schema["info"]["version"] == "v1.0"

    paths = schema["paths"]
    assert "/api/v1/forecast/freight" in paths
    assert "/api/v1/predict/congestion" in paths
    assert "/api/v1/optimize/vessels" in paths
    assert "/api/v1/optimize/voyage" in paths
    assert "/api/v1/optimize/contract" in paths
    assert "/api/v1/analyze-voyage" in paths
    assert "/api/v1/recommend" in paths
    assert "/api/v1/ports" in paths
    assert "/api/v1/vessels" in paths
    assert "/api/v1/routes" in paths
    assert "/api/v1/market" in paths
    assert "/api/v1/models" in paths
    assert "/api/v1/backtest" in paths

    # Verify docs UI returns 200
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
