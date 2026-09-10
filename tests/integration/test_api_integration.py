"""
Integration Tests: Production FastAPI Endpoints Suite.

Verifies end-to-end API interaction across all 13 v1 endpoints,
Pydantic schema validation, structured error envelopes, and metadata tracking.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestApiIntegration:
    """End-to-end integration tests for all 13 core & auxiliary endpoints."""

    def test_post_forecast_freight(self, client):
        payload = {
            "origin": "AUS_NEW",
            "destination": "IND_GVM",
            "vessel_class": "Capesize",
            "horizon_days": 14,
            "cargo_type": "thermal_coal",
        }
        res = client.post("/api/v1/forecast/freight", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "timestamp" in data
        assert "model_versions" in data
        assert "forecast_rate" in data
        assert data["lower_bound"] <= data["forecast_rate"] <= data["upper_bound"]

    def test_post_predict_congestion(self, client):
        payload = {
            "port_id": "IND_GVM",
            "vessel_class": "Capesize",
            "target_date": "2026-04-15",
            "cargo_type": "thermal_coal",
            "cargo_quantity": 80000.0,
        }
        res = client.post("/api/v1/predict/congestion", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "expected_wait_days" in data
        assert data["p10_wait_days"] <= data["expected_wait_days"] <= data["p90_wait_days"]

    def test_post_optimize_vessels(self, client):
        payload = {
            "origin_port_id": "AUS_NEW",
            "destination_port_id": "IND_GVM",
            "cargo_tonnage": 82000.0,
            "cargo_type": "coal",
        }
        res = client.post("/api/v1/optimize/vessels", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert len(data["feasible"]) >= 1

    def test_post_optimize_voyage(self, client):
        payload = {
            "cargo_quantity_t": 100000.0,
            "origin_port_id": "AUS_NEW",
            "destination_port_id": "IND_GVM",
            "cargo_type": "Coal",
            "route_distance_nm": 4800.0,
            "freight_rate_usd": 22.0,
            "max_voyages": 3,
        }
        res = client.post("/api/v1/optimize/voyage", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["total_plans_evaluated"] >= 1

    def test_post_optimize_contract(self, client):
        payload = {
            "cargo_quantity_t": 75000.0,
            "spot_freight_rate": 22.0,
            "freight_volatility_pct": 18.0,
            "risk_tolerance": "MEDIUM",
            "n_simulations": 1000,
            "seed": 42,
        }
        res = client.post("/api/v1/optimize/contract", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["recommended_strategy"] != ""
        total_pct = data["spot_percentage"] + data["short_term_percentage"] + data["medium_term_percentage"]
        assert total_pct == pytest.approx(100.0)

    def test_post_analyze_voyage(self, client):
        payload = {
            "cargo_type": "coal",
            "cargo_quantity": 80000,
            "origin": "AUS_NEW",
            "destination": "IND_GVM",
            "required_delivery_date": "2026-11-15",
            "number_of_voyages": 1,
        }
        res = client.post("/api/v1/analyze-voyage", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert "request_id" in data

    def test_post_recommend_pipeline_execution(self, client):
        payload = {
            "origin_port_id": "AUS_NEW",
            "destination_port_id": "IND_GVM",
            "cargo_type": "coal",
            "cargo_tonnage": 80000,
            "earliest_date": "2026-10-01",
            "latest_date": "2026-11-01",
            "risk_appetite": "moderate",
        }
        res = client.post("/api/v1/recommend", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "recommendation" in data
        assert data["recommendation"]["vessel_class"] in ["Capesize", "Panamax", "Supramax", "Handysize"]

    def test_get_ports_catalog(self, client):
        res = client.get("/api/v1/ports")
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["total_count"] > 0
        assert len(data["ports"]) > 0

    def test_get_vessels_catalog(self, client):
        res = client.get("/api/v1/vessels")
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["total_count"] > 0
        assert len(data["vessels"]) > 0

    def test_get_routes_catalog(self, client):
        res = client.get("/api/v1/routes")
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["total_count"] > 0

    def test_get_market_intelligence(self, client):
        res = client.get("/api/v1/market")
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "dry_bulk_indices" in data
        assert "bunker_prices" in data

    def test_get_models_registry(self, client):
        res = client.get("/api/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert data["total_models"] > 0

    def test_post_backtest_execution(self, client):
        payload = {
            "years": [2023],
            "horizons": [7],
            "n_scenarios": 1,
            "seed": 42,
        }
        res = client.post("/api/v1/backtest", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "savings_summary" in data

    def test_request_validation_error_envelope(self, client):
        """Invalid payload must return 422 with structured ErrorResponse (no stack traces)."""
        payload = {
            "origin": "AUS_NEW",
            # Missing destination, cargo_quantity <= 0
            "cargo_quantity": -500.0,
        }
        res = client.post("/api/v1/analyze-voyage", json=payload)
        assert res.status_code == 422
        data = res.json()
        assert data["error"] == "Validation Error"
        assert "request_id" in data
        assert "timestamp" in data

    def test_http_404_error_envelope(self, client):
        """Non-existent endpoint must return 404 with structured ErrorResponse."""
        res = client.get("/api/v1/non_existent_route")
        assert res.status_code == 404
        data = res.json()
        assert "request_id" in data
        assert "timestamp" in data

    def test_openapi_schema_availability(self, client):
        """OpenAPI JSON definition and Swagger docs must be accessible."""
        res_json = client.get("/openapi.json")
        assert res_json.status_code == 200
        schema = res_json.json()
        assert "paths" in schema
        assert "/api/v1/recommend" in schema["paths"]

        res_docs = client.get("/docs")
        assert res_docs.status_code == 200
