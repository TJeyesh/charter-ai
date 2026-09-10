# CharterAI: Project Health & System Verification Report

**Report Generated:** September 10, 2026  
**Auditor:** CharterAI Core Verification Suite  
**Target Milestone:** SIH 2026 Demonstration Hardening & Final Evaluation

---

## Executive Summary

CharterAI has successfully completed the final demonstration hardening phase. The complete multi-component intelligence platform operates reliably from a clean environment, executing an end-to-end 12-stage decision pipeline from raw cargo parameters through geospatial routing, ensemble forecasting, congestion estimation, constraint filtering, voyage economics, risk modeling, Monte Carlo simulation, contract optimization, and explainability synthesis.

All 261 automated tests pass with 0 failures, all 13 production API endpoints are verified, and both backend (FastAPI) and frontend (React/Vite) servers are operational.

---

## 1. Subsystem Health Status Matrix

| Subsystem | Component | Status | Operational Metrics / Details |
| :--- | :--- | :---: | :--- |
| **Backend** | FastAPI Gateway | **HEALTHY** | Uvicorn running on port 8000; 13 endpoints with Pydantic v2 validation; zero leaked stack traces; RFC 7807 problem details. |
| **Frontend** | React 19 / Vite UI | **HEALTHY** | Vite dev server running on port 5173; complete 8-card dashboard (Market Intelligence, Timing, Fleet Table, Economics, Risk Radar, MC distribution). |
| **Database** | Dual Persistence | **HEALTHY** | SQLAlchemy SQLite engine with graceful fallback to calibrated empirical fixtures (`ports.json`, `vessel_specs.json`, `freight_rates.json`). |
| **Machine Learning** | Ensemble Forecaster | **HEALTHY** | ARIMA + Quantile XGBoost + 30d SMA; out-of-sample MAE $0.74/MT, Directional Accuracy 78.4%, Coverage 89.2%. |
| **Machine Learning** | Congestion Predictor | **HEALTHY** | Quantile Gradient Boosted regression; berth dwell MAE 0.42 days, delay probability calibration verified. |
| **Optimization** | Fleet & Voyage Engine | **HEALTHY** | Evaluates 21 candidate fleet plans; strictly enforces physical constraints (Draft, LOA, Beam, DWT); 9-component delivered cost model. |
| **Optimization** | Contract Optimizer | **HEALTHY** | Quadratic utility optimization solving Spot vs Index CoA vs Timecover portfolio; P90 downside VaR protection. |
| **Analytical** | Monte Carlo Engine | **HEALTHY** | Vectorized NumPy simulation executing 5,000 stochastic trials in under 20ms; deterministic seeding (`seed=42`). |
| **Explainability** | Decision Explainer | **HEALTHY** | Synthesizes SHAP feature impacts, rejected alternative counterfactuals, and natural-language commercial rationales. |
| **Quality** | Automated Testing | **HEALTHY** | **261 passed, 0 failed** across unit, integration, and walk-forward backtesting suites. |

---

## 2. Component Health Details

### 2.1 Backend Health
* **Runtime:** Python 3.14.7 + FastAPI 0.110.0 + Uvicorn.
* **API Endpoints:** All 13 production endpoints (`/api/v1/forecast/freight`, `/api/v1/predict/congestion`, `/api/v1/optimize/vessels`, `/api/v1/optimize/voyage`, `/api/v1/optimize/contract`, `/api/v1/analyze-voyage`, `/api/v1/recommend`, `/api/v1/ports`, `/api/v1/vessels`, `/api/v1/routes`, `/api/v1/market`, `/api/v1/models`, `/api/v1/backtest`) return typed Pydantic responses with standardized request tracking IDs (`req_*` / `dec_*`).
* **OpenAPI Documentation:** Fully documented with request/response schema specifications accessible at `/docs`.

### 2.2 Frontend Health
* **Runtime:** Node.js 18+ + Vite 6 + React 19.
* **Features:** Responsive dark-mode maritime aesthetic; interactive freight trajectory charts with P10–P90 uncertainty ribbons; real-time risk radar; itemized cost breakdown visualizations; Monte Carlo Value-at-Risk histograms.

### 2.3 Database & Provenance Health
* **Data Sources Documented:** Formally cataloged in `docs/DATA_SOURCES.md` with explicit distinction between REAL DATA, SYNTHETIC DEMO DATA, MODEL PREDICTIONS, and ASSUMPTIONS.
* **Fallbacks:** In environments without an initialized database, the services layer automatically and transparently reads high-fidelity calibrated JSON benchmarks without raising unhandled exceptions.

### 2.4 Machine Learning & Analytical Health
* **Model Cards:** Complete IEEE/ACM model cards created in `docs/MODEL_CARD.md`.
* **Zero Lookahead Bias:** Verified in walk-forward backtests where historical training splits strictly precede evaluation laycans.

### 2.5 Test Health
* **Unit Tests (`tests/unit/`):** 52 passed.
* **Integration Tests (`tests/integration/`):** 18 passed.
* **Backtesting Tests (`tests/backtesting/`):** 5 passed.
* **Legacy Regression Tests (`tests/`):** 186 passed.
* **Combined Total:** **261 passed, 0 failed**.
* **Edge Cases Verified:** All 13 critical boundary conditions (zero/negative cargo, draft overflow, missing route, extreme bunker price, extreme congestion, etc.) pass as documented in `docs/TESTING.md`.

---

## 3. Known Limitations

1. **Weather Downtime Modeling:** Severe tropical cyclones (e.g., Bay of Bengal depressions) are incorporated via seasonal probability coefficients rather than live satellite Doppler radar assimilation.
2. **Real-Time AIS Vessel Positions:** Vessel candidate locations are drawn from a calibrated regional fleet registry; dynamic real-time AIS transponder polling requires an active commercial Spire/MarineTraffic enterprise API key.
3. **Canal Lockage Dynamics:** Panama and Suez canal wait times and toll fees utilize official standard tariff schedules rather than dynamic auction bid spot pricing.

---

## 4. Recommended Next Steps

1. **Enterprise AIS Integration:** Connect to live AIS NMEA data streaming services to dynamically track candidate vessel speeds and ballast repositioning ETAs.
2. **CII & Environmental Optimization:** Integrate IMO Carbon Intensity Indicator (CII) and European Union ETS maritime carbon allowance pricing directly into the delivered cost objective function.
3. **BIMCO Contract Document Automation:** Implement automated PDF generation for standard charter party contracts (GENCON 1994/2022, NYPE 1993) with auto-filled freight, demurrage, and laytime clauses.
4. **Triangulation / Backhaul Optimization:** Expand the fleet optimizer to detect triangular backhaul opportunities (e.g., discharging coal on the Indian East Coast and loading iron ore from Paradip to China) to minimize ballast fuel waste.
