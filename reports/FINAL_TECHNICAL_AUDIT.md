# CharterAI V2: Comprehensive Technical Audit & Architectural Review

**Document:** `reports/FINAL_TECHNICAL_AUDIT.md`  
**Role:** Senior Maritime Technology Architect, Principal ML Engineer, Optimization Engineer & Software Quality Reviewer  
**Audit Date:** September 10, 2026  
**Scope:** Full CharterAI V2 Codebase (Backend, Frontend, ML Models, Optimization, Economics, Risk, Data, Testing)  
**Constraint Enforced:** No source code was modified during this review.

---

## Executive Summary

An independent technical audit of the CharterAI V2 codebase was conducted across 24 critical operational, mathematical, and algorithmic dimensions. 

While the system presents an impressive end-to-end architecture with 261 passing unit and integration tests and a polished demo runner, this audit identified **several high-impact domain discrepancies, target leakages, maritime calculation double-counts, and frontend-backend catalog mismatches** that must be addressed prior to final Smart India Hackathon (SIH) live jury evaluation.

---

## Comprehensive Issue Log by Category (All 24 Dimensions)

---

### 1. Data Leakage

#### Issue 1.1: Target Leakage via Contemporaneous Momentum Feature in Freight Forecaster
* **Severity:** **CRITICAL**
* **File:** [`src/models/xgboost_forecaster.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/xgboost_forecaster.py#L111-L126) & [`src/models/forecast_features.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/forecast_features.py#L71-L76)
* **Lines:** `xgboost_forecaster.py:113`, `forecast_features.py:74`
* **Problem:** In `_prepare_features()`, the momentum feature is computed as:
  $$\text{momentum}_p[t] = y[t] - y[t-p]$$
  When fitting `model_p50.fit(X, y)` with $y[t] = \text{freight\_rate}[t]$, $X[t]$ includes `momentum_7` and `lag_7` ($y[t-7]$). Because $y[t] \equiv \text{momentum}_7 + \text{lag}_7$, the target variable $y[t]$ is algebraically leaked directly into the training feature matrix $X[t]$.
* **Why it matters:** The model learns a trivial identity function during training ($R^2 \approx 1.0$), resulting in severe generalization failure when deployed on unseen forward horizons where $y[t]$ is not yet known.
* **Recommended Fix:** Lag all momentum and difference features by at least 1 step:
  $$\text{momentum}_p[t] = y[t-1] - y[t-1-p]$$

---

### 2. Future Information Leakage

#### Issue 2.1: Full-Series Mean in Congestion Port Moving Average Baseline
* **Severity:** **HIGH**
* **File:** [`src/models/congestion_predictor.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/congestion_predictor.py#L126-L129)
* **Lines:** 127–128
* **Problem:** `self.port_averages = df_congestion.groupby(port_col)[target_col].mean().to_dict()` computes the global mean across the entire historical dataframe without respecting evaluation cutoff dates.
* **Why it matters:** In backtesting, future port congestion levels leak into early backtest periods, creating an unrealistic baseline.
* **Recommended Fix:** Compute expanding or rolling window averages strictly up to evaluation timestamp $T$.

---

### 3. Hard-Coded Business Assumptions

#### Issue 3.1: Synthetic Feature Manufacture in Congestion Predictor Inference
* **Severity:** **HIGH**
* **File:** [`src/models/congestion_predictor.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/congestion_predictor.py#L361-L383)
* **Lines:** 361–378
* **Problem:** During inference in `predict_congestion()`, rather than extracting actual trailing port queue observations, rolling statistics and queue counts are synthetically synthesized using hardcoded multipliers:
  ```python
  "vessels_waiting": max(2, int(round(hist_avg * 2.0))),
  "berth_occupancy_pct": min(95.0, max(60.0, 70.0 + hist_avg * 4.0)),
  "congestion_index": hist_avg * 0.65,
  ```
* **Why it matters:** The model does not respond to real-time live port queue changes; it generates predictions from static formulas.
* **Recommended Fix:** Query trailing 7/14/30-day records from `congestion.csv` or port call databases when available, falling back to heuristic multipliers only when no history exists.

#### Issue 3.2: Arbitrary Maximum DWT Formula in Port Constraints
* **Severity:** **HIGH**
* **File:** [`src/api/routes/optimize.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/api/routes/optimize.py#L58-L62)
* **Line:** 58
* **Problem:** Port maximum deadweight is computed as:
  `max_dwt = orig_info.max_draft_m * 10000`
* **Why it matters:** Maximum permissible deadweight is a structural and bathymetric property of port berths, not draft $\times 10,000$. For Haldia (`max_draft=8.5m`), this assigns 85,000 DWT (actual limit is 55,000 DWT). For Newcastle (`max_draft=15.2m`), this assigns 152,000 DWT (actual limit is 200,000+ DWT).
* **Recommended Fix:** Read `max_dwt` directly from port metadata (`ports.csv`).

---

### 4. Synthetic Data Presented as Real

#### Issue 4.1: Hardcoded Source Tag in Congestion Prediction Dataclass
* **Severity:** **MEDIUM**
* **File:** [`src/models/congestion_predictor.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/congestion_predictor.py#L60)
* **Line:** 60
* **Problem:** `CongestionPrediction` defines default `data_source: str = "PORT_AUTHORITY_HISTORICAL"`, even when predictions are generated from synthetic demo weights or Erlang-C heuristics.
* **Why it matters:** Violates the project's strict policy against misrepresenting synthetic data as official port authority records.
* **Recommended Fix:** Set `data_source` dynamically to `"CALIBRATED_SYNTHETIC_MODEL"` or `"MOCK_BENCHMARK"`.

---

### 5. Incorrect Maritime Calculations & Cost Double-Counting

#### Issue 5.1: Double-Counting Bunker Fuel Expenses Under Voyage Chartering
* **Severity:** **CRITICAL**
* **File:** [`src/economics/voyage_cost.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/economics/voyage_cost.py#L326-L335) & [`src/backtesting/optimization_backtester.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/backtesting/optimization_backtester.py#L263)
* **Lines:** `voyage_cost.py:326-335`, `optimization_backtester.py:263`
* **Problem:** Total delivered voyage cost is calculated as:
  $$\text{TotalCost} = \text{FreightCost} + \text{BunkerCost} + \text{PortCost} + \text{WaitingCost} + \text{Demurrage} + \dots$$
  Under standard commercial dry-bulk **Voyage Charters** (FIOST or Gross Terms), the freight rate ($/MT) paid to the shipowner **already includes the vessel's bunker fuel consumption and capital operating costs**. By adding `bunker_cost` ($354,297 for Capesize) on top of `freight_cost` ($1,413,000), the charterer is billed for fuel twice.
* **Why it matters:** Overstates delivered charter costs by 15% to 25%, distorting commercial cost comparisons against benchmark baselines.
* **Recommended Fix:** 
  - For **Voyage Charters**: `bunker_cost` should only be added if there is a specific Bunker Adjustment Factor (BAF) clause, or set `bunker_exposure_pct = 0.0`.
  - For **Time Charters**: Charterer pays `DailyHire * Days + BunkerCost + PortCost`.

#### Issue 5.2: Double-Counting Waiting Days (Anchorage Hire + Demurrage)
* **Severity:** **HIGH**
* **File:** [`src/economics/voyage_cost.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/economics/voyage_cost.py#L287-L335)
* **Lines:** 287, 306, 330–331
* **Problem:** The calculation adds:
  `waiting_cost = waiting_days * vessel_daily_hire_cost_usd`
  AND
  `demurrage_exposure = excess_time * daily_demurrage_rate_usd`
* **Why it matters:** At anchorage, a voyage charterer only pays demurrage if laytime is exceeded. Charging vessel daily hire for idle days while also charging demurrage penalizes the same delay twice.
* **Recommended Fix:** Set `waiting_cost = 0.0` under voyage charters and let demurrage model the financial liability of delays.

---

### 6. Unit Conversion Errors

#### Issue 6.1: Static Barrel-to-Tonne Conversion Ratio for Bunker Fuel
* **Severity:** **LOW**
* **File:** [`src/models/forecast_features.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/forecast_features.py#L156)
* **Line:** 156
* **Problem:** `piv_bk["crude_oil_price"] = piv_bk["bunker_price"] / 7.4` assumes an invariant conversion factor of 7.4 barrels per metric tonne.
* **Why it matters:** Specific gravity of heavy fuel oil (VLSFO ~0.95 kg/l) yields ~6.65 bbl/MT, while crude oils range from 7.1 to 7.6 bbl/MT. Using 7.4 for VLSFO skews the crude oil price proxy by ~11%.
* **Recommended Fix:** Use proper conversion factor ($6.65$ bbl/MT for VLSFO) or load crude oil prices directly from benchmark series.

---

### 7. Port Constraint Errors

#### Issue 7.1: Zero Under-Keel Clearance (UKC) Safety Margin
* **Severity:** **HIGH**
* **File:** [`src/optimization/constraint_engine.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/optimization/constraint_engine.py#L91-L98) & [`src/optimization/port_compatibility.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/optimization/port_compatibility.py#L60)
* **Lines:** `constraint_engine.py:91-98`, `port_compatibility.py:60`
* **Problem:** Rejection occurs strictly when `vessel.draft > port.max_draft_m`. 
* **Why it matters:** International maritime navigation rules require a mandatory Under-Keel Clearance (UKC) buffer (typically 1.0m to 1.5m, or 10% of draft). Allowing a 14.5m draft vessel into a 14.5m draft port models a ship scraping the channel seabed.
* **Recommended Fix:** Enforce `vessel.draft + MIN_UKC_M <= port.max_draft_m` where `MIN_UKC_M = 1.0m`.

---

### 8. Vessel Feasibility Errors

#### Issue 8.1: Invariant Maximum Summer Draft Used for Short-Loaded Parcels
* **Severity:** **HIGH**
* **File:** [`src/optimization/constraint_engine.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/optimization/constraint_engine.py#L91-L98)
* **Lines:** 91–98
* **Problem:** Draft compatibility checks `v_spec.draft_max_m` (full-load summer draft), ignoring actual parcel tonnage loaded (`parcel_mt`).
* **Why it matters:** A Capesize loading only 100,000 MT has an actual arrival draft of ~11.5m (due to TPC immersion reduction), but is evaluated as if drawing its full 18.5m summer draft. This falsely eliminates short-loaded larger vessels at ports with 14–16m draft.
* **Recommended Fix:** Calculate dynamic sailing draft:
  $$\text{Draft}_{\text{dynamic}} = \text{Draft}_{\text{ballast}} + (\text{Draft}_{\text{scantling}} - \text{Draft}_{\text{ballast}}) \times \left(\frac{\text{ParcelWeight}}{\text{MaxDWT}}\right)$$

---

### 9. Freight Forecasting Problems

#### Issue 9.1: Frozen Feature Vectors in Multi-Step Recursive Forecasts
* **Severity:** **HIGH**
* **File:** [`src/models/xgboost_forecaster.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/xgboost_forecaster.py#L233-L244)
* **Lines:** 233–244
* **Problem:** In the recursive forecasting loop for horizons 1 to 30 days:
  Only `lag_1`..`lag_28` and `rolling_mean` are updated. Momentum features (`momentum_7`, `momentum_30`), rolling standard deviations (`rolling_std_7`, `rolling_std_28`), and calendar flags remain completely frozen at their day-0 values.
* **Why it matters:** At day 14 and day 30, the model feeds stale momentum and volatility signals, leading to distorted long-horizon forecast curves.
* **Recommended Fix:** Update rolling momentum, rolling std, and calendar date features dynamically inside the recursive step loop.

---

### 10. Overfitting

#### Issue 10.1: High Model Capacity with Dense Collinear Lags on Short Timeseries
* **Severity:** **MEDIUM**
* **File:** [`src/models/xgboost_forecaster.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/xgboost_forecaster.py#L43-L49)
* **Lines:** 43–47
* **Problem:** Default parameters configure `max_depth=5`, `n_estimators=120`, and 22 collinear features without L1 regularization (`reg_alpha=0`).
* **Why it matters:** On regional routes with only 100–300 historical trading days, deep decision trees overfit to short-term noise.
* **Recommended Fix:** Set `max_depth=3`, `n_estimators=60`, and introduce `reg_alpha=0.5`.

---

### 11. Incorrect Train/Test Methodology

#### Issue 11.1: Apparent In-Sample Evaluation Metrics in Congestion Predictor
* **Severity:** **HIGH**
* **File:** [`src/models/congestion_predictor.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/congestion_predictor.py#L292-L295)
* **Lines:** 292–295
* **Problem:** In `fit()`, MAE and RMSE are evaluated directly on the training set `X` (`preds_p50 = model_p50.predict(X)`).
* **Why it matters:** Reports apparent training fit error rather than out-of-sample generalization error, creating false confidence in model accuracy.
* **Recommended Fix:** Compute out-of-fold cross-validated or holdout temporal metrics.

---

### 12. Unrealistic Uncertainty Estimates

#### Issue 12.1: Static Fallback Uncertainty Ribbon Invariant to Forecast Horizon
* **Severity:** **MEDIUM**
* **File:** [`src/models/xgboost_forecaster.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/xgboost_forecaster.py#L87-L88) & [`src/models/xgboost_forecaster.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/models/xgboost_forecaster.py#L215-L219)
* **Lines:** 87–88, 215–219
* **Problem:** Fallback quantile spread uses constant offsets:
  `residual_p10_diff = -0.8`, `residual_p90_diff = +0.8` ($/MT).
* **Why it matters:** Forecast uncertainty remains constant ($\pm \$0.80$) at day 1 and day 30, violating the stochastic principle that forecast variance expands with horizon length ($\sigma_h \propto \sqrt{h}$).
* **Recommended Fix:** Scale fallback residual quantiles with horizon: $\Delta_h = \text{residual\_diff} \times \sqrt{h}$.

---

### 13. Optimization Objective Errors

#### Issue 13.1: Extremely Low Utilization Weight Distorts Fleet Choice
* **Severity:** **HIGH**
* **File:** [`src/optimization/vessel_scoring.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/optimization/vessel_scoring.py#L28-L32)
* **Line:** 31
* **Problem:** `utilization_weight` is set to only `0.05` (5%).
* **Why it matters:** A Capesize vessel sailing 45% empty (100k MT on 180k DWT) suffers almost no optimization penalty ($<0.3$ points), consistently beating 100% utilized Panamax/Supramax vessels due to uncalibrated economies of scale.
* **Recommended Fix:** Increase `utilization_weight` to $0.15$ and add an explicit deadweight freight penalty for utilization $<65\%$.

---

### 14. Double-Counting Costs

*(Documented under Category 5: Issues 5.1 and 5.2 — Bunker fuel and waiting anchorage hire).*

---

### 15. Incorrect Demurrage Calculations

#### Issue 15.1: Berth Handling Duration Omitted from Demurrage Calculation
* **Severity:** **CRITICAL**
* **File:** [`src/economics/demurrage_calculator.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/economics/demurrage_calculator.py#L60-L63) & [`src/economics/demurrage_calculator.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/economics/demurrage_calculator.py#L91-L96)
* **Lines:** 61–63, 91–96
* **Problem:** The calculation determines demurrage and despatch as:
  ```python
  demurrage = max(0, predicted_idle_days - allowed_laytime) * demurrage_rate
  despatch  = max(0, allowed_laytime - predicted_idle_days) * despatch_rate
  ```
  It compares **only the anchorage waiting days** (`predicted_idle_days`) against `allowed_laytime`, completely omitting the days required to load/discharge cargo at berth.
* **Why it matters:** If a vessel waits 1.5 days at anchorage and allowed laytime is 6 days, it awards 4.5 days of **despatch cash credit**, even if discharging took 5 days at berth (total time = 6.5 days, which should trigger 0.5 days of demurrage!).
* **Recommended Fix:** Total laytime used must equal `waiting_days + cargo_handling_days`.

#### Issue 15.2: Self-Negating Dynamic Laytime in Voyage Cost Engine
* **Severity:** **HIGH**
* **File:** [`src/economics/voyage_cost.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/economics/voyage_cost.py#L295)
* **Line:** 295
* **Problem:** When explicit handling rates are provided, laytime allowed is set to:
  `laytime_allowed = actual_operational_time + 1.0`
* **Why it matters:** This dynamically expands contractual laytime to match whatever time the port takes to handle cargo, plus a 1-day bonus. Slow port handling can never trigger demurrage.
* **Recommended Fix:** Contractual laytime must be fixed by contract or calculated using a standard charterparty handling rate benchmark: $\text{Laytime} = \text{Cargo} / \text{ContractedRate}$.

---

### 16. Risk Scoring Problems

#### Issue 16.1: Duplicate Case-Sensitive Keys in Risk Breakdown Output
* **Severity:** **LOW**
* **File:** [`src/risk/risk_engine.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/risk/risk_engine.py#L180-L210) & [`scripts/run_demo.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/scripts/run_demo.py#L280-L290)
* **Lines:** `run_demo.py:280-290`
* **Problem:** The risk breakdown dictionary stores both title-case keys (`"Market"`, `"Port Congestion"`) and lowercase keys (`"market"`, `"port_congestion"`), causing duplicate bar printouts in CLI dashboards.
* **Why it matters:** Degrades presentation quality during live demonstrations.
* **Recommended Fix:** Standardize risk dimension keys to snake_case or Title Case consistently.

---

### 17. Monte Carlo Assumptions

#### Issue 17.1: Zero Correlation Between Freight Rate and Bunker Price
* **Severity:** **HIGH**
* **File:** [`src/risk/monte_carlo.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/risk/monte_carlo.py#L167-L180)
* **Lines:** 167, 180
* **Problem:** Freight rates and bunker prices are sampled independently using separate random lognormal draws.
* **Why it matters:** In global shipping, bunker fuel is ~50% of vessel operating costs; freight and bunker prices exhibit strong positive correlation ($r \approx 0.60–0.75$). Independent sampling distorts tail Value-at-Risk (VaR) estimates.
* **Recommended Fix:** Sample freight and bunker price perturbations jointly using a bivariate Gaussian copula with $\rho = 0.65$.

---

### 18. API Inconsistencies

#### Issue 18.1: Inconsistent Cargo Field Naming Across Production Endpoints
* **Severity:** **MEDIUM**
* **File:** [`src/api/serializers.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/api/serializers.py) & [`src/api/routes/`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/api/routes/)
* **Problem:** Cargo tonnage is inconsistently named across endpoints:
  - `POST /api/v1/recommend`: `cargo_tonnage`
  - `POST /api/v1/optimize/voyage`: `cargo_quantity_t`
  - `POST /api/v1/analyze-voyage`: `cargo_quantity`
* **Why it matters:** Confuses client integrations and frontend developers consuming multiple endpoints.
* **Recommended Fix:** Provide alias support in Pydantic schemas so all three variants are accepted everywhere.

---

### 19. Database Inconsistencies

#### Issue 19.1: PostgreSQL DDL Mismatch with Default SQLite / Mock DB Run Modes
* **Severity:** **LOW**
* **File:** [`src/data/db.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/data/db.py) & [`alembic/versions/`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/alembic/versions/)
* **Problem:** Database models define PostgreSQL `JSONB` fields which error if SQLite is used as a standalone database file without JSON fallback wrappers.
* **Why it matters:** Running migration scripts against local SQLite environments can fail.
* **Recommended Fix:** Use SQLAlchemy's `JSON` type (which compiles to `JSONB` on Postgres and `TEXT` on SQLite).

---

### 20. Frontend Hard-Coded Values & Catalog Mismatch

#### Issue 20.1: Non-Standard UN/LOCODE Port Identifiers in Frontend Dropdown
* **Severity:** **CRITICAL**
* **File:** [`frontend/src/pages/VoyagePlanner.tsx`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/frontend/src/pages/VoyagePlanner.tsx#L46-L54) & [`frontend/src/pages/VoyagePlanner.tsx`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/frontend/src/pages/VoyagePlanner.tsx#L263-L279)
* **Lines:** 49–50, 264–278
* **Problem:** The UI hardcodes port codes that do not match the backend database:
  - Frontend has: `INA_TAB` (Taboneo) -> Backend expects: `IDN_TAB`
  - Frontend has: `ZAF_RB` (Richards Bay) -> Backend expects: `ZAF_RIC`
  - Frontend has: `IND_DHA` (Dhamra) -> Backend expects: `IND_DHM`
  - Frontend has: `IND_VIS` (Visakhapatnam) -> Backend expects: `IND_VZG`
  - Frontend has: `IND_HAL` (Haldia) -> Backend expects: `IND_HLD`
  - Frontend has: `IND_GAN` (Gangavaram) -> Backend expects: `IND_GVM`
* **Why it matters:** Submitting the default frontend form sends `INA_TAB` and `IND_DHA` to the API. Because these codes are unrecognized, the backend treats them as unknown ports and falls back to default coordinates, producing inaccurate route distances and draft checks.
* **Recommended Fix:** Update dropdown `<option>` values to match backend UN/LOCODE keys, or fetch the port list dynamically from `GET /api/v1/ports`.

---

### 21. Security Problems

#### Issue 21.1: Default Database Password in Production Configuration
* **Severity:** **MEDIUM**
* **File:** [`src/utils/config.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/utils/config.py#L35)
* **Line:** 35
* **Problem:** `postgres_password: str = "changeme_in_production"` is checked into version control.
* **Why it matters:** Security vulnerability if deployed in shared or public cloud environments.
* **Recommended Fix:** Require `POSTGRES_PASSWORD` to be explicitly provided via environment variables in production.

---

### 22. Missing Validation

#### Issue 22.1: Missing Identity Validation Between Origin and Destination Ports
* **Severity:** **LOW**
* **File:** [`src/api/serializers.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/api/serializers.py)
* **Problem:** None of the request schemas validate that `origin_port_id != destination_port_id`.
* **Why it matters:** Submitting the same origin and destination produces distance = 0 nm and divides by zero in duration estimators.
* **Recommended Fix:** Add a `@model_validator(mode="after")` verifying origin and destination ports differ.

---

### 23. Performance Bottlenecks

#### Issue 23.1: Synchronous CPU-Bound Pipeline Blocking Async FastAPI Event Loop
* **Severity:** **HIGH**
* **File:** [`src/api/routes/recommend.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/src/api/routes/recommend.py#L195-L225)
* **Lines:** 195, 214–225
* **Problem:** The endpoint is declared `async def get_recommendation()`, but executes `engine.evaluate()` synchronously on the main event loop.
* **Why it matters:** Running heavy XGBoost, fleet ranking, and 5,000 Monte Carlo runs inside an `async def` function blocks the Python event loop for 200–500ms, halting all concurrent HTTP traffic.
* **Recommended Fix:** Change `async def` to regular `def` (FastAPI automatically runs sync endpoints in a background threadpool) or use `await anyio.to_thread.run_sync(engine.evaluate, engine_inputs)`.

---

### 24. Test Gaps

#### Issue 24.1: Missing Correlation Tests for Monte Carlo Joint Distributions
* **Severity:** **MEDIUM**
* **File:** [`tests/unit/test_risk_monte_carlo.py`](file:///Users/tjeyesh/Personal%20Folder/Work/sih2026/charter-ai/tests/unit/test_risk_monte_carlo.py)
* **Problem:** Unit tests verify cost quantiles exist ($P_{10} \le P_{50} \le P_{90}$) but do not test covariance or realistic joint distributions between simulated variables.
* **Why it matters:** Unrealistic independent variable draws pass test assertions without validation.
* **Recommended Fix:** Add unit tests computing Pearson correlation between simulated freight and bunker price vectors.

---

## Top 10 Issues to Fix Before SIH

These 10 issues represent the highest-priority architectural, mathematical, and user-facing vulnerabilities that must be rectified before final SIH jury demonstration:

| Rank | Issue Title | Severity | Location | Immediate Operational Impact |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Target Leakage in Freight Feature Builder** | **CRITICAL** | `src/models/xgboost_forecaster.py:113` | Contemporaneous momentum feature leaks target $y[t]$ directly into training matrix. |
| **2** | **Frontend-Backend Port Code Mismatch** | **CRITICAL** | `frontend/src/pages/VoyagePlanner.tsx:49, 264` | Frontend sends `INA_TAB`, `IND_DHA`, `IND_GAN`; backend fails to resolve bathymetric limits. |
| **3** | **Double-Counting Bunker Costs in Voyage Charters** | **CRITICAL** | `src/economics/voyage_cost.py:328` | Ocean freight rate already covers fuel; adding bunker cost inflates voyage cost by $350k+. |
| **4** | **Demurrage Calculator Omitting Berth Handling Days** | **CRITICAL** | `src/economics/demurrage_calculator.py:61` | Compares only anchorage waiting against laytime, awarding false despatch bonuses. |
| **5** | **Double-Counting Waiting Costs (Hire + Demurrage)** | **HIGH** | `src/economics/voyage_cost.py:287, 306` | Charges both daily vessel hire and demurrage liability for the same anchorage delay days. |
| **6** | **Synchronous Blocking on FastAPI Async Event Loop** | **HIGH** | `src/api/routes/recommend.py:195` | `async def` with synchronous CPU-bound pipeline blocks all concurrent requests for 500ms. |
| **7** | **Zero Under-Keel Clearance (UKC) Allowed** | **HIGH** | `src/optimization/constraint_engine.py:91` | Allows vessel draft equal to port depth with 0.0m safety margin, violating maritime standards. |
| **8** | **Static Full-Load Draft Used for Short-Loaded Ships** | **HIGH** | `src/optimization/constraint_engine.py:91` | Uses 18.5m summer draft for Capesize with 100k MT parcel, ignoring real ~11.5m dynamic draft. |
| **9** | **Arbitrary Port Max DWT Formula (`draft * 10000`)** | **HIGH** | `src/api/routes/optimize.py:58` | Computes max port deadweight from draft rather than reading actual structural quay limits. |
| **10** | **Independent Freight & Bunker Monte Carlo Draws** | **HIGH** | `src/risk/monte_carlo.py:167, 180` | Simulates bunker and freight as independent variables, miscalculating joint tail risk. |

---

## Conclusion & Next Steps

CharterAI V2 has established an exceptional technological foundation, featuring working FastAPI endpoints, modern React visualizations, and comprehensive deterministic pipelines. Resolving the Top 10 issues will elevate the codebase from a promising hackathon prototype to an architecturally rigorous, commercially credible maritime platform ready for SIH victory.
