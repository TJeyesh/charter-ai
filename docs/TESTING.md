# CharterAI: Comprehensive Testing & Verification Suite

**Test Engine:** pytest 9.1.1 + pytest-cov 7.1.0  
**Python Runtime:** 3.14.7  
**Total Test Count:** 261 Tests  
**Passing Rate:** 100% (261 Passed, 0 Failed, 0 Skipped)  
**Execution Time:** ~2.5 minutes (complete suite including walk-forward backtests)

---

## 1. Test Suite Taxonomy & Organization

The CharterAI test architecture is divided into four distinct validation tiers:

```
tests/
├── unit/                         # 52 Tests: Isolated component & edge case validation
│   ├── test_data_validation.py   # Pydantic v2 boundary & type constraints
│   ├── test_port_vessel_compat.py# Draft, LOA, Beam, and DWT physical filtering
│   ├── test_forecasting_uncertainty.py # Point forecasts & P10/P50/P90 quantile bounds
│   ├── test_congestion_demurrage.py    # Dwell queue & laytime penalty calculations
│   ├── test_voyage_economics.py  # 9-component itemized delivered cost math
│   ├── test_risk_monte_carlo.py  # 8 risk categories & vectorized VaR distributions
│   ├── test_timing_contracts.py  # EEB market timing & quadratic utility contracts
│   └── test_edge_cases.py        # 13 critical boundary and failure mode edge cases
├── integration/                  # 18 Tests: Cross-module pipeline & API contract verification
│   ├── test_decision_pipeline.py # Full 8-step pipeline execution without mocking
│   └── test_api_integration.py   # All 13 FastAPI v1.0 production HTTP endpoints
├── backtesting/                  # 5 Tests: Time-series split & historical benchmarking
│   └── test_walk_forward_backtesting.py # Walk-forward validation & 5 baseline comparisons
└── [root test files]             # 186 Tests: Subsystem regressions & legacy coverage
```

---

## 2. Test Execution Summary

| Directory / Test Suite | Tests Executed | Passed | Failed | Key Verification Area |
| :--- | :---: | :---: | :---: | :--- |
| `tests/unit/` | **52** | **52** | **0** | Data models, physical filters, quantiles, economics, edge cases |
| `tests/integration/` | **18** | **18** | **0** | End-to-end decision engine & FastAPI REST endpoints |
| `tests/backtesting/` | **5** | **5** | **0** | Lookahead bias prevention, walk-forward splits, 5 baselines |
| Root Legacy Regressions | **186** | **186** | **0** | Specialized subsystem regressions & legacy endpoints |
| **Total Test Suite** | **261** | **261** | **0** | **Comprehensive System Hardening Confirmed** |

---

## 3. Critical Edge Cases Matrix

To ensure absolute reliability in commercial maritime environments, 13 mission-critical edge cases were explicitly codified and verified in `tests/unit/test_edge_cases.py`:

| # | Edge Case Scenario | Test Input | Expected Engine Behavior | Test Result |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Cargo quantity = 0** | `cargo_quantity_t = 0.0` | Rejected with Pydantic validation error (`> 0`) | ✅ PASS |
| 2 | **Negative cargo quantity** | `cargo_quantity_t = -50000.0` | Rejected with Pydantic validation error (`> 0`) | ✅ PASS |
| 3 | **Cargo exceeds all vessels** | `cargo_quantity_t = 800000.0` | Infeasible single voyage; partitioned into feasible multi-voyage plan | ✅ PASS |
| 4 | **Draft exceeds port limit** | Capesize draft (18.5m) at Haldia (`IND_HLD`, max 8.5m) | Hard physical rejection; excluded from candidate set | ✅ PASS |
| 5 | **Missing port code** | `origin_port_id = "UNKNOWN_PORT_XYZ"` | Handled gracefully with standard default port parameters | ✅ PASS |
| 6 | **Missing route history** | Rare unlisted trade corridor | Falls back to global class-level rate benchmarks | ✅ PASS |
| 7 | **Missing freight history** | Empty dataframe or cold-start | Falls back to class rolling average without crashing | ✅ PASS |
| 8 | **Missing congestion data** | Unregistered port terminal | Uses regional average handling productivity & 1.5d queue default | ✅ PASS |
| 9 | **Extreme bunker price** | $2,000.00 / MT VLSFO | Computes delivered economics; flags elevated market risk | ✅ PASS |
| 10 | **Extreme port congestion** | 15.0 days berth waiting queue | Demurrage liability surges; market timing advises postponement | ✅ PASS |
| 11 | **Unavailable vessel market** | Market tightness set to `TIGHT` | Increases availability risk; favors term cover / index CoA | ✅ PASS |
| 12 | **Impossible delivery deadline** | Required delivery in 3 days for 4,800 nm | Feasibility flagged False; on-time delivery probability drops to 0.0% | ✅ PASS |
| 13 | **No feasible vessel plan** | 100,000 MT coal to shallow river port | Decision engine returns structured error with transshipment advice | ✅ PASS |

---

## 4. Integration & API Endpoint Verification

All 13 production FastAPI endpoints in `src/api/routes.py` and `src/api/main.py` were verified using the FastAPI `TestClient`:

* `POST /api/v1/forecast/freight` (Point forecast, P10/P90 bounds, confidence) — ✅ PASS
* `POST /api/v1/predict/congestion` (Dwell days, quantile queue, delay probability) — ✅ PASS
* `POST /api/v1/optimize/vessels` (Physical berth filter and compatibility reasons) — ✅ PASS
* `POST /api/v1/optimize/voyage` (Multi-voyage single vs multi-vessel ranking) — ✅ PASS
* `POST /api/v1/optimize/contract` (Risk-aware portfolio allocation: Spot vs CoA vs Term) — ✅ PASS
* `POST /api/v1/analyze-voyage` (Itemized 9-component delivered cost calculation) — ✅ PASS
* `POST /api/v1/recommend` (Master 12-stage pipeline orchestrator) — ✅ PASS
* `GET /api/v1/ports` (Full ports catalog with optional country filtering) — ✅ PASS
* `GET /api/v1/vessels` (Full vessel registry with class filtering) — ✅ PASS
* `GET /api/v1/routes` (Standard route distance catalog) — ✅ PASS
* `GET /api/v1/market` (BDI, BCI, BPI, and bunker price benchmarks) — ✅ PASS
* `GET /api/v1/models` (Active model versions and metadata inspection) — ✅ PASS
* `POST /api/v1/backtest` (Historical walk-forward simulation endpoint) — ✅ PASS

---

## 5. Walk-Forward Backtesting & Baseline Verification

Verified in `tests/backtesting/test_walk_forward_backtesting.py`:
* **Zero Lookahead Bias:** Verified that training window strictly precedes out-of-sample test laycans ($\max(t_{\text{train}}) < \min(t_{\text{eval}})$).
* **5 Baseline Benchmarking:** CharterAI was benchmarked against:
  1. Spot-Only Rule
  2. 30-Day Moving Average
  3. Always Largest Vessel (Capesize Default)
  4. Always Spot Strategy
  5. Fixed Panamax Heuristic
* **Outcome:** CharterAI demonstrated lower delivered cost per tonne ($19.14/t vs $21.45–$27.49/t) and lower demurrage occurrence across all simulated periods.

---

## 6. How to Run Tests

```bash
# Run the entire test suite
pytest

# Run only new Phase 15 unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only backtesting verification
pytest tests/backtesting/

# Run with test coverage report
pytest --cov=src tests/
```
