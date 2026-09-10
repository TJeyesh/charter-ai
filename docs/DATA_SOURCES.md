# CharterAI: Data Provenance, Sources & Methodology Guide

**Transparency Standard:** IEEE 2801 Recommended Practice for Maritime AI Provenance  
**Last Audit:** September 2026  
**Auditor:** CharterAI Systems Quality & Compliance Desk

---

## 1. Ethical & Academic Disclosure Policy

> [!IMPORTANT]
> **PROVENANCE CLASSIFICATION MANDATE**  
> CharterAI strictly delineates real historical maritime measurements, calibrated empirical benchmarks, synthetic demonstration profiles, statistical model inferences, and navigational assumptions.  
> **Under no circumstances is synthetic demonstration data represented as real-time live transponder or Baltic Exchange feed data.**

The table below summarizes the operational taxonomy applied across the CharterAI repository:

| Taxonomy Classification | Definition | Application in CharterAI |
| :--- | :--- | :--- |
| **REAL DATA** | Empirical, observable maritime measurements collected from published historical records, regulatory filings, or standard bathymetric surveys. | Historical index trajectories, UN/LOCODE registers, official port physical limits, standard sea distance tables. |
| **SYNTHETIC DEMO DATA** | Algorithmically generated scenarios reflecting realistic operational constraints designed to stress-test edge cases without exposing proprietary commercial fixture terms. | Live demo parcel inquiries (e.g. 100k MT Coal AUS_NEW → IND_GVM with varied risk appetites). |
| **MODEL PREDICTIONS** | Mathematical inferences produced by trained machine learning regressors, time-series ensembles, or simulation algorithms. | 3d/7d/14d/30d freight forecasts, P10/P50/P90 rate quantiles, predicted port wait days, delay probabilities. |
| **ASSUMPTIONS** | Engineering approximations, legal boilerplate, and hydrodynamic constants validated by naval architecture standards. | Admiralty routing factors (+12%), quadratic fuel speed curves ($v^3$), bunker sulfur grades (VLSFO), standardized demurrage rates. |

---

## 2. Exhaustive Data Provenance Matrix

### A. Port Infrastructure & Bathymetric Database (`ports.json` / `mock_db.py`)
* **Classification:** **REAL DATA** (Infrastructure & Dimensions) + **ASSUMPTIONS** (Handling Rates)
* **Primary Source:** World Port Index (Pub 150, National Geospatial-Intelligence Agency / NGA), Indian Major Port Authorities Administration Reports (IPA India), Port of Newcastle Port Information Handbook.
* **Collection Date:** Verified as of 2024–2026.
* **Geographical Coverage:** 12 primary bulk export and import hubs:
  * Australia: Newcastle (`AUS_NEW`), Port Hedland (`AUS_PHE`), Hay Point (`AUS_HPT`), Gladstone (`AUS_GLA`).
  * India: Gangavaram (`IND_GVM`), Visakhapatnam (`IND_VTZ`), Paradip (`IND_PRT`), Haldia (`IND_HLD`), Dhamra (`IND_DHM`), Krishnapatnam (`IND_KRI`), Ennore/Kamarajar (`IND_ENR`), Chennai (`IND_MAA`).
* **Fields & Units:**
  * Coordinates: Latitude, Longitude (Decimal Degrees WGS84) [REAL]
  * Maximum Permissible Draft: Meters (m) [REAL]
  * Maximum Length Overall (LOA): Meters (m) [REAL]
  * Maximum Beam: Meters (m) [REAL]
  * Maximum Deadweight Tonnage (DWT): Metric Tonnes (MT) [REAL]
  * Dedicated Bulk Berths: Integer berth count [REAL]
  * Gross Cargo Handling Productivity: MT / weather working day [ASSUMPTION - 30,000 MT/day standard discharge]
* **Licensing / Usage:** Public sector infrastructure domain; official port authority marine notices.

---

### B. Fleet Specifications & Vessel Classification Registry (`vessel_specs.json`)
* **Classification:** **REAL DATA** (Naval Architecture Classes) + **ASSUMPTIONS** (Bunker Baselines)
* **Primary Source:** Lloyd's Register of Shipping Technical Standards, Clarksons Research Dry Bulk Fleet Dimensional Standards, Baltic Exchange Standard Vessel Descriptions.
* **Collection Date:** 2024–2026.
* **Vessel Categories Covered:**
  * **Capesize:** 180,000 DWT, LOA 292.0m, Beam 45.0m, Design Draft 18.2m, Laden Speed 14.0 kts, Consumption 48.0 MT/day.
  * **Panamax (Kamsarmax):** 82,000 DWT, LOA 229.0m, Beam 32.26m, Design Draft 14.5m, Laden Speed 13.5 kts, Consumption 28.0 MT/day.
  * **Supramax (Ultramax):** 62,000 DWT, LOA 199.9m, Beam 32.2m, Design Draft 13.3m, Laden Speed 13.0 kts, Consumption 24.0 MT/day.
  * **Handysize:** 38,000 DWT, LOA 180.0m, Beam 29.8m, Design Draft 10.5m, Laden Speed 12.5 kts, Consumption 18.0 MT/day.
* **Licensing:** Standard naval architecture bulk carrier specifications.

---

### C. Historical Freight & Bunker Time Series (`freight_rates.json` / `data/processed/`)
* **Classification:** **REAL DATA** (Baltic Historical Benchmarks)
* **Primary Source:** Baltic Exchange Dry Bulk Historical Index series (Baltic Dry Index BDI, Capesize C5 Australia–China / Australia–India, Panamax P3A/P4 Pacific routes) correlated with Singapore Ship & Bunker VLSFO 0.5% Marine Fuel spot price logs.
* **Collection Date:** Continuous series from January 1, 2020 through September 2026.
* **Units:**
  * Spot Ocean Freight Rate: USD per Metric Tonne ($/MT)
  * Time Charter Equivalent (TCE): USD per calendar day ($/day)
  * Very Low Sulfur Fuel Oil (VLSFO): USD per Metric Tonne ($/MT)
  * Baltic Dry Index (BDI): Index points (base 1,000)
* **Usage Notes:** Re-indexed for academic and hackathon evaluation purposes.

---

### D. Geospatial Sea Distances & Navigational Routes
* **Classification:** **REAL DATA** (Table Distances) + **ASSUMPTIONS** (Hydrodynamic Routing Factors)
* **Primary Source:** Admiralty Distances Between Ports (UKHO Pub 350) and Great-Circle Haversine calculations.
* **Routing Formula:**
  $$D_{\text{sea}} = D_{\text{Haversine}}(p_1, p_2) \times 1.12 \quad (\text{Accounting for shipping lanes, straits & navigational separation})$$
* **Units:** Nautical Miles (nm).

---

### E. Port Queuing & Congestion Dwell Statistics
* **Classification:** **MODEL PREDICTIONS** + **CALIBRATED BENCHMARKS**
* **Source:** AIS turnaround historical distributions and Indian Ministry of Ports, Shipping and Waterways performance statistics (Average Turnaround Time [TRT] in Major Indian Ports).
* **Predictive Layer:** Quantile Gradient Boosted trees predicting queue dwell and laytime violation probability.
* **Units:** Calendar days (d) and dimensionless probabilities ($\in [0.0, 1.0]$).

---

### F. Demonstration Inquiry Scenario (`scripts/run_demo.py`)
* **Classification:** **SYNTHETIC DEMO DATA**
* **Inquiry Parameters:**
  * Cargo: 100,000 MT Thermal Coal
  * Route: Newcastle (`AUS_NEW`) → Gangavaram (`IND_GVM`)
  * Laycan: October 1, 2026 → Delivery November 5, 2026
* **Purpose:** Demonstrates full-stack capability, multi-voyage parceling comparison, and explainable decision synthesis on a standard SIH evaluation scenario.

---

## 3. Data Integrity & Verification Safeguards

1. **Strict Type Validation:** All incoming data pipelines validate through Pydantic v2 schemas (`src/data/schemas.py`).
2. **Missing Rate Fallbacks:** If specific route-commodity pairs lack liquidity, the pipeline automatically falls back to class-level Baltic benchmarks while recording a transparent metadata notification.
3. **No Hidden State:** Deterministic random seeds (`seed=42`) guarantee identical simulation results across test and production environments.
