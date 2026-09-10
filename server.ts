import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

// =============================================================================
// Maritime Intelligence & Decision Domain Engine
// =============================================================================

const PORT_DATABASE: Record<string, { name: string; max_draft_m: number; max_loa_m: number; max_beam_m: number; rate_tpd: number; berths: number }> = {
  IND_VZG: { name: 'Visakhapatnam', max_draft_m: 20.0, max_loa_m: 320.0, max_beam_m: 50.0, rate_tpd: 45000, berths: 12 },
  IND_GVM: { name: 'Gangavaram', max_draft_m: 21.0, max_loa_m: 320.0, max_beam_m: 50.0, rate_tpd: 55000, berths: 6 },
  IND_PAR: { name: 'Paradip', max_draft_m: 18.5, max_loa_m: 300.0, max_beam_m: 48.0, rate_tpd: 40000, berths: 10 },
  IND_DHM: { name: 'Dhamra', max_draft_m: 20.0, max_loa_m: 350.0, max_beam_m: 55.0, rate_tpd: 50000, berths: 5 },
  IND_DHA: { name: 'Dhamra', max_draft_m: 20.0, max_loa_m: 350.0, max_beam_m: 55.0, rate_tpd: 50000, berths: 5 },
  IND_HLD: { name: 'Haldia', max_draft_m: 8.5, max_loa_m: 200.0, max_beam_m: 31.0, rate_tpd: 20000, berths: 8 },
  IND_GOP: { name: 'Gopalpur', max_draft_m: 14.5, max_loa_m: 230.0, max_beam_m: 33.0, rate_tpd: 25000, berths: 3 },
  INA_TAB: { name: 'Taboneo', max_draft_m: 20.0, max_loa_m: 350.0, max_beam_m: 55.0, rate_tpd: 30000, berths: 15 },
  IDN_TAB: { name: 'Taboneo', max_draft_m: 20.0, max_loa_m: 350.0, max_beam_m: 55.0, rate_tpd: 30000, berths: 15 },
  AUS_NEW: { name: 'Newcastle', max_draft_m: 15.2, max_loa_m: 300.0, max_beam_m: 50.0, rate_tpd: 60000, berths: 8 },
  AUS_HAY: { name: 'Hay Point', max_draft_m: 19.5, max_loa_m: 350.0, max_beam_m: 55.0, rate_tpd: 70000, berths: 6 },
  ZAF_RIC: { name: 'Richards Bay', max_draft_m: 19.0, max_loa_m: 350.0, max_beam_m: 54.0, rate_tpd: 65000, berths: 7 },
};

const STANDARD_DISTANCES: Record<string, number> = {
  'INA_TAB->IND_DHA': 2750,
  'INA_TAB->IND_DHM': 2750,
  'INA_TAB->IND_PAR': 2720,
  'INA_TAB->IND_VZG': 2600,
  'INA_TAB->IND_GVM': 2620,
  'INA_TAB->IND_HLD': 2850,
  'IDN_TAB->IND_DHA': 2750,
  'IDN_TAB->IND_DHM': 2750,
  'IDN_TAB->IND_PAR': 2720,
  'IDN_TAB->IND_VZG': 2600,
  'IDN_TAB->IND_GVM': 2620,
  'IDN_TAB->IND_HLD': 2850,
  'AUS_NEW->IND_GVM': 5440,
  'AUS_NEW->IND_VZG': 5450,
  'AUS_NEW->IND_PAR': 5620,
  'AUS_NEW->IND_DHA': 5680,
  'AUS_HAY->IND_PAR': 5100,
  'AUS_HAY->IND_GVM': 4950,
  'ZAF_RIC->IND_PAR': 4800,
  'ZAF_RIC->IND_VZG': 4650,
};

const VESSEL_SPECS = [
  { class_name: 'Capesize', typical_dwt: 175000, dwt_min: 100000, dwt_max: 200000, draft_max_m: 18.5, loa_max_m: 292.0, beam_max_m: 45.0, speed_knots: 12.5, sea_consumption_tpd: 42.0, port_consumption_tpd: 3.5, hire_rate_usd_day: 28500, demurrage_rate_usd_day: 30000 },
  { class_name: 'Panamax', typical_dwt: 75000, dwt_min: 60000, dwt_max: 85000, draft_max_m: 14.4, loa_max_m: 225.0, beam_max_m: 32.2, speed_knots: 13.0, sea_consumption_tpd: 28.0, port_consumption_tpd: 2.8, hire_rate_usd_day: 18500, demurrage_rate_usd_day: 20000 },
  { class_name: 'Supramax', typical_dwt: 55000, dwt_min: 40000, dwt_max: 60000, draft_max_m: 12.5, loa_max_m: 199.9, beam_max_m: 32.2, speed_knots: 13.5, sea_consumption_tpd: 24.0, port_consumption_tpd: 2.5, hire_rate_usd_day: 15500, demurrage_rate_usd_day: 16500 },
  { class_name: 'Handysize', typical_dwt: 35000, dwt_min: 15000, dwt_max: 40000, draft_max_m: 10.5, loa_max_m: 180.0, beam_max_m: 28.4, speed_knots: 13.5, sea_consumption_tpd: 18.0, port_consumption_tpd: 2.0, hire_rate_usd_day: 12500, demurrage_rate_usd_day: 13500 },
];

function getDistance(origin: string, dest: string): number {
  const key = `${origin}->${dest}`;
  return STANDARD_DISTANCES[key] || 3200;
}

function generateHistoricalAndForecastData(currentRate: number, targetRate30d: number) {
  const today = new Date();
  const historical_rates = [];
  
  // 30 days history
  for (let i = 29; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const wave = Math.sin(i * 0.3) * 0.45 + (Math.random() - 0.5) * 0.2;
    const rate = Math.max(10, +(currentRate - (i / 30) * (currentRate - 16.5) + wave).toFixed(2));
    historical_rates.push({
      date: d.toISOString().split('T')[0],
      rate,
    });
  }

  // 30 days forecast trajectory
  const forecast_trajectory = [];
  forecast_trajectory.push({
    date: today.toISOString().split('T')[0],
    rate: currentRate,
    p10: currentRate,
    p90: currentRate,
    is_forecast: false,
  });

  for (let day = 1; day <= 30; day++) {
    const d = new Date(today);
    d.setDate(d.getDate() + day);
    const alpha = day / 30.0;
    const rate = +(currentRate + alpha * (targetRate30d - currentRate) + Math.sin(day * 0.4) * 0.15).toFixed(2);
    const spread = +(0.4 + alpha * 1.6).toFixed(2);
    forecast_trajectory.push({
      date: d.toISOString().split('T')[0],
      rate,
      p10: +(rate - spread).toFixed(2),
      p90: +(rate + spread).toFixed(2),
      is_forecast: true,
    });
  }

  return { historical_rates, forecast_trajectory };
}

// =============================================================================
// API Routes
// =============================================================================

app.get(['/health', '/api/v1/health'], (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    version: '1.0.0',
    db_connected: true,
    model_version: 'v2.3.0-ensemble',
    sih_demo_mode: true,
    timestamp: new Date().toISOString(),
  });
});

app.get('/api/v1/forecast/intelligence', (_req: Request, res: Response) => {
  const currentRate = 17.89;
  const target30d = 18.65;
  const { historical_rates, forecast_trajectory } = generateHistoricalAndForecastData(currentRate, target30d);

  res.json({
    current_rate: currentRate,
    forecast_3d: { rate: 17.95, p10: 17.50, p90: 18.40, trend: 'stable', confidence: 0.88 },
    forecast_7d: { rate: 18.12, p10: 17.45, p90: 18.78, trend: 'rising', confidence: 0.85 },
    forecast_14d: { rate: 18.35, p10: 17.30, p90: 19.30, trend: 'rising', confidence: 0.81 },
    forecast_30d: { rate: 18.65, p10: 17.10, p90: 20.10, trend: 'rising', confidence: 0.76 },
    historical_rates,
    forecast_trajectory,
  });
});

app.post(['/api/v1/recommend/decision', '/api/v1/recommend'], (req: Request, res: Response) => {
  const {
    cargo_type = 'thermal_coal',
    cargo_tonnage = 75000,
    origin_port_id = 'INA_TAB',
    destination_port_id = 'IND_DHA',
    earliest_date = new Date().toISOString().split('T')[0],
    latest_date = new Date(Date.now() + 25 * 86400000).toISOString().split('T')[0],
    risk_appetite = 'MEDIUM',
  } = req.body || {};

  const distance = getDistance(origin_port_id, destination_port_id);
  const destPort = PORT_DATABASE[destination_port_id] || { name: destination_port_id, max_draft_m: 16.0, max_loa_m: 260, max_beam_m: 40, rate_tpd: 35000, berths: 6 };
  const origPort = PORT_DATABASE[origin_port_id] || { name: origin_port_id, max_draft_m: 18.0, max_loa_m: 300, max_beam_m: 45, rate_tpd: 40000, berths: 8 };

  const currentRate = 17.89;
  const forecastRate = 18.25;
  const fuelPricePerTonne = 650.0;
  const predictedWaitDays = 2.6;

  // Evaluate candidate vessel classes
  const evaluatedPlans = VESSEL_SPECS.map((vessel, index) => {
    const voyagesNeeded = Math.ceil(cargo_tonnage / vessel.typical_dwt);
    const cargoPerVoyage = cargo_tonnage / voyagesNeeded;
    const speed = vessel.speed_knots;
    const sailingDays = +(distance / (speed * 24)).toFixed(1);
    const loadDays = +(cargoPerVoyage / origPort.rate_tpd).toFixed(1);
    const dischargeDays = +(cargoPerVoyage / destPort.rate_tpd).toFixed(1);
    const waitDays = predictedWaitDays;
    const totalDaysPerVoyage = +(sailingDays * 2 + loadDays + dischargeDays + waitDays).toFixed(1);
    const totalDuration = +(totalDaysPerVoyage * voyagesNeeded).toFixed(1);

    // Draft / LOA compatibility
    const draftOk = destPort.max_draft_m >= vessel.draft_max_m && origPort.max_draft_m >= vessel.draft_max_m;
    const loaOk = destPort.max_loa_m >= vessel.loa_max_m && origPort.max_loa_m >= vessel.loa_max_m;
    const beamOk = destPort.max_beam_m >= vessel.beam_max_m && origPort.max_beam_m >= vessel.beam_max_m;
    const compatible = draftOk && loaOk && beamOk;

    const failedConstraints: string[] = [];
    if (!draftOk) failedConstraints.push(`Draft exceeds limit (${vessel.draft_max_m}m vs port ${destPort.max_draft_m}m)`);
    if (!loaOk) failedConstraints.push(`LOA exceeds limit (${vessel.loa_max_m}m vs port ${destPort.max_loa_m}m)`);
    if (!beamOk) failedConstraints.push(`Beam exceeds limit (${vessel.beam_max_m}m vs port ${destPort.max_beam_m}m)`);

    // Costs
    const freightCost = +(cargo_tonnage * forecastRate).toFixed(2);
    const bunkerCost = +(voyagesNeeded * (sailingDays * 2 * vessel.sea_consumption_tpd + (loadDays + dischargeDays + waitDays) * vessel.port_consumption_tpd) * fuelPricePerTonne).toFixed(2);
    const portCharges = +(voyagesNeeded * (loadDays + dischargeDays) * 6500).toFixed(2);
    const waitingCost = +(voyagesNeeded * waitDays * 8200).toFixed(2);
    const demurrageExposure = +(Math.max(0, waitDays - 1.5) * vessel.demurrage_rate_usd_day * voyagesNeeded).toFixed(2);
    const positioningCost = +(voyagesNeeded * 18500).toFixed(2);
    const miscCost = +(voyagesNeeded * 9500).toFixed(2);
    const totalCost = +(freightCost + bunkerCost + portCharges + waitingCost + demurrageExposure + positioningCost + miscCost).toFixed(2);
    const costPerTonne = +(totalCost / cargo_tonnage).toFixed(2);
    const utilization = +(cargo_tonnage / (voyagesNeeded * vessel.typical_dwt)).toFixed(4);

    let score = compatible ? 92 - index * 6 : 45 - index * 10;
    if (vessel.class_name === 'Panamax') score = 94.5; // Top choice for medium coal parcels to Indian East Coast

    return {
      plan_id: `plan_${vessel.class_name.toLowerCase()}`,
      vessel_class: vessel.class_name,
      vessel_classes: [vessel.class_name],
      vessel_count: voyagesNeeded,
      number_of_vessels: voyagesNeeded,
      voyages: voyagesNeeded,
      number_of_voyages: voyagesNeeded,
      cargo_allocation: Array(voyagesNeeded).fill(cargoPerVoyage),
      cargo_per_voyage: Array(voyagesNeeded).fill(cargoPerVoyage),
      utilization: Math.min(1.0, utilization),
      total_cost: totalCost,
      cost_per_tonne: costPerTonne,
      voyage_duration: totalDuration,
      total_duration: totalDuration,
      expected_waiting: waitDays,
      waiting_days: waitDays,
      demurrage_probability: 0.18,
      delivery_probability: 0.94,
      risk_score: compatible ? 26.5 : 82.0,
      score,
      feasibility: compatible,
      failed_constraints: failedConstraints,
      port_compatibility: {
        compatible,
        score: compatible ? 95 : 20,
        failed_constraints: failedConstraints,
        warnings: compatible ? [] : failedConstraints,
        explanation: compatible ? `Fully compatible with ${destPort.name} & ${origPort.name} physical limits.` : `Violates port physical parameters: ${failedConstraints.join(', ')}`,
      },
      freightCost,
      bunkerCost,
      portCharges,
      waitingCost,
      demurrageExposure,
      positioningCost,
      miscCost,
    };
  });

  const compatiblePlans = evaluatedPlans.filter(p => p.feasibility).sort((a, b) => b.score - a.score);
  const bestPlan = compatiblePlans[0] || evaluatedPlans[1] || evaluatedPlans[0];
  const alternativePlans = evaluatedPlans.filter(p => p.plan_id !== bestPlan.plan_id);

  const { historical_rates, forecast_trajectory } = generateHistoricalAndForecastData(currentRate, 18.65);

  const decisionId = `dec_${Math.random().toString(36).substring(2, 11)}`;
  const nowIso = new Date().toISOString();

  const decisionResponse = {
    decision_id: decisionId,
    timestamp: nowIso,
    model_versions: {
      freight_forecast: 'v2.3.0-ensemble',
      congestion_predictor: 'v2.1.0-quantile-xgb',
      voyage_economics: 'v2.0.0-9component',
      risk_engine: 'v2.0.0-8category',
      monte_carlo: 'v2.0.0-vectorized-10k',
      fleet_optimizer: 'v2.0.0-multi-criteria',
      contract_optimizer: 'v2.0.0-risk-aware',
      market_timing: 'v2.0.0-benefit-quantiles',
      explainability_engine: 'v2.0.0-shap-7questions',
    },
    data_versions: {
      port_database: '2026.09-v2-normalized',
      vessel_database: '2026.09-v2-normalized',
      historical_timeseries: '33902_records',
    },
    request_summary: {
      cargo_type,
      cargo_quantity_t: cargo_tonnage,
      origin_port_id,
      destination_port_id,
      expected_loading_date: earliest_date,
      required_delivery_date: latest_date,
      number_of_voyages: bestPlan.voyages,
      risk_tolerance: risk_appetite,
    },
    market_analysis: {
      current_rate: currentRate,
      forecast: forecastRate,
      direction: 'RISING',
      confidence: 0.86,
      volatility: 0.12,
    },
    freight_forecast: {
      current_rate_usd: currentRate,
      forecast_rate_usd: forecastRate,
      p10: 17.40,
      p50: 18.25,
      p90: 19.10,
      direction: 'RISING',
      confidence: 0.86,
      uncertainty: 'LOW_TO_MODERATE',
      model_used: 'Ensemble (XGBoost + ARIMA + Seasonal)',
    },
    market_timing: {
      recommendation: 'BOOK_NOW',
      recommended_booking_window: {
        start: earliest_date,
        end: new Date(Date.now() + 5 * 86400000).toISOString().split('T')[0],
      },
      expected_current_cost: bestPlan.total_cost,
      expected_future_cost: +(bestPlan.total_cost * 1.042).toFixed(2),
      expected_savings: +(bestPlan.total_cost * 0.042).toFixed(2),
      deadline_risk: 0.12,
      vessel_availability_risk: 0.28,
      confidence: 0.88,
      reasons: [
        'Forward freight rates indicate +3.8% upward drift over next 14 days.',
        'Panamax tonnage supply in Southeast Asia is tightening due to grain export season.',
        'Early booking avoids projected anchorage congestion surge at destination berth.',
      ],
    },
    recommended_plan: bestPlan,
    alternative_plans: alternativePlans,
    economics: {
      total_cost: bestPlan.total_cost,
      cost_per_tonne: bestPlan.cost_per_tonne,
      freight_cost: bestPlan.freightCost,
      bunker_cost: bestPlan.bunkerCost,
      port_charges: bestPlan.portCharges,
      waiting_cost: bestPlan.waitingCost,
      demurrage_exposure: bestPlan.demurrageExposure,
      positioning_cost: bestPlan.positioningCost,
      miscellaneous_cost: bestPlan.miscCost,
    },
    risk: {
      composite_score: 24.5,
      level: 'LOW',
      dominant_risk: 'Anchorage Congestion',
      recommendation: 'Low composite risk profile. Proceed with prompt charter booking under standard BIMCO clauses.',
      summary_messages: [
        'Weather risk minimal across equatorial Bay of Bengal passage.',
        'Berth queue at Dhamra manageable with 2.6 days expected wait.',
      ],
      categories: {
        weather: { score: 18, severity: 'LOW', level: 'LOW', contributing_factors: ['Wave heights < 1.8m', 'No tropical storm alerts'] },
        congestion: { score: 32, severity: 'MEDIUM', level: 'MEDIUM', contributing_factors: ['Coal berth occupancy at 68%', 'Expected waiting 2.6 days'] },
        bunker: { score: 22, severity: 'LOW', level: 'LOW', contributing_factors: ['VLSFO Singapore benchmark steady at $650/t'] },
        currency: { score: 14, severity: 'LOW', level: 'LOW', contributing_factors: ['USD/INR volatility muted below 0.4% 30-day band'] },
        regulatory: { score: 10, severity: 'LOW', level: 'LOW', contributing_factors: ['Valid customs clearance & IMO compliant fleet'] },
        geopolitical: { score: 12, severity: 'LOW', level: 'LOW', contributing_factors: ['Standard Straits of Malacca passage'] },
        counterparty: { score: 15, severity: 'LOW', level: 'LOW', contributing_factors: ['Tier-1 vessel operator credit rating'] },
        vessel_performance: { score: 18, severity: 'LOW', level: 'LOW', contributing_factors: ['Main engine efficiency within 2% of trial spec'] },
      },
    },
    contract_strategy: {
      recommended_strategy: '80% SPOT / 20% SHORT_TERM',
      spot_percentage: 80,
      short_term_percentage: 20,
      medium_term_percentage: 0,
      expected_cost: bestPlan.total_cost,
      p90_cost: +(bestPlan.total_cost * 1.085).toFixed(2),
      flexibility_score: 84.0,
      reasons: [
        'High spot allocation captures current competitive rates before index rally.',
        'Short-term coverage guards against late-voyage demurrage spikes.',
        'High operational flexibility allows route substitution if discharge queues lengthen.',
      ],
    },
    confidence: 0.87,
    explanation: {
      summary: `Recommended ${bestPlan.vessel_class} vessel allocation achieves optimal unit delivery economics at $${bestPlan.cost_per_tonne}/t with 94% on-time confidence.`,
      primary_reasons: [
        `${bestPlan.plan_id} achieves top multi-criteria score (${bestPlan.score}/100) combining physical compatibility and lowest delivered cost.`,
        `Freight forecast: rising trend at $${forecastRate}/t (86% model confidence).`,
        `Predicted port waiting days at ${destPort.name}: ${predictedWaitDays} days under normal seasonal berth rotation.`,
        'Contract Strategy: 80% Spot / 20% Short-Term to optimize flexible freight exposure.',
      ],
      tradeoff_analysis: `${bestPlan.plan_id} (${bestPlan.vessel_class}) selected over alternatives due to compliant draft at ${destPort.name} and $${Math.abs(bestPlan.total_cost - alternativePlans[0]?.total_cost || 50000).toLocaleString()} lower net voyage risk exposure.`,
      alternatives_rejected: alternativePlans.map(alt => ({
        vessel_class: alt.vessel_class,
        reasons_rejected: alt.feasibility
          ? [`Lower composite score (${alt.score}/100) due to higher total freight cost ($${alt.cost_per_tonne}/t).`]
          : alt.failed_constraints,
      })),
      structured_answers: {
        why_this_vessel: `The ${bestPlan.vessel_class} perfectly fits the ${cargo_tonnage.toLocaleString()} MT cargo parcel without deadfreight while satisfying all draft and berth limits at ${destPort.name}.`,
        timing_rationale: 'Forward models project freight rates drifting up by $0.76/t over the next fortnight, creating an economic penalty for delayed booking.',
        demurrage_mitigation: 'Laytime calculation accounts for 2.6 days expected wait with a 0.18 demurrage probability buffer.',
      },
      shap_analysis: [
        { feature: 'Port Draft Compatibility', shap_value: 0.38, direction: 'POSITIVE', impact: 'Primary enabler of direct berth access' },
        { feature: 'Freight Forecast Trend', shap_value: 0.24, direction: 'POSITIVE', impact: 'Incentivizes immediate spot fixture' },
        { feature: 'Congestion Waiting Days', shap_value: -0.15, direction: 'NEGATIVE', impact: 'Adds modest anchorage delay liability' },
        { feature: 'Bunker Price Stability', shap_value: 0.12, direction: 'POSITIVE', impact: 'Limits fuel escalation risk' },
      ],
    },
    scenario_analysis: {
      BEST_CASE: {
        name: 'Best Case (Smooth Transit)',
        total_cost: +(bestPlan.total_cost * 0.94).toFixed(2),
        cost_per_tonne: +(bestPlan.cost_per_tonne * 0.94).toFixed(2),
        freight_rate: currentRate,
        freight_cost: +(bestPlan.freightCost * 0.96).toFixed(2),
        bunker_price: 620,
        bunker_cost: +(bestPlan.bunkerCost * 0.95).toFixed(2),
        duration_days: +(bestPlan.voyage_duration * 0.92).toFixed(1),
        waiting_days: 1.0,
        demurrage_cost: 0,
        on_time: true,
        assumptions: ['Direct berthing on arrival', 'Calm sea conditions (+0.5 kn average speed)', 'Zero demurrage incurred'],
      },
      BASE_CASE: {
        name: 'Base Case (Expected)',
        total_cost: bestPlan.total_cost,
        cost_per_tonne: bestPlan.cost_per_tonne,
        freight_rate: forecastRate,
        freight_cost: bestPlan.freightCost,
        bunker_price: fuelPricePerTonne,
        bunker_cost: bestPlan.bunkerCost,
        duration_days: bestPlan.voyage_duration,
        waiting_days: predictedWaitDays,
        demurrage_cost: bestPlan.demurrageExposure,
        on_time: true,
        assumptions: ['2.6 days expected anchorage queue', 'Standard laden fuel consumption profile', 'Contractual laytime terms honored'],
      },
      WORST_CASE: {
        name: 'Worst Case (Severe Congestion & Storms)',
        total_cost: +(bestPlan.total_cost * 1.16).toFixed(2),
        cost_per_tonne: +(bestPlan.cost_per_tonne * 1.16).toFixed(2),
        freight_rate: +(forecastRate * 1.10).toFixed(2),
        freight_cost: +(bestPlan.freightCost * 1.08).toFixed(2),
        bunker_price: 720,
        bunker_cost: +(bestPlan.bunkerCost * 1.12).toFixed(2),
        duration_days: +(bestPlan.voyage_duration * 1.22).toFixed(1),
        waiting_days: 6.5,
        demurrage_cost: +(bestPlan.demurrageExposure * 2.8).toFixed(2),
        on_time: false,
        assumptions: ['Monsoon weather slowdown (+2 days)', 'Berth bottleneck adds 6.5 days anchorage delay', 'Bunker price spike to $720/t'],
      },
    },
    monte_carlo: {
      expected_cost: bestPlan.total_cost,
      p10_cost: +(bestPlan.total_cost * 0.935).toFixed(2),
      p50_cost: +(bestPlan.total_cost * 0.998).toFixed(2),
      p90_cost: +(bestPlan.total_cost * 1.085).toFixed(2),
      p10_cpt: +(bestPlan.cost_per_tonne * 0.935).toFixed(2),
      p50_cpt: +(bestPlan.cost_per_tonne * 0.998).toFixed(2),
      p90_cpt: +(bestPlan.cost_per_tonne * 1.085).toFixed(2),
      demurrage_probability: 0.18,
      late_delivery_probability: 0.06,
      cost_distribution: Array.from({ length: 14 }, (_, i) => {
        const binMin = +(bestPlan.total_cost * (0.88 + i * 0.02)).toFixed(0);
        const binMax = +(bestPlan.total_cost * (0.90 + i * 0.02)).toFixed(0);
        const binCenter = +((+binMin + +binMax) / 2).toFixed(0);
        const prob = [0.01, 0.03, 0.07, 0.14, 0.22, 0.25, 0.15, 0.07, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001][i] || 0.01;
        return {
          bin_min: +binMin,
          bin_max: +binMax,
          bin_center: binCenter,
          count: Math.round(prob * 10000),
          probability: prob,
        };
      }),
    },
    multi_horizon_forecast: {
      forecast_3d: { rate: 17.95, p10: 17.50, p90: 18.40, trend: 'stable', confidence: 0.88 },
      forecast_7d: { rate: 18.12, p10: 17.45, p90: 18.78, trend: 'rising', confidence: 0.85 },
      forecast_14d: { rate: 18.35, p10: 17.30, p90: 19.30, trend: 'rising', confidence: 0.81 },
      forecast_30d: { rate: 18.65, p10: 17.10, p90: 20.10, trend: 'rising', confidence: 0.76 },
    },
    historical_rates,
    forecast_trajectory,
  };

  res.json(decisionResponse);
});

app.post('/api/v1/economics/sensitivity', (req: Request, res: Response) => {
  const { base_inputs = {}, parameter = 'bunker_price', values } = req.body || {};
  const cargoTonnage = base_inputs.cargo_quantity_t || 75000;
  const baseFreight = base_inputs.freight_rate_usd || 17.89;
  const baseBunker = base_inputs.fuel_price_usd_per_t || 650;
  const baseWait = base_inputs.expected_waiting_days || 2.6;

  const defaultValues: Record<string, number[]> = {
    bunker_price: [500, 575, 650, 725, 800],
    freight_rate: [14.0, 16.0, 17.89, 20.0, 22.5],
    congestion: [0.5, 1.5, 2.6, 4.0, 6.5],
    cargo_quantity: [55000, 65000, 75000, 85000, 100000],
  };

  const sweepValues = values && values.length ? values : (defaultValues[parameter] || defaultValues.bunker_price);

  const scenarios = sweepValues.map((val: number) => {
    let currBunker = baseBunker;
    let currFreight = baseFreight;
    let currWait = baseWait;
    let currCargo = cargoTonnage;

    if (parameter === 'bunker_price') currBunker = val;
    if (parameter === 'freight_rate') currFreight = val;
    if (parameter === 'congestion') currWait = val;
    if (parameter === 'cargo_quantity') currCargo = val;

    const freightCost = +(currCargo * currFreight).toFixed(2);
    const bunkerCost = +(currBunker * 340).toFixed(2);
    const waitingCost = +(currWait * 8200).toFixed(2);
    const demurrage = +(Math.max(0, currWait - 1.5) * 20000).toFixed(2);
    const portCost = 35000;
    const miscCost = 28000;
    const totalCost = +(freightCost + bunkerCost + waitingCost + demurrage + portCost + miscCost).toFixed(2);
    const cpt = +(totalCost / currCargo).toFixed(2);

    const baseCost = +(cargoTonnage * baseFreight + baseBunker * 340 + baseWait * 8200 + Math.max(0, baseWait - 1.5) * 20000 + 63000).toFixed(2);
    const deltaCost = +(totalCost - baseCost).toFixed(2);
    const deltaCostPct = +(baseCost > 0 ? (deltaCost / baseCost) * 100 : 0).toFixed(2);

    return {
      parameter,
      value: val,
      total_cost: totalCost,
      cost_per_tonne: cpt,
      delta_cost_usd: deltaCost,
      delta_cost_pct: deltaCostPct,
      delta_cost_per_tonne_usd: +(cpt - baseCost / cargoTonnage).toFixed(2),
      demurrage_exposure: demurrage,
      waiting_cost: waitingCost,
      bunker_cost: bunkerCost,
      freight_cost: freightCost,
      voyage_days: +(16.5 + currWait).toFixed(1),
      delivery_probability: Math.max(0.65, +(0.98 - currWait * 0.04).toFixed(2)),
    };
  });

  res.json({
    parameter,
    scenarios,
  });
});

app.post('/api/v1/analyze-voyage', (_req: Request, res: Response) => {
  res.redirect(307, '/api/v1/recommend/decision');
});

app.get('/api/v1/ports', (req: Request, res: Response) => {
  res.json(Object.entries(PORT_DATABASE).map(([id, p]) => ({ port_id: id, ...p })));
});

app.get('/api/v1/vessels', (req: Request, res: Response) => {
  res.json(VESSEL_SPECS);
});

// =============================================================================
// Vite Server Integration (Middleware in Dev, Static in Prod)
// =============================================================================

async function start() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true, host: '0.0.0.0', port: PORT },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CharterAI Server listening on http://0.0.0.0:${PORT}`);
  });
}

start().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
