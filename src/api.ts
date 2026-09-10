import axios from 'axios';

const API_BASE_URL = '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface CharterDecisionRequest {
  origin_port_id: string;
  destination_port_id: string;
  cargo_type: string;
  cargo_tonnage: number;
  earliest_date: string;
  latest_date: string;
  risk_appetite: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface MarketAnalysis {
  current_rate: number;
  forecast: number;
  direction: string;
  confidence: number;
  volatility: number;
}

export interface FreightForecast {
  current_rate_usd: number;
  forecast_rate_usd: number;
  p10: number;
  p50: number;
  p90: number;
  direction: string;
  confidence: number;
  uncertainty: string;
  model_used: string;
}

export interface MarketTiming {
  recommendation: 'BOOK_NOW' | 'WAIT' | 'MONITOR' | 'START_NEGOTIATION' | 'HYBRID_BOOKING' | 'HYBRID' | string;
  recommended_booking_window: {
    start: string;
    end: string;
  };
  expected_current_cost: number;
  expected_future_cost: number;
  expected_savings: number;
  deadline_risk: number;
  vessel_availability_risk: number;
  confidence: number;
  reasons: string[];
}

export interface PlanItem {
  plan_id?: string;
  vessel_class: string;
  vessel_classes?: string[];
  vessel_count?: number;
  number_of_vessels?: number;
  voyages?: number;
  number_of_voyages?: number;
  cargo_allocation?: number[];
  cargo_per_voyage?: number[];
  utilization: number;
  total_cost: number;
  cost_per_tonne: number;
  voyage_duration?: number;
  total_duration?: number;
  expected_waiting?: number;
  waiting_days?: number;
  demurrage_probability: number;
  delivery_probability: number;
  risk_score: number;
  score?: number;
  feasibility?: boolean;
  failed_constraints?: string[];
  port_compatibility?: {
    compatible: boolean;
    score: number;
    failed_constraints?: string[];
    warnings?: string[];
    explanation?: string;
  };
}

export interface EconomicsBreakdown {
  total_cost: number;
  cost_per_tonne: number;
  freight_cost: number;
  bunker_cost: number;
  port_charges: number;
  waiting_cost: number;
  demurrage_exposure: number;
  positioning_cost: number;
  miscellaneous_cost: number;
}

export interface RiskCategoryDetail {
  category?: string;
  score: number;
  severity: string;
  level: string;
  contributing_factors?: string[];
  warnings?: string[];
  reasoning?: string;
}

export interface RiskAssessment {
  overall_score?: number;
  composite_score: number;
  overall_severity?: string;
  level: string;
  dominant_risk: string;
  recommendation: string;
  summary_messages?: string[];
  categories: Record<string, RiskCategoryDetail>;
}

export interface ScenarioDetail {
  name: string;
  total_cost: number;
  cost_per_tonne: number;
  freight_rate: number;
  freight_cost: number;
  bunker_price: number;
  bunker_cost: number;
  duration_days: number;
  waiting_days: number;
  handling_days?: number;
  demurrage_days?: number;
  demurrage_cost: number;
  fixed_costs?: number;
  on_time: boolean;
  schedule_slack_days?: number;
  assumptions: string[];
}

export interface MonteCarloData {
  expected_cost: number;
  p10_cost: number;
  p50_cost: number;
  p90_cost: number;
  p10_cpt: number;
  p50_cpt: number;
  p90_cpt: number;
  demurrage_probability: number;
  late_delivery_probability: number;
  cost_distribution: Array<{
    bin_min: number;
    bin_max: number;
    bin_center: number;
    count: number;
    probability: number;
  }>;
}

export interface ContractStrategy {
  recommended_strategy: string;
  spot_percentage: number;
  short_term_percentage: number;
  medium_term_percentage: number;
  expected_cost: number;
  p90_cost: number;
  flexibility_score: number;
  reasons: string[];
}

export interface DecisionExplanation {
  summary: string;
  primary_reasons: string[];
  tradeoff_analysis: string;
  alternatives_rejected: Array<{
    vessel_class: string;
    reasons_rejected: string[];
  }>;
  recommendation_summary?: string;
  structured_answers?: Record<string, string>;
  shap_analysis?: Array<{
    feature: string;
    importance?: number;
    shap_value?: number;
    direction?: string;
    impact?: string;
  }>;
}

export interface HorizonForecastItem {
  rate: number;
  p10: number;
  p90: number;
  trend: string;
  confidence: number;
}

export interface MultiHorizonForecast {
  forecast_3d: HorizonForecastItem;
  forecast_7d: HorizonForecastItem;
  forecast_14d: HorizonForecastItem;
  forecast_30d: HorizonForecastItem;
}

export interface HistoricalRatePoint {
  date: string;
  rate: number;
}

export interface ForecastTrajectoryPoint {
  date: string;
  rate: number;
  p10: number;
  p90: number;
  is_forecast: boolean;
}

export interface DecisionResponse {
  decision_id: string;
  timestamp: string;
  model_versions: Record<string, string>;
  data_versions: Record<string, string>;
  request_summary: {
    cargo_type: string;
    cargo_quantity_t: number;
    origin_port_id: string;
    destination_port_id: string;
    expected_loading_date: string;
    required_delivery_date: string;
    number_of_voyages: number;
    risk_tolerance: string;
  };
  market_analysis: MarketAnalysis;
  freight_forecast: FreightForecast;
  market_timing: MarketTiming;
  recommended_plan: PlanItem;
  alternative_plans: PlanItem[];
  economics: EconomicsBreakdown;
  risk: RiskAssessment;
  contract_strategy: ContractStrategy;
  confidence: number;
  explanation: DecisionExplanation;
  scenario_analysis?: {
    BEST_CASE?: ScenarioDetail;
    BASE_CASE?: ScenarioDetail;
    WORST_CASE?: ScenarioDetail;
    [key: string]: any;
  };
  monte_carlo?: MonteCarloData;
  multi_horizon_forecast?: MultiHorizonForecast;
  historical_rates?: HistoricalRatePoint[];
  forecast_trajectory?: ForecastTrajectoryPoint[];
}

export interface MarketIntelligenceResponse {
  current_rate: number;
  forecast_3d: HorizonForecastItem;
  forecast_7d: HorizonForecastItem;
  forecast_14d: HorizonForecastItem;
  forecast_30d: HorizonForecastItem;
  historical_rates: HistoricalRatePoint[];
  forecast_trajectory: ForecastTrajectoryPoint[];
}

export interface SensitivityScenario {
  parameter: string;
  value: number;
  total_cost: number;
  cost_per_tonne: number;
  delta_cost_usd: number;
  delta_cost_pct: number;
  delta_cost_per_tonne_usd: number;
  demurrage_exposure: number;
  waiting_cost: number;
  bunker_cost: number;
  freight_cost: number;
  voyage_days: number;
  delivery_probability: number;
}

export interface SensitivityMatrixResponse {
  matrix?: {
    bunker_price: SensitivityScenario[];
    freight_rate: SensitivityScenario[];
    congestion: SensitivityScenario[];
    cargo_quantity: SensitivityScenario[];
  };
  parameter?: string;
  scenarios?: SensitivityScenario[];
}

// =============================================================================
// API Call Functions
// =============================================================================

export const fetchCharterDecision = async (payload: CharterDecisionRequest): Promise<DecisionResponse> => {
  const response = await apiClient.post<DecisionResponse>('/recommend/decision', payload);
  return response.data;
};

export const fetchMarketIntelligence = async (
  origin: string,
  destination: string,
  vessel_class: string = 'Panamax',
  cargo_type: string = 'thermal_coal'
): Promise<MarketIntelligenceResponse> => {
  const response = await apiClient.get<MarketIntelligenceResponse>('/forecast/intelligence', {
    params: { origin, destination, vessel_class, cargo_type },
  });
  return response.data;
};

export const fetchSensitivity = async (payload: {
  base_inputs: {
    cargo_quantity_t: number;
    freight_rate_usd: number;
    origin_port_id: string;
    destination_port_id: string;
    vessel_class: string;
    fuel_price_usd_per_t?: number;
    expected_waiting_days?: number;
  };
  parameter?: string;
  values?: number[];
}): Promise<SensitivityMatrixResponse> => {
  const response = await apiClient.post<SensitivityMatrixResponse>('/economics/sensitivity', payload);
  return response.data;
};

// Legacy backward-compatibility helpers
export interface AnalyzeVoyagePayload {
  cargo_type: string;
  cargo_quantity: number;
  origin: string;
  destination: string;
  required_delivery_date: string;
  number_of_voyages: number;
}

export const analyzeVoyage = async (payload: AnalyzeVoyagePayload) => {
  const response = await apiClient.post('/analyze-voyage', payload);
  return response.data;
};

export const getHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};
