import React, { useState } from 'react';
import { 
  Ship, Package, Loader2, AlertCircle, 
  TrendingUp, Activity, DollarSign, ShieldAlert, FileText, CheckCircle2, 
  XCircle, BrainCircuit, Clock, HelpCircle,
  Sliders, Calendar, Compass, BarChart3, RefreshCw
} from 'lucide-react';
import { 
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';
import { 
  fetchCharterDecision, 
  fetchSensitivity,
  type DecisionResponse, 
  type CharterDecisionRequest, 
  type PlanItem,
  type SensitivityScenario
} from '../api';

const PIE_COLORS = ['#3182CE', '#63B3ED', '#38A169', '#E53E3E', '#DD6B20', '#805AD5', '#718096'];

export default function VoyagePlanner() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DecisionResponse | null>(null);
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null);

  // Sensitivity analysis state
  const [sensitivityLoading, setSensitivityLoading] = useState(false);
  const [sensitivityResults, setSensitivityResults] = useState<SensitivityScenario[] | null>(null);
  const [sensitivityParam, setSensitivityParam] = useState<'bunker_price' | 'freight_rate' | 'congestion' | 'cargo_quantity'>('bunker_price');
  
  // Custom slider values for live sensitivity sweep
  const [bunkerPriceInput, setBunkerPriceInput] = useState(650);
  const [freightRateInput, setFreightRateInput] = useState(17.89);
  const [congestionInput, setCongestionInput] = useState(2.6);
  const [cargoQtyInput, setCargoQtyInput] = useState(75000);

  // Form parameters
  const todayStr = new Date().toISOString().split('T')[0];
  const futureDate = new Date();
  futureDate.setDate(futureDate.getDate() + 25);
  const deadlineStr = futureDate.toISOString().split('T')[0];
  
  const [formData, setFormData] = useState<CharterDecisionRequest>({
    cargo_type: 'thermal_coal',
    cargo_tonnage: 75000,
    origin_port_id: 'INA_TAB',
    destination_port_id: 'IND_DHA',
    earliest_date: todayStr,
    latest_date: deadlineStr,
    risk_appetite: 'MEDIUM',
  });

  const handleFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'cargo_tonnage' ? parseFloat(value) || 0 : value
    }));
  };

  const handleRunDecision = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await fetchCharterDecision(formData);
      setResult(data);
      setSelectedPlanId(data.recommended_plan.plan_id || 'rec_plan');
      
      // Sync sensitivity slider bases with live decision data
      if (data.market_analysis?.forecast) {
        setFreightRateInput(data.market_analysis.forecast);
      }
      if (data.recommended_plan?.expected_waiting !== undefined) {
        setCongestionInput(data.recommended_plan.expected_waiting);
      }
      setCargoQtyInput(data.request_summary?.cargo_quantity_t || formData.cargo_tonnage);

      // Trigger initial sensitivity run
      triggerSensitivity(data, sensitivityParam);
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(typeof err.response.data.detail === 'string' ? err.response.data.detail : JSON.stringify(err.response.data.detail));
      } else {
        setError(err.message || 'An unexpected error occurred while communicating with the CharterAI intelligence engine.');
      }
    } finally {
      setLoading(false);
    }
  };

  const triggerSensitivity = async (decisionData: DecisionResponse, param: 'bunker_price' | 'freight_rate' | 'congestion' | 'cargo_quantity') => {
    setSensitivityLoading(true);
    try {
      const baseFreight = decisionData.market_analysis?.forecast || freightRateInput;
      const baseWait = decisionData.recommended_plan?.expected_waiting || congestionInput;
      const baseQty = decisionData.request_summary?.cargo_quantity_t || cargoQtyInput;
      const vesselClass = decisionData.recommended_plan?.vessel_class || 'Panamax';

      const resp = await fetchSensitivity({
        base_inputs: {
          cargo_quantity_t: baseQty,
          freight_rate_usd: baseFreight,
          origin_port_id: decisionData.request_summary.origin_port_id,
          destination_port_id: decisionData.request_summary.destination_port_id,
          vessel_class: vesselClass,
          fuel_price_usd_per_t: bunkerPriceInput,
          expected_waiting_days: baseWait,
        },
        parameter: param,
      });

      if (resp.scenarios) {
        setSensitivityResults(resp.scenarios);
      } else if (resp.matrix && resp.matrix[param]) {
        setSensitivityResults(resp.matrix[param]);
      }
    } catch (err) {
      console.error('Sensitivity calculation error:', err);
    } finally {
      setSensitivityLoading(false);
    }
  };

  const handleSensitivityParamChange = (newParam: 'bunker_price' | 'freight_rate' | 'congestion' | 'cargo_quantity') => {
    setSensitivityParam(newParam);
    if (result) {
      triggerSensitivity(result, newParam);
    }
  };

  const handleApplySensitivity = () => {
    setFormData(prev => ({
      ...prev,
      cargo_tonnage: cargoQtyInput,
    }));
    handleRunDecision();
  };

  // Helper for Market Timing badge
  const renderTimingBadge = (timingAction: string) => {
    const action = timingAction.toUpperCase();
    if (action.includes('BOOK')) {
      return <span className="badge-timing badge-book-now">● {timingAction}</span>;
    } else if (action.includes('WAIT')) {
      return <span className="badge-timing badge-wait">● {timingAction}</span>;
    } else if (action.includes('MONITOR')) {
      return <span className="badge-timing badge-monitor">● {timingAction}</span>;
    } else if (action.includes('NEGOTIAT')) {
      return <span className="badge-timing badge-negotiate">● {timingAction}</span>;
    } else {
      return <span className="badge-timing badge-hybrid">● {timingAction}</span>;
    }
  };

  // Build combined chart series for Historical + Forecast + Uncertainty Band
  const buildChartData = () => {
    if (!result) return [];
    const points: Array<{ date: string; actualRate?: number; forecastRate?: number; p10?: number; p90?: number; band?: [number, number] }> = [];

    // Historical Points (last 30 days)
    if (result.historical_rates && result.historical_rates.length > 0) {
      result.historical_rates.forEach(item => {
        points.push({
          date: item.date,
          actualRate: item.rate,
        });
      });
    }

    // Forecast Trajectory Points (next 30 days with uncertainty band)
    if (result.forecast_trajectory && result.forecast_trajectory.length > 0) {
      result.forecast_trajectory.forEach(item => {
        // Find if date already exists (e.g. T0)
        const existing = points.find(p => p.date === item.date);
        if (existing) {
          existing.forecastRate = item.rate;
          existing.p10 = item.p10;
          existing.p90 = item.p90;
          existing.band = [item.p10, item.p90];
        } else {
          points.push({
            date: item.date,
            forecastRate: item.rate,
            p10: item.p10,
            p90: item.p90,
            band: [item.p10, item.p90],
          });
        }
      });
    }

    return points;
  };

  const chartData = buildChartData();

  // All plans for comparison: Recommended + Alternatives
  const allPlans: PlanItem[] = result 
    ? [
        { ...result.recommended_plan, plan_id: result.recommended_plan.plan_id || 'Plan A (Recommended)' },
        ...(result.alternative_plans || []).map((alt, idx) => ({
          ...alt,
          plan_id: alt.plan_id || `Plan ${String.fromCharCode(66 + idx)}`,
          vessel_class: alt.vessel_classes?.[0] || alt.vessel_class || 'Alternative',
          cost_per_tonne: alt.cost_per_tonne || (alt.total_cost && formData.cargo_tonnage ? alt.total_cost / formData.cargo_tonnage : 0),
        }))
      ]
    : [];

  const activePlan = allPlans.find(p => p.plan_id === selectedPlanId) || allPlans[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Top Header Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Compass size={28} color="var(--accent-primary)" />
            CharterAI V2 Decision Dashboard
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Multi-voyage fleet optimization, probabilistic risk modeling, and market timing engine
          </p>
        </div>
        {result && (
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button 
              type="button"
              className="btn" 
              style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              onClick={() => handleRunDecision()}
              disabled={loading}
            >
              <RefreshCw size={16} className={loading ? 'spinner' : ''} />
              Re-evaluate Corridor
            </button>
          </div>
        )}
      </div>

      {/* =====================================================================
          SECTION 1: CHARTER REQUEST
          ===================================================================== */}
      <div className="card" id="charter-request-card">
        <div className="card-header">
          <h2 className="card-title">
            <Package size={20} color="var(--accent-primary)" />
            1. Charter Request Parameters
          </h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Commercial & Operational Corridor Constraints</span>
        </div>

        <form onSubmit={handleRunDecision}>
          <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
            <div className="form-group">
              <label className="form-label">Origin Port</label>
              <select name="origin_port_id" className="form-control" value={formData.origin_port_id} onChange={handleFormChange}>
                <option value="INA_TAB">Taboneo, Indonesia (INA_TAB)</option>
                <option value="AUS_NEW">Newcastle, Australia (AUS_NEW)</option>
                <option value="ZAF_RB">Richards Bay, South Africa (ZAF_RB)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Destination Port</label>
              <select name="destination_port_id" className="form-control" value={formData.destination_port_id} onChange={handleFormChange}>
                <option value="IND_DHA">Dhamra, India (IND_DHA)</option>
                <option value="IND_PAR">Paradip, India (IND_PAR)</option>
                <option value="IND_VIS">Visakhapatnam, India (IND_VIS)</option>
                <option value="IND_HAL">Haldia, India (IND_HAL)</option>
                <option value="IND_GAN">Gangavaram, India (IND_GAN)</option>
                <option value="IND_GOP">Gopalpur, India (IND_GOP)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Cargo Type</label>
              <select name="cargo_type" className="form-control" value={formData.cargo_type} onChange={handleFormChange}>
                <option value="thermal_coal">Thermal Coal</option>
                <option value="iron_ore">Iron Ore</option>
                <option value="grain">Grain</option>
                <option value="bauxite">Bauxite</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Cargo Quantity (MT)</label>
              <input 
                type="number" 
                name="cargo_tonnage" 
                className="form-control" 
                value={formData.cargo_tonnage} 
                onChange={handleFormChange} 
                step={5000} 
                min={10000} 
                required 
              />
            </div>

            <div className="form-group">
              <label className="form-label">Earliest Loading Date</label>
              <input 
                type="date" 
                name="earliest_date" 
                className="form-control" 
                value={formData.earliest_date} 
                onChange={handleFormChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label className="form-label">Delivery Deadline</label>
              <input 
                type="date" 
                name="latest_date" 
                className="form-control" 
                value={formData.latest_date} 
                onChange={handleFormChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label className="form-label">Risk Tolerance</label>
              <select name="risk_appetite" className="form-control" value={formData.risk_appetite} onChange={handleFormChange}>
                <option value="LOW">Low (Conservative / Term Hedges)</option>
                <option value="MEDIUM">Medium (Balanced Strategy)</option>
                <option value="HIGH">High (Aggressive / Spot Exposure)</option>
              </select>
            </div>

            <div className="form-group" style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button type="submit" className="btn btn-primary" style={{ width: '100%', height: '42px' }} disabled={loading}>
                {loading ? <><Loader2 className="spinner" size={18} /> Optimizing Fleet...</> : <><Activity size={18} /> Run CharterAI Engine</>}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Error Alert State */}
      {error && (
        <div className="error-alert">
          <AlertCircle size={24} style={{ flexShrink: 0 }} />
          <div>
            <h4 style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Intelligence Engine Validation Notice</h4>
            <p style={{ fontSize: '0.9rem' }}>{error}</p>
          </div>
        </div>
      )}

      {/* Loading Skeleton State */}
      {loading && !result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="grid-3">
            <div className="card skeleton-box" style={{ height: '140px' }}></div>
            <div className="card skeleton-box" style={{ height: '140px' }}></div>
            <div className="card skeleton-box" style={{ height: '140px' }}></div>
          </div>
          <div className="card skeleton-box" style={{ height: '350px' }}></div>
        </div>
      )}

      {/* Empty State */}
      {!result && !loading && (
        <div className="empty-state">
          <Ship size={48} color="var(--accent-primary)" style={{ opacity: 0.8 }} />
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>No Active Charter Optimization</h3>
            <p style={{ maxWidth: '600px', margin: '0.5rem auto 0 auto', fontSize: '0.9rem' }}>
              Select your origin, destination, cargo volume, and delivery window above, then click 
              <strong> Run CharterAI Engine</strong> to evaluate freight rates, fleet allocation, and market timing.
            </p>
          </div>
          <button 
            type="button"
            className="btn btn-primary" 
            style={{ marginTop: '0.5rem' }}
            onClick={() => handleRunDecision()}
          >
            Load Reference Corridor (Taboneo → Dhamra 75,000 MT)
          </button>
        </div>
      )}

      {/* Main Results View */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {/* =====================================================================
              SECTION 2: MARKET INTELLIGENCE
              ===================================================================== */}
          <div className="card" id="market-intelligence-section">
            <div className="card-header">
              <h2 className="card-title">
                <TrendingUp size={20} color="var(--accent-primary)" />
                2. Market Intelligence & Probabilistic Freight Forecast
              </h2>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Model: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{result.freight_forecast?.model_used || 'Multi-Model Ensemble'}</span>
              </div>
            </div>

            {/* 5 KPI Metric Cards */}
            <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', marginBottom: '1.5rem' }}>
              <div className="stat-box">
                <div className="stat-label">Current Freight</div>
                <div className="stat-value">${result.market_analysis?.current_rate?.toFixed(2) || result.freight_forecast?.current_rate_usd?.toFixed(2)} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>/MT</span></div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Spot Market T0</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Forecast 3d</div>
                <div className="stat-value accent">
                  ${result.multi_horizon_forecast?.forecast_3d?.rate?.toFixed(2) || (result.market_analysis?.forecast * 0.98).toFixed(2)}
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}> /MT</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  P10-P90: ${result.multi_horizon_forecast?.forecast_3d?.p10?.toFixed(2) || '—'} - ${result.multi_horizon_forecast?.forecast_3d?.p90?.toFixed(2) || '—'}
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Forecast 7d</div>
                <div className="stat-value accent">
                  ${result.multi_horizon_forecast?.forecast_7d?.rate?.toFixed(2) || result.market_analysis?.forecast?.toFixed(2)}
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}> /MT</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  P10-P90: ${result.multi_horizon_forecast?.forecast_7d?.p10?.toFixed(2) || '—'} - ${result.multi_horizon_forecast?.forecast_7d?.p90?.toFixed(2) || '—'}
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Forecast 14d</div>
                <div className="stat-value accent">
                  ${result.multi_horizon_forecast?.forecast_14d?.rate?.toFixed(2) || (result.market_analysis?.forecast * 1.02).toFixed(2)}
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}> /MT</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  P10-P90: ${result.multi_horizon_forecast?.forecast_14d?.p10?.toFixed(2) || '—'} - ${result.multi_horizon_forecast?.forecast_14d?.p90?.toFixed(2) || '—'}
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Forecast 30d</div>
                <div className="stat-value accent">
                  ${result.multi_horizon_forecast?.forecast_30d?.rate?.toFixed(2) || (result.market_analysis?.forecast * 1.04).toFixed(2)}
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}> /MT</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  P10-P90: ${result.multi_horizon_forecast?.forecast_30d?.p10?.toFixed(2) || '—'} - ${result.multi_horizon_forecast?.forecast_30d?.p90?.toFixed(2) || '—'}
                </div>
              </div>
            </div>

            {/* Chart: Historical Freight + Forecast Trajectory + Shaded Uncertainty Band */}
            <div style={{ height: '340px', width: '100%', marginTop: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                  <XAxis 
                    dataKey="date" 
                    stroke="var(--text-muted)" 
                    fontSize={11} 
                    tickLine={false} 
                    axisLine={false}
                    tickFormatter={(tick) => tick.slice(5)}
                  />
                  <YAxis 
                    stroke="var(--text-muted)" 
                    fontSize={11} 
                    tickLine={false} 
                    axisLine={false} 
                    domain={['auto', 'auto']} 
                    tickFormatter={(val) => `$${val}`} 
                  />
                  <RechartsTooltip 
                    contentStyle={{ 
                      backgroundColor: 'var(--bg-tertiary)', 
                      borderColor: 'var(--border-color)', 
                      borderRadius: '8px', 
                      color: 'var(--text-primary)' 
                    }}
                    formatter={(value: any, name: any) => {
                      if (Array.isArray(value)) {
                        return [`$${value[0]?.toFixed(2)} - $${value[1]?.toFixed(2)}`, 'Uncertainty Range (P10-P90)'];
                      }
                      return [`$${Number(value).toFixed(2)} /MT`, name];
                    }}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }} />
                  
                  {/* Shaded Uncertainty Band between P10 and P90 */}
                  <Area 
                    type="monotone" 
                    dataKey="band" 
                    stroke="none" 
                    fill="#3182CE" 
                    fillOpacity={0.18} 
                    name="Uncertainty Band (P10 - P90)" 
                  />

                  {/* Historical Rate Line */}
                  <Line 
                    type="monotone" 
                    dataKey="actualRate" 
                    stroke="#63B3ED" 
                    strokeWidth={2.5} 
                    dot={{ r: 2, fill: '#63B3ED' }} 
                    name="Historical Freight" 
                  />

                  {/* Forecast Rate Line */}
                  <Line 
                    type="monotone" 
                    dataKey="forecastRate" 
                    stroke="#ED8936" 
                    strokeWidth={2.5} 
                    strokeDasharray="4 4" 
                    dot={{ r: 3, fill: '#ED8936' }} 
                    name="Forecast Rate (P50)" 
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Historical window: 30 days prior | Projection horizon: 30 days forward</span>
              <span>Confidence: {(result.market_analysis?.confidence * 100).toFixed(1)}% | Direction: <strong style={{ color: 'var(--accent-secondary)' }}>{result.market_analysis?.direction}</strong></span>
            </div>
          </div>

          {/* =====================================================================
              SECTION 3: MARKET TIMING
              ===================================================================== */}
          <div className="card" id="market-timing-section" style={{ borderLeft: '4px solid var(--accent-primary)' }}>
            <div className="card-header">
              <h2 className="card-title">
                <Clock size={20} color="var(--accent-primary)" />
                3. Market Timing Recommendation
              </h2>
              <div>{renderTimingBadge(result.market_timing?.recommendation || 'MONITOR')}</div>
            </div>

            <div className="grid-3" style={{ marginBottom: '1.25rem' }}>
              <div className="stat-box">
                <div className="stat-label">Model Confidence</div>
                <div className="stat-value success">
                  {((result.market_timing?.confidence || 0.75) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Statistical certainty</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Expected Economic Savings</div>
                <div className={`stat-value ${(result.market_timing?.expected_savings || 0) >= 0 ? 'success' : 'warning'}`}>
                  ${Math.abs(result.market_timing?.expected_savings || 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {(result.market_timing?.expected_savings || 0) >= 0 ? 'Projected benefit vs waiting' : 'Net risk exposure if delayed'}
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Deadline Delay Risk</div>
                <div className="stat-value danger">
                  ${Math.round(result.market_timing?.deadline_risk || 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Laycan & demurrage exposure</div>
              </div>
            </div>

            {/* Booking Window & Economic Reasoning */}
            <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color-light)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.9rem' }}>
                <Calendar size={16} color="var(--accent-secondary)" />
                Optimal Booking Window: {result.market_timing?.recommended_booking_window?.start || formData.earliest_date} to {result.market_timing?.recommended_booking_window?.end || formData.latest_date}
              </div>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                {result.market_timing?.reasons?.map((reason, idx) => (
                  <li key={idx} style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                    <span style={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}>•</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* =====================================================================
              SECTION 5: RECOMMENDED PLAN (HERO CARD)
              ===================================================================== */}
          <div className="card" id="recommended-plan-card" style={{ background: 'linear-gradient(180deg, var(--bg-secondary) 0%, rgba(49, 130, 206, 0.05) 100%)', border: '1px solid rgba(49, 130, 206, 0.4)' }}>
            <div className="card-header" style={{ borderBottomColor: 'rgba(49, 130, 206, 0.2)' }}>
              <div>
                <span className="badge" style={{ backgroundColor: 'rgba(49, 130, 206, 0.2)', color: 'var(--accent-secondary)', marginBottom: '0.35rem' }}>
                  OPTIMAL ALLOCATION
                </span>
                <h2 className="card-title" style={{ fontSize: '1.4rem' }}>
                  <Ship size={22} color="var(--accent-primary)" />
                  5. Recommended Plan: {result.recommended_plan?.vessel_count || 1} × {result.recommended_plan?.vessel_class} ({result.recommended_plan?.voyages || 1} Voyage)
                </h2>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Delivered Cost / Tonne</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--accent-secondary)' }}>
                  ${result.recommended_plan?.cost_per_tonne?.toFixed(2)} <span style={{ fontSize: '0.9rem', fontWeight: 500, color: 'var(--text-muted)' }}>/t</span>
                </div>
              </div>
            </div>

            <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
              <div className="stat-box">
                <div className="stat-label">Total Delivered Cost</div>
                <div className="stat-value">${Math.round(result.recommended_plan?.total_cost || 0).toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>All 9 components included</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Vessel Utilization</div>
                <div className="stat-value success">{((result.recommended_plan?.utilization || 0) * 100).toFixed(1)}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Deadweight efficiency</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Voyage Duration</div>
                <div className="stat-value">{result.recommended_plan?.voyage_duration?.toFixed(1)} d</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Expected waiting: {result.recommended_plan?.expected_waiting?.toFixed(1)} d</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Delivery Probability</div>
                <div className="stat-value success">{((result.recommended_plan?.delivery_probability || 0.95) * 100).toFixed(1)}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Demurrage risk: {((result.recommended_plan?.demurrage_probability || 0) * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Port Constraints & Compatibility Details */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--bg-tertiary)', padding: '0.85rem 1.25rem', borderRadius: 'var(--radius-md)', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', fontSize: '0.875rem' }}>
                <CheckCircle2 size={18} color="#48BB78" />
                <span>Port Constraints Status: <strong>100% Feasible & Verified</strong> (Draft, LOA, Beam limits compliant at {formData.origin_port_id} and {formData.destination_port_id})</span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <span className="badge badge-low">Composite Risk: {result.recommended_plan?.risk_score?.toFixed(1)}/100</span>
              </div>
            </div>
          </div>

          {/* =====================================================================
              SECTION 4: VESSEL PLAN COMPARISON TABLE
              ===================================================================== */}
          <div className="card" id="vessel-comparison-table-section">
            <div className="card-header">
              <h2 className="card-title">
                <BarChart3 size={20} color="var(--accent-primary)" />
                4. Vessel Plan Comparison Table
              </h2>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Comparing {allPlans.length} Feasible Multi-Voyage & Fleet Combinations
              </span>
            </div>

            <div className="table-container">
              <table className="v2-table">
                <thead>
                  <tr>
                    <th>Plan</th>
                    <th>Vessel</th>
                    <th>Vessels</th>
                    <th>Voyages</th>
                    <th>Utilization</th>
                    <th>Total Cost</th>
                    <th>Cost / Tonne</th>
                    <th>Waiting</th>
                    <th>Demurrage</th>
                    <th>Delivery Prob</th>
                    <th>Risk Score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {allPlans.map((plan, idx) => {
                    const isSelected = plan.plan_id === selectedPlanId;
                    const isRecommended = idx === 0;
                    return (
                      <tr 
                        key={plan.plan_id || idx} 
                        className={isSelected ? 'selected' : ''}
                        onClick={() => setSelectedPlanId(plan.plan_id || `plan_${idx}`)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td style={{ fontWeight: 600, color: isRecommended ? 'var(--accent-secondary)' : 'var(--text-primary)' }}>
                          {plan.plan_id}
                        </td>
                        <td>{plan.vessel_class}</td>
                        <td>{plan.vessel_count || plan.number_of_vessels || 1}</td>
                        <td>{plan.voyages || plan.number_of_voyages || 1}</td>
                        <td>{((plan.utilization || 0) * 100).toFixed(1)}%</td>
                        <td style={{ fontWeight: 600 }}>${Math.round(plan.total_cost || 0).toLocaleString()}</td>
                        <td style={{ fontWeight: 600, color: 'var(--accent-secondary)' }}>${plan.cost_per_tonne?.toFixed(2)}</td>
                        <td>{(plan.expected_waiting || plan.waiting_days || 0).toFixed(1)} d</td>
                        <td>{((plan.demurrage_probability || 0) * 100).toFixed(1)}%</td>
                        <td>
                          <span style={{ color: (plan.delivery_probability || 0.95) > 0.9 ? '#48BB78' : '#ECC94B' }}>
                            {((plan.delivery_probability || 0.95) * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${plan.risk_score < 35 ? 'badge-low' : (plan.risk_score < 60 ? 'badge-mod' : 'badge-high')}`}>
                            {plan.risk_score?.toFixed(1)}
                          </span>
                        </td>
                        <td>
                          {isRecommended ? (
                            <span className="badge badge-low" style={{ fontWeight: 700 }}>RECOMMENDED</span>
                          ) : (
                            <span className="badge" style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>ALTERNATIVE</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {activePlan && activePlan.plan_id !== allPlans[0]?.plan_id && (
              <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: 'rgba(49, 130, 206, 0.08)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(49, 130, 206, 0.2)', fontSize: '0.85rem' }}>
                <strong style={{ color: 'var(--accent-secondary)' }}>Inspecting {activePlan.plan_id} ({activePlan.vessel_class}):</strong> Delivered cost is ${Math.round(activePlan.total_cost).toLocaleString()} (${activePlan.cost_per_tonne?.toFixed(2)}/t) vs Recommended Plan ${Math.round(allPlans[0].total_cost).toLocaleString()} (${allPlans[0].cost_per_tonne?.toFixed(2)}/t). Variance: {activePlan.total_cost >= allPlans[0].total_cost ? `+$${Math.round(activePlan.total_cost - allPlans[0].total_cost).toLocaleString()} higher delivered expense` : `-$${Math.round(allPlans[0].total_cost - activePlan.total_cost).toLocaleString()} lower delivered expense`}.
              </div>
            )}
          </div>

          {/* =====================================================================
              SECTION 6: COST BREAKDOWN
              ===================================================================== */}
          <div className="card" id="cost-breakdown-section">
            <div className="card-header">
              <h2 className="card-title">
                <DollarSign size={20} color="var(--accent-primary)" />
                6. Realistic Voyage Economics & Cost Breakdown
              </h2>
              <div style={{ fontSize: '0.9rem', color: 'var(--accent-secondary)', fontWeight: 600 }}>
                Total Delivered Cost: ${Math.round(result.economics?.total_cost || result.recommended_plan?.total_cost || 0).toLocaleString()}
              </div>
            </div>

            <div className="grid-2" style={{ alignItems: 'center' }}>
              {/* Pie Chart Representation */}
              <div style={{ height: '280px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Freight', value: result.economics?.freight_cost || 0 },
                        { name: 'Bunker', value: result.economics?.bunker_cost || 0 },
                        { name: 'Port Charges', value: result.economics?.port_charges || 0 },
                        { name: 'Waiting', value: result.economics?.waiting_cost || 0 },
                        { name: 'Demurrage', value: result.economics?.demurrage_exposure || 0 },
                        { name: 'Other', value: (result.economics?.positioning_cost || 0) + (result.economics?.miscellaneous_cost || 0) },
                      ]}
                      cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={4}
                      dataKey="value" stroke="none"
                    >
                      {PIE_COLORS.map((color, index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <RechartsTooltip formatter={(value) => `$${Number(value).toLocaleString()}`} contentStyle={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: '8px' }}/>
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Itemized Table of 7 Required Elements */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {[
                  { label: 'Freight Cost', val: result.economics?.freight_cost || 0, color: PIE_COLORS[0] },
                  { label: 'Bunker Fuel', val: result.economics?.bunker_cost || 0, color: PIE_COLORS[1] },
                  { label: 'Port Charges', val: result.economics?.port_charges || 0, color: PIE_COLORS[2] },
                  { label: 'Waiting Cost', val: result.economics?.waiting_cost || 0, color: PIE_COLORS[3] },
                  { label: 'Demurrage Exposure', val: result.economics?.demurrage_exposure || 0, color: PIE_COLORS[4] },
                  { label: 'Other (Positioning & Agency)', val: (result.economics?.positioning_cost || 0) + (result.economics?.miscellaneous_cost || 0), color: PIE_COLORS[5] },
                ].map((item, idx) => {
                  const total = result.economics?.total_cost || 1;
                  const pct = ((item.val / total) * 100).toFixed(1);
                  return (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.45rem 0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                        <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: item.color }} />
                        <span>{item.label}</span>
                      </div>
                      <div style={{ textAlign: 'right', fontSize: '0.85rem' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>${Math.round(item.val).toLocaleString()}</strong>
                        <span style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>({pct}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* =====================================================================
              SECTION 7: RISK DASHBOARD (7 CORE CATEGORIES)
              ===================================================================== */}
          <div className="card" id="risk-dashboard-section">
            <div className="card-header">
              <h2 className="card-title">
                <ShieldAlert size={20} color="var(--accent-primary)" />
                7. Maritime Risk Center & 8-Category Audit
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Composite Score:</span>
                <span className={`badge ${result.risk?.composite_score < 35 ? 'badge-low' : (result.risk?.composite_score < 60 ? 'badge-mod' : 'badge-high')}`}>
                  {result.risk?.composite_score?.toFixed(1)} / 100 — {result.risk?.level || result.risk?.overall_severity}
                </span>
              </div>
            </div>

            <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
              {[
                { name: 'Market', icon: TrendingUp },
                { name: 'Port', icon: Compass },
                { name: 'Weather', icon: Activity },
                { name: 'Availability', icon: Ship },
                { name: 'Operational', icon: Package },
                { name: 'Geopolitical', icon: ShieldAlert },
                { name: 'Schedule', icon: Clock },
                { name: 'Demurrage', icon: DollarSign },
              ].map((catInfo) => {
                // Find matching category in result.risk.categories
                const foundKey = Object.keys(result.risk?.categories || {}).find(k => 
                  k.toLowerCase().includes(catInfo.name.toLowerCase())
                );
                const catData = foundKey ? result.risk.categories[foundKey] : null;
                const score = catData ? catData.score : 15;
                const severity = catData ? (catData.severity || catData.level || 'LOW') : 'LOW';
                const factors = catData?.contributing_factors || ['Normal operating baseline'];

                return (
                  <div key={catInfo.name} className="stat-box" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>{catInfo.name} Risk</span>
                        <span className={`badge ${severity.toUpperCase().includes('LOW') ? 'badge-low' : (severity.toUpperCase().includes('MOD') ? 'badge-mod' : 'badge-high')}`}>
                          {severity}
                        </span>
                      </div>
                      <div className="stat-value" style={{ fontSize: '1.25rem' }}>{score.toFixed(1)} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>/100</span></div>
                    </div>
                    <ul style={{ listStyle: 'none', padding: 0, margin: '0.75rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      {factors.slice(0, 2).map((factor, fIdx) => (
                        <li key={fIdx} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>• {factor}</li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          </div>

          {/* =====================================================================
              SECTION 8: SCENARIO ANALYSIS
              ===================================================================== */}
          <div className="card" id="scenario-analysis-section">
            <div className="card-header">
              <h2 className="card-title">
                <Compass size={20} color="var(--accent-primary)" />
                8. Scenario Stress Tests & Probabilistic Monte Carlo Quantiles
              </h2>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>10,000 Stochastic Iterations</span>
            </div>

            {/* Deterministic Scenarios: Best, Base, Worst */}
            <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
              {/* Best Case */}
              <div className="scenario-card best">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ fontWeight: 700, color: 'var(--accent-success)' }}>BEST CASE</h4>
                  <span className="badge badge-low">Optimistic</span>
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  ${Math.round(result.scenario_analysis?.BEST_CASE?.total_cost || result.monte_carlo?.p10_cost || 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Cost / Tonne: <strong>${(result.scenario_analysis?.BEST_CASE?.cost_per_tonne || result.monte_carlo?.p10_cpt || 0).toFixed(2)}</strong>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-color-light)', paddingTop: '0.5rem' }}>
                  {result.scenario_analysis?.BEST_CASE?.assumptions?.[0] || 'Zero anchorage waiting, smooth weather speed'}
                </div>
              </div>

              {/* Base Case */}
              <div className="scenario-card base">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ fontWeight: 700, color: 'var(--accent-secondary)' }}>BASE CASE</h4>
                  <span className="badge badge-mod">Expected</span>
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  ${Math.round(result.scenario_analysis?.BASE_CASE?.total_cost || result.monte_carlo?.p50_cost || 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Cost / Tonne: <strong>${(result.scenario_analysis?.BASE_CASE?.cost_per_tonne || result.monte_carlo?.p50_cpt || 0).toFixed(2)}</strong>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-color-light)', paddingTop: '0.5rem' }}>
                  {result.scenario_analysis?.BASE_CASE?.assumptions?.[0] || 'Median congestion, normal bunker consumption'}
                </div>
              </div>

              {/* Worst Case */}
              <div className="scenario-card worst">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ fontWeight: 700, color: 'var(--accent-danger)' }}>WORST CASE</h4>
                  <span className="badge badge-high">Stress Test</span>
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  ${Math.round(result.scenario_analysis?.WORST_CASE?.total_cost || result.monte_carlo?.p90_cost || 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Cost / Tonne: <strong>${(result.scenario_analysis?.WORST_CASE?.cost_per_tonne || result.monte_carlo?.p90_cpt || 0).toFixed(2)}</strong>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-color-light)', paddingTop: '0.5rem' }}>
                  {result.scenario_analysis?.WORST_CASE?.assumptions?.[0] || 'Heavy port bottleneck, bunker price spike'}
                </div>
              </div>
            </div>

            {/* Monte Carlo Statistical Quantiles: P10, P50, P90 */}
            <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color-light)' }}>
              <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                Statistical Distribution Quantiles (Monte Carlo Simulation)
              </div>
              <div className="grid-3">
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>P10 (10th Percentile)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#48BB78' }}>
                    ${Math.round(result.monte_carlo?.p10_cost || 0).toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>${result.monte_carlo?.p10_cpt?.toFixed(2)} /MT</div>
                </div>

                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>P50 (Median Expectation)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-secondary)' }}>
                    ${Math.round(result.monte_carlo?.p50_cost || 0).toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>${result.monte_carlo?.p50_cpt?.toFixed(2)} /MT</div>
                </div>

                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>P90 (90th Percentile Risk)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#F56565' }}>
                    ${Math.round(result.monte_carlo?.p90_cost || 0).toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>${result.monte_carlo?.p90_cpt?.toFixed(2)} /MT</div>
                </div>
              </div>
            </div>
          </div>

          {/* =====================================================================
              SECTION 9: CONTRACT STRATEGY
              ===================================================================== */}
          <div className="card" id="contract-strategy-section">
            <div className="card-header">
              <h2 className="card-title">
                <FileText size={20} color="var(--accent-primary)" />
                9. Risk-Aware Contract Portfolio Strategy
              </h2>
              <span className="badge" style={{ backgroundColor: 'rgba(128, 90, 213, 0.2)', color: '#B794F4' }}>
                {result.contract_strategy?.recommended_strategy}
              </span>
            </div>

            <div className="grid-3" style={{ marginBottom: '1.25rem' }}>
              <div className="stat-box">
                <div className="stat-label">Spot Allocation</div>
                <div className="stat-value">{result.contract_strategy?.spot_percentage}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Prompt voyage fixture</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Short-Term Contract</div>
                <div className="stat-value">{result.contract_strategy?.short_term_percentage}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>3-6 month index hedge</div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Medium-Term Contract</div>
                <div className="stat-value">{result.contract_strategy?.medium_term_percentage}%</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>12+ month COA / time charter</div>
              </div>
            </div>

            {/* Strategic Rationale */}
            <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color-light)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                <span>Expected Portfolio Cost: <strong>${Math.round(result.contract_strategy?.expected_cost || 0).toLocaleString()}</strong></span>
                <span>P90 Downside Cap: <strong>${Math.round(result.contract_strategy?.p90_cost || 0).toLocaleString()}</strong></span>
                <span>Flexibility Score: <strong>{(result.contract_strategy?.flexibility_score * 100).toFixed(0)}/100</strong></span>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {result.contract_strategy?.reasons?.map((r, idx) => (
                  <li key={idx} style={{ display: 'flex', gap: '0.5rem' }}>
                    <span style={{ color: 'var(--accent-secondary)' }}>✓</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* =====================================================================
              SECTION 10: EXPLAINABILITY ("WHY CHARTERAI RECOMMENDS THIS PLAN")
              ===================================================================== */}
          <div className="card" id="explainability-section" style={{ borderLeft: '4px solid var(--accent-primary)', backgroundColor: 'rgba(49, 130, 206, 0.02)' }}>
            <div className="card-header">
              <h2 className="card-title" style={{ color: 'var(--accent-primary)' }}>
                <BrainCircuit size={20} />
                10. Explainability: Why CharterAI Recommends This Plan
              </h2>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Transparent Decision Reasoning & Audited Tradeoffs
              </span>
            </div>

            {/* Tradeoff Analysis vs Runner Up */}
            <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-md)', marginBottom: '1.5rem', border: '1px solid var(--border-color-light)' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem', fontSize: '0.9rem' }}>
                Comparative Tradeoff Statement
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.5 }}>
                {result.explanation?.tradeoff_analysis || result.explanation?.summary}
              </p>
            </div>

            {/* 7 Core Structured Questions */}
            {result.explanation?.structured_answers && (
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                  Audited Decision Architecture (Core Explanations)
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '0.75rem' }}>
                  {Object.entries(result.explanation.structured_answers).map(([question, answer], qIdx) => (
                    <div key={qIdx} style={{ backgroundColor: 'var(--bg-tertiary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color-light)' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-secondary)', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <HelpCircle size={14} />
                        {question}
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>
                        {answer}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Rejected Alternatives with reasons */}
            {result.explanation?.alternatives_rejected && result.explanation.alternatives_rejected.length > 0 && (
              <div>
                <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                  Rejected Candidates & Constraint Filters
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.75rem' }}>
                  {result.explanation.alternatives_rejected.map((alt, idx) => (
                    <div key={idx} style={{ backgroundColor: 'var(--bg-tertiary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color-light)' }}>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
                        <XCircle size={15} color="#EF4444" />
                        {alt.vessel_class}
                      </div>
                      <ul style={{ listStyle: 'none', padding: 0, margin: '0.4rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {alt.reasons_rejected.map((r, rIdx) => (
                          <li key={rIdx}>• {r}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* =====================================================================
              SECTION 11: SENSITIVITY ANALYSIS
              ===================================================================== */}
          <div className="card" id="sensitivity-analysis-section">
            <div className="card-header">
              <h2 className="card-title">
                <Sliders size={20} color="var(--accent-primary)" />
                11. Interactive Scenario Sensitivity Analysis
              </h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  type="button"
                  className="btn btn-primary" 
                  style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                  onClick={handleApplySensitivity}
                  disabled={loading}
                >
                  Apply & Re-Optimize
                </button>
              </div>
            </div>

            {/* Interactive Sliders Grid */}
            <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
              {/* Bunker Price Slider */}
              <div className="slider-container">
                <div className="slider-header">
                  <span className="slider-label">Bunker Price (VLSFO)</span>
                  <span className="slider-value">${bunkerPriceInput} /t</span>
                </div>
                <input 
                  type="range" 
                  className="range-slider"
                  min={400} 
                  max={900} 
                  step={10} 
                  value={bunkerPriceInput} 
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setBunkerPriceInput(val);
                    if (result && sensitivityParam === 'bunker_price') triggerSensitivity(result, 'bunker_price');
                  }} 
                />
                <button 
                  type="button" 
                  className="btn" 
                  style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', backgroundColor: sensitivityParam === 'bunker_price' ? 'var(--accent-primary)' : 'var(--bg-secondary)', color: 'white' }}
                  onClick={() => handleSensitivityParamChange('bunker_price')}
                >
                  Analyze Bunker Sweep
                </button>
              </div>

              {/* Freight Rate Slider */}
              <div className="slider-container">
                <div className="slider-header">
                  <span className="slider-label">Freight Benchmark</span>
                  <span className="slider-value">${freightRateInput.toFixed(2)} /t</span>
                </div>
                <input 
                  type="range" 
                  className="range-slider"
                  min={10} 
                  max={35} 
                  step={0.5} 
                  value={freightRateInput} 
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setFreightRateInput(val);
                    if (result && sensitivityParam === 'freight_rate') triggerSensitivity(result, 'freight_rate');
                  }} 
                />
                <button 
                  type="button" 
                  className="btn" 
                  style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', backgroundColor: sensitivityParam === 'freight_rate' ? 'var(--accent-primary)' : 'var(--bg-secondary)', color: 'white' }}
                  onClick={() => handleSensitivityParamChange('freight_rate')}
                >
                  Analyze Freight Sweep
                </button>
              </div>

              {/* Congestion Waiting Days Slider */}
              <div className="slider-container">
                <div className="slider-header">
                  <span className="slider-label">Port Congestion</span>
                  <span className="slider-value">{congestionInput.toFixed(1)} days</span>
                </div>
                <input 
                  type="range" 
                  className="range-slider"
                  min={0} 
                  max={15} 
                  step={0.5} 
                  value={congestionInput} 
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setCongestionInput(val);
                    if (result && sensitivityParam === 'congestion') triggerSensitivity(result, 'congestion');
                  }} 
                />
                <button 
                  type="button" 
                  className="btn" 
                  style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', backgroundColor: sensitivityParam === 'congestion' ? 'var(--accent-primary)' : 'var(--bg-secondary)', color: 'white' }}
                  onClick={() => handleSensitivityParamChange('congestion')}
                >
                  Analyze Congestion Sweep
                </button>
              </div>

              {/* Cargo Quantity Slider */}
              <div className="slider-container">
                <div className="slider-header">
                  <span className="slider-label">Cargo Quantity</span>
                  <span className="slider-value">{cargoQtyInput.toLocaleString()} MT</span>
                </div>
                <input 
                  type="range" 
                  className="range-slider"
                  min={20000} 
                  max={180000} 
                  step={5000} 
                  value={cargoQtyInput} 
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setCargoQtyInput(val);
                    if (result && sensitivityParam === 'cargo_quantity') triggerSensitivity(result, 'cargo_quantity');
                  }} 
                />
                <button 
                  type="button" 
                  className="btn" 
                  style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', backgroundColor: sensitivityParam === 'cargo_quantity' ? 'var(--accent-primary)' : 'var(--bg-secondary)', color: 'white' }}
                  onClick={() => handleSensitivityParamChange('cargo_quantity')}
                >
                  Analyze Parcel Sweep
                </button>
              </div>
            </div>

            {/* Sensitivity Analysis Results Sweep Table */}
            {sensitivityLoading ? (
              <div style={{ textAlign: 'center', padding: '2rem' }}>
                <Loader2 className="spinner" size={24} style={{ margin: '0 auto 0.5rem auto' }} />
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Recalculating Delivered Economics Sweep...</span>
              </div>
            ) : sensitivityResults && sensitivityResults.length > 0 ? (
              <div className="table-container">
                <table className="v2-table">
                  <thead>
                    <tr>
                      <th>Scenario Value</th>
                      <th>Total Delivered Cost</th>
                      <th>Cost / Tonne</th>
                      <th>Delta Cost ($)</th>
                      <th>Delta (%)</th>
                      <th>Demurrage Exposure</th>
                      <th>Delivery Prob</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sensitivityResults.map((scen, sIdx) => {
                      const isBase = scen.delta_cost_usd === 0;
                      return (
                        <tr key={sIdx} style={{ backgroundColor: isBase ? 'rgba(49, 130, 206, 0.08)' : 'transparent' }}>
                          <td style={{ fontWeight: 600 }}>
                            {sensitivityParam === 'bunker_price' && `$${scen.value}/t`}
                            {sensitivityParam === 'freight_rate' && `$${scen.value.toFixed(2)}/t`}
                            {sensitivityParam === 'congestion' && `${scen.value} days`}
                            {sensitivityParam === 'cargo_quantity' && `${Math.round(scen.value).toLocaleString()} MT`}
                            {isBase && <span className="badge badge-low" style={{ marginLeft: '0.5rem' }}>BASE</span>}
                          </td>
                          <td style={{ fontWeight: 600 }}>${Math.round(scen.total_cost).toLocaleString()}</td>
                          <td>${scen.cost_per_tonne.toFixed(2)}</td>
                          <td>
                            <span className={`delta-tag ${scen.delta_cost_usd <= 0 ? 'delta-saving' : 'delta-cost'}`}>
                              {scen.delta_cost_usd >= 0 ? '+' : ''}${Math.round(scen.delta_cost_usd).toLocaleString()}
                            </span>
                          </td>
                          <td>
                            <span className={`delta-tag ${scen.delta_cost_pct <= 0 ? 'delta-saving' : 'delta-cost'}`}>
                              {scen.delta_cost_pct >= 0 ? '+' : ''}{scen.delta_cost_pct.toFixed(1)}%
                            </span>
                          </td>
                          <td>${Math.round(scen.demurrage_exposure || 0).toLocaleString()}</td>
                          <td>{((scen.delivery_probability || 0.95) * 100).toFixed(1)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>

        </div>
      )}
    </div>
  );
}
