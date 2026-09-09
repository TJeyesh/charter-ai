# Optimization Methodology

Unlike black-box neural networks, CharterAI uses a transparent, rules-based multi-criteria optimization methodology for its Decision Engine.

## Scoring System
Each vessel class is scored out of 100 based on:
- **Cargo Efficiency (30%)**: How closely the vessel's typical Deadweight Tonnage matches the requested cargo.
- **Port Compatibility (40%)**: Strict binary check. If draft/LOA/beam constraints fail, the vessel is disqualified immediately.
- **Voyage Economics (30%)**: Calculates bunker consumption against expected freight rate.

## Contract Strategy Optimization
The Contract Optimizer aggregates the outputs of the Forecasting Engine and the Risk Engine:
- If Forecast = RISING and Risk = LOW: Promotes **MEDIUM_TERM** or **HYBRID** to lock in rates.
- If Forecast = FALLING: Promotes **SPOT** to take advantage of cheaper future rates.
