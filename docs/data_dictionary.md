# Data Dictionary

## 1. Vessel Database
| Field | Type | Description |
|-------|------|-------------|
| `class_name` | String | e.g. "Capesize", "Panamax" |
| `dwt_max` | Float | Maximum Deadweight Tonnage capacity |
| `dwt_min` | Float | Minimum economically viable cargo |
| `draft_max_m` | Float | Maximum draft in meters |
| `loa_max_m` | Float | Maximum Length Overall in meters |
| `beam_max_m` | Float | Maximum width in meters |

## 2. Port Database
| Field | Type | Description |
|-------|------|-------------|
| `port_id` | String | e.g. "IND_DHA" for Dhamra |
| `max_draft_m` | Float | Physical channel limit |
| `max_loa_m` | Float | Berthing limit |
| `cargo_handling_rate_tpd`| Float | Tonnes per day loaded/discharged |
| `current_congestion_factor` | Float | Multiplier for baseline waiting time |
