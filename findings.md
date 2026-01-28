# Experiment Findings & Observations

## 1. Learning Rate Analysis

| Learning Rate | Observation |
| :--- | :--- |
| **`1e-3`** | Movement was erratic, but the agent managed to balance the pole for a duration. |
| **`1e-4`** | Movement became more stable, but the cart eventually moves off-screen. |

## 2. Conclusion
The agent successfully learned to balance but eventually overcame the soft virtual boundary (`BOUNDARY_LIMIT`), leading to it drifting off-screen to avoid penalty accumulation.

## 3. Action Plan
To resolve the drifting issue, strict physical constraints are required:
- **Hard Limit**: Define the actual physical limit of the environment (e.g., `2.0`).
- **Kill Zones**: Implement penalties for breaching the hard limit to prevent the model from exploiting the soft limit.

> **Note**: These changes effectively force the agent to respect the physical boundaries of the CartPole environment.
