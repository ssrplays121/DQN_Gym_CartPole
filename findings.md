# Experiment Findings & Observations

## 1. Learning Rate Analysis

| Learning Rate | Observation |
| :--- | :--- |
| **`1e-3`** | Movement was erratic, but the agent managed to balance the pole for a duration. |
| **`1e-4`** | Movement became more stable, but the cart eventually moves off-screen. |

### Conclusion

The agent successfully learned to balance but eventually overcame the soft virtual boundary (`BOUNDARY_LIMIT`), leading to it drifting off-screen to avoid penalty accumulation.

### Action Plan

To resolve the drifting issue, strict physical constraints are required:

- **Hard Limit**: Define the actual physical limit of the environment (e.g., `2.0`).
- **Kill Zones**: Implement penalties for breaching the hard limit to prevent the model from exploiting the soft limit.

> **Note**: These changes effectively force the agent to respect the physical boundaries of the CartPole environment.

## 2. Kill Zones Analysis

| Configuration | Observation |
| :--- | :--- |
| **`1e-4` + Soft Limit** | Pole is balanced but cart keeps leaving the boundary while balancing |
| **`1e-4` + Soft Limit + Hard Limit** | Model is much more stable and balances for longer durations. However, the pole still eventually falls at the end of the episode. |

## Next Step

Increase the required reward threshold to encourage the model to learn to balance for even longer periods.

## 3. Stress Test Analysis (Reward Threshold = 500)

**Simulation Runs**: 10

| Outcome | Frequency | Description |
| :--- | :--- | :--- |
| **Excellent Balance** | 7 | Constantly balanced in middle, minimal deviation, stayed within soft limit. |
| **Near Boundary** | 1 | Balanced but almost left the boundary. |
| **Crash** | 1 | Pole crashed. |
| **Unstable** | 1 | Jerking left/right within hard limit. |

> **(Note: Observations based on manual review of runs)**

## 6. Future Plans

- **Initial Conditions**: Test by placing the pole at an angle to see if the agent can recover.
- **Memory Scaling**: Experiment with larger replay memory sizes (Current: 10,000 samples ≈ 316 MB RAM).
