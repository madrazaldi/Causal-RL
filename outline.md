# Working Title

**Causal Reinforcement Learning for Urban Logistics: A Confounder-Aware Offline Case Study of Eco-Mode Control**

# Positioning

Treat the paper as an **IEEM-style applied logistics case study**. The contribution is not that causal RL solves logistics control in general. The contribution is that a **confounder-aware offline RL benchmark** can be built around a realistic logistics action, can be evaluated conservatively, and can provide **decision support for eco-mode selection** in a synthetic urban logistics setting.

Keep these claims consistent throughout:

- this is a **synthetic urban logistics case study**,
- the method is **confounder-aware offline RL**,
- the intended use is **decision support for eco-mode selection**,
- the broader implication is **intervention-aware logistics analytics**,
- the evidence does **not** show that causal FQI is the best overall policy,
- the evidence does **not** justify generalizing beyond the eco-mode case study.

# Abstract (150-200 words)

Use four short parts.

**Part 1: Logistics motivation**

Frame eco-mode selection as a last-mile operations problem under pressure to reduce emissions and fuel use without degrading service reliability or safety.

**Part 2: Research gap**

State that logistics analytics often emphasizes ETA prediction, lateness prediction, or planning, but much less often learns action policies from observational operational logs. Offline RL is attractive because live experimentation is risky, yet logged actions are confounded by urgency, congestion, weather, and route conditions.

**Part 3: Method**

Describe the method as a **confounder-aware offline policy-learning pipeline** for a synthetic urban logistics dataset with an observed action `eco_mode`, fitted Q iteration, support-constrained overrides, and off-policy evaluation.

**Part 4: Main findings**

Report only the findings that are actually supported:

- `non_causal_fqi` is the strongest overall policy by doubly robust value at `-4.457` with 95% CI `[-4.636, -4.280]`,
- `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) and remains operationally interpretable,
- subgroup analysis shows **segment-dependent trade-offs** and supports conservative deployment rather than universal automation.

# 1. Introduction

## Paragraph 1: Urban logistics problem

Open with sustainable urban logistics pressures:

- emissions and fuel costs,
- service reliability and time windows,
- safety in congested and weather-sensitive operations,
- pressure to make operational recommendations from data rather than fixed heuristics.

End with the main decision problem:

The challenge is not only predicting lateness or congestion, but recommending an operational action that balances timeliness, safety, and sustainability under observational bias.

## Paragraph 2: Limits of current logistics analytics

Review common logistics analytics tasks briefly:

- ETA and lateness prediction,
- routing and stop-sequence optimization,
- green logistics and emissions estimation,
- heuristic operational rules.

Then state the gap:

These approaches support planning or prediction, but they do not directly learn **action policies from logged operational decisions**.

## Paragraph 3: Why offline RL and why caution

Use this logic:

- online experimentation in logistics can be costly, unsafe, or impractical,
- offline RL can learn from logged decisions,
- logged eco-mode choices are observational and confounded,
- naive offline RL can exploit unstable correlations,
- causal state design and conservative support constraints can improve credibility.

Be explicit that the dataset is synthetic and that this improves internal clarity, not external validity:

The paper studies a synthetic urban logistics log with an observed binary action `eco_mode`, allowing a clean intervention-learning benchmark without claiming direct field deployment readiness.

## Paragraph 4: Research question and contributions

Use this research question:

**RQ:** In a synthetic urban logistics case study, can confounder-aware offline RL provide a credible and operationally useful decision-support policy for eco-mode selection relative to logged behavior, heuristic rules, and a broader non-causal offline RL comparator?

State four contributions in this order:

1. We formulate urban logistics eco-mode selection as an offline sequential decision problem over vehicle-day trajectories.
2. We build a **confounder-aware offline RL benchmark and evaluation pipeline** using backdoor-guided state design, fitted Q iteration, and conservative support-constrained overrides.
3. We show that **confounder-aware offline RL improves over logged behavior** while remaining interpretable and deployable as a decision-support layer.
4. We use the benchmark to reveal **trade-offs, robustness differences, and deployment limits**, including the fact that the broader non-causal comparator achieves the strongest overall doubly robust value.

# 2. Related Work and Research Gap

## 2.1 Logistics analytics and operational decision support

Cover four clusters:

- ETA and lateness prediction,
- routing and service planning,
- green logistics and emissions analytics,
- heuristic control and dispatch policies.

For each, emphasize the same gap:

- predicts risk but does not recommend actions,
- optimizes plans but not adaptive logged controls,
- often remains correlational,
- offers limited counterfactual support.

## 2.2 Offline RL and causal decision learning

Summarize:

- offline RL learns from logged data without live exploration,
- performance depends on overlap, extrapolation control, and state design,
- causal adjustment helps separate plausible confounders from unstable proxies,
- support constraints matter for safe deployment.

Then connect to the paper:

Eco-mode is an observed logistics action, not a label. This makes the task suitable for **intervention-aware logistics analytics** while also making confounding a first-order concern.

## Benchmark table

|**Study type**|**Task**|**Method**|**Output**|**Limitation**|
|---|---|---|---|---|
|Logistics ML|ETA/lateness prediction|RF/XGB/DL|Risk score|No action recommendation|
|Green logistics analytics|Fuel/emissions estimation|Regression/optimization|Efficiency estimate|Weak policy learning from logs|
|Offline RL|Policy learning|Batch RL|Action policy|Sensitive to confounding and shift|
|Causal decision learning|Intervention-aware policy learning|Causal adjustment + policy learning|Counterfactual action guidance|Rare in logistics operations|
|This study|Eco-mode control|Confounder-aware offline RL|Decision support for eco-mode selection|Synthetic case study only|

## Synthesis paragraph

Prior logistics studies largely stop at prediction, planning, or fixed heuristics. This paper instead treats eco-mode selection as an **applied logistics case study** in offline policy learning, with causal guidance used to design a more credible and deployable state representation rather than to claim universal causal identification.

# 3. Problem Formulation and Proposed Method

## 3.1 Sequential decision formulation

Define the sequential problem over trajectories formed by `date`-`vehicle_id`.

- **State**: pre-decision operational context available to a deployable controller
  - time and location: `day_idx`, `dow`, `hour`, `zone`
  - vehicle and task: `vehicle_id`, `vehicle_type`, `vehicle_age_years`, `vehicle_efficiency_index`, `commodity_type`, `demand_size`, `time_window_tightness`, `service_time_min`
  - road and environment: `speed_limit_kmph`, `intersection_density`, `road_grade_index`, `road_risk_index`, `rain`, `rain_intensity`, `temperature_c`, `visibility_km`, `event_indicator`, `roadworks_indicator`, `traffic_index`, `traffic_state`, `route_risky`
  - operational backlog: `dispatch_delay_min`
  - sequential context: `step_idx`, `remaining_steps`, `rolling_mean_traffic`, `rolling_cumulative_lateness`, `rolling_incident_count`, `prior_reward_primary`, `prior_eco_mode`

- **Action**: `eco_mode in {0,1}`
  - `0`: standard mode
  - `1`: eco mode

- **Reward**:

\[
r_t = -(\text{lateness\_min}_t + 10 \cdot \text{crash}_t + 2 \cdot \text{near\_miss}_t + 0.2 \cdot \text{co2\_kg}_t)
\]

Retain the two sensitivity rewards:

- service-heavy,
- sustainability-heavy.

- **Objective**: learn a policy that maximizes discounted return from historical logs without online exploration.

## 3.2 Confounder-aware offline RL framework

Present the method as a **confounder-aware offline policy-learning pipeline** inside a synthetic urban logistics case study.

**Step 1: Domain-guided causal adjustment**

- eco-mode choice is observational and may depend on urgency, congestion, weather, route risk, and recent history,
- use a small domain DAG to motivate a backdoor-style state,
- exclude post-action variables and latent simulator columns from the deployable policy state.

Use language like:

The causal design goal is not to prove full identification, but to construct a more credible pre-decision state for offline policy learning in urban logistics.

**Step 2: Offline policy learning**

Train two fitted Q iteration variants:

- **Non-causal FQI** with the broader deployable state including `risk_score`, `distance_km`, and `compatibility_violation`,
- **Causal FQI** with the backdoor-guided state only.

State plainly that this is a benchmark comparison:

The comparison tests whether confounder-aware state design changes policy quality and deployability; it does not assume in advance that the causal-state policy will dominate every comparator.

**Step 3: Conservative policy improvement**

Estimate the behavior policy with a calibrated classifier and allow overrides only when:

- action support is adequate,
- the Q-gap is large enough.

Frame this as operations support:

The resulting policy is a **decision-support layer for eco-mode selection**, not a fully autonomous controller.

## 3.3 Baselines and evaluation logic

Baselines:

1. `logged_behavior`
2. `always_eco`
3. `never_eco`
4. `heuristic_risk_rule`
5. `non_causal_fqi`
6. `causal_fqi`

Keep the two causal ablations as appendix diagnostics:

- `causal_no_history_fqi`
- `causal_no_vehicle_id_fqi`

Evaluate with:

- doubly robust policy value as the primary headline metric,
- IPS as a consistency check,
- FQE as an appendix diagnostic,
- estimated lateness, CO2, on-time rate, crash rate, and near-miss rate,
- override, fallback, and low-support rates,
- robustness across operational slices.

# 4. Experimental Setup

## 4.1 Dataset and decision log construction

State the concrete values:

- 96,000 logged operational records,
- 120 dates,
- 120 vehicles,
- 15 urban zones,
- observed action `eco_mode`,
- measured outcomes `lateness_min`, `co2_kg`, `near_miss`, `crash`, `on_time`.

Explain:

- each trajectory is defined by `date`-`vehicle_id`,
- rows are ordered by `hour` and row index,
- lagged features use only prior events within a trajectory,
- post-action and latent variables are excluded from the deployable state.

## 4.2 Implementation details

Keep this concise:

- Python with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib`
- temporal split by date into train / validation / test
- calibrated behavior model
- fitted Q iteration for causal and non-causal policies
- outcome models for decomposed metrics
- fixed seed for reproducibility

## 4.3 Evaluation protocol

State:

- held-out temporal test set,
- doubly robust estimator for the main table,
- self-normalized IPS for consistency,
- FQE in the appendix,
- bootstrap confidence intervals,
- robustness slices:
  - `high_traffic`
  - `rain_or_event`
  - `tight_window`
  - `late_day`

Add the fairness sentence:

All policies are evaluated on the same historical partitions and under the same logged support conditions.

# 5. Results and Discussion

## 5.1 Main policy comparison

Make the headline fully honest:

- `non_causal_fqi` is the strongest overall policy by doubly robust value at `-4.457` with 95% CI `[-4.636, -4.280]`,
- `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) but trails the broader non-causal comparator,
- `heuristic_risk_rule` is also competitive at `-4.638`, showing that simpler policies remain strong baselines.

Interpretation:

The case study supports confounder-aware offline RL as a viable approach, but not a claim that causal FQI is the best overall policy in this benchmark.

## 5.2 What the causal framing contributes

Make three points:

1. it enforces an honest pre-decision state design,
2. it reduces the temptation to rely on opaque proxy variables,
3. it supports conservative policy improvement suitable for operational decision support.

Do not claim that the causal model wins overall. Instead say:

The value of the causal framing in this paper is **credibility, interpretability, and deployment discipline**, not universal empirical dominance.

## 5.3 Ablations and robustness

Use the ablation results to make a modest point:

- `causal_no_vehicle_id_fqi` reaches `-4.670`,
- `causal_no_history_fqi` reaches `-4.687`,
- `causal_fqi` reaches `-4.694`.

Interpretation:

The causal-state family is relatively stable to removing history or vehicle identity, but the differences are small and should be read as sensitivity checks rather than proof of a unique causal mechanism.

For robustness, describe the segment pattern truthfully:

- `non_causal_fqi` is the strongest policy in all four evaluated segments,
- the causal policy still improves over logged behavior in `high_traffic`, `rain_or_event`, and `tight_window`,
- in `late_day`, `causal_fqi` falls slightly below logged behavior,
- heuristic or static policies sometimes outperform the causal policy in stressed segments.

Use this exact takeaway:

The benchmark reveals **segment-dependent trade-offs and the need for conservative policy deployment**.

## 5.4 Practical interpretation

End with a managerial paragraph:

The project should be presented as a tool for recommending when eco mode is more defensible and when operators should fall back to logged behavior or simpler rules. That makes the method a credible **decision-support layer for eco-mode selection**, not a justification for autonomous control.

# 6. Conclusion

Conclude in three sentences:

- The paper reframes eco-mode selection as an offline sequential decision problem in urban logistics.
- The synthetic urban logistics case study shows that confounder-aware offline RL can improve over logged behavior and support intervention-aware logistics analytics.
- The same results also show that stronger overall value can still come from a broader non-causal comparator, reinforcing the need for conservative interpretation and deployment.

## Limitations and future work

Keep these explicit:

- synthetic data rather than field interventions,
- binary action space only,
- no live deployment,
- no claim of full causal identification,
- future work should test richer action spaces, partial observability, and real operational logs.
