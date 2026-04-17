# Working Title

**Confounder-Aware Offline RL for Eco-Mode Selection in Urban Last-Mile Logistics**

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
- `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) and remains operationally interpretable; a minimal 5-feature ablation reaches only `-4.810`, confirming the full causal state adds measurable value,
- support constraint thresholds are selected on a held-out validation partition (τ_μ = 0.05, τ_Q = 0.50) and then fixed for all test-set evaluation,
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

Use the provider clarification explicitly:

- one row is one dispatch decision / trip segment / decision epoch,
- `eco_mode` is an epoch-level controllable decision and can change during the day,
- `hour` is the decision-time bucket, not departure or arrival time.

- **State**: pre-decision operational context available to a deployable controller
  - time and location: `day_idx`, `dow`, `hour`, `zone`
  - vehicle and task: `vehicle_id`, `vehicle_type`, `vehicle_age_years`, `vehicle_efficiency_index`, `commodity_type`, `demand_size`, `time_window_tightness`, `service_time_min`
  - road and environment: `speed_limit_kmph`, `intersection_density`, `road_grade_index`, `road_risk_index`, `rain`, `rain_intensity`, `temperature_c`, `visibility_km`, `event_indicator`, `roadworks_indicator`, `traffic_index`, `traffic_state`
  - prior decisions: `route_risky`, `dispatch_delay_min` (deliberate choices made before eco-mode)
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

- **Non-causal FQI** with the broader deployable proxy state including `risk_score`, `distance_km`, and `compatibility_violation`,
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

Include the ablations as diagnostics:

- `causal_no_history_fqi`
- `causal_no_vehicle_id_fqi`
- `minimal_fqi` (5-feature minimal state — establishes the floor for causal state benefit)

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
- rows are ordered by `hour` and original row index because the raw schema has no finer-grained timestamp,
- about 19.4% of `(date, vehicle_id, hour)` groups contain multiple rows, so within-hour order is a documented source-row tie-break assumption,
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
- FQE in the appendix (note: FQE values are on a different scale than DR/IPS because FQE estimates discounted trajectory-accumulated Q-values, while DR/IPS report per-step averages; this is expected and does not indicate an error),
- bootstrap confidence intervals (500 replicates),
- **support constraint thresholds τ_μ and τ_Q are selected on the validation partition** using a conservative score that penalizes override rate; the selected pair (τ_μ = 0.05, τ_Q = 0.50) is then held fixed for all test evaluation — this prevents implicit test-set tuning,
- robustness slices:
  - `high_traffic`
  - `rain_or_event`
  - `tight_window`
  - `late_day`
- **cluster bootstrap** (resampling full vehicle-day trajectories) yields CIs 2.9% wider on average than row-level bootstrap, confirming that row-level intervals are mildly optimistic approximations of trajectory-level uncertainty.

Add the fairness sentence:

All policies are evaluated on the same historical partitions and under the same logged support conditions.

# 5. Results and Discussion

## 5.1 Main policy comparison

Make the headline fully honest:

- `non_causal_fqi` is the strongest overall policy by doubly robust value at `-4.457` with 95% CI `[-4.636, -4.280]`,
- `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) but trails the broader non-causal comparator,
- `heuristic_risk_rule` is also competitive at `-4.638`, showing that simpler policies remain strong baselines,
- the confidence intervals for the top policies overlap substantially, indicating that the differences are not decisive at standard significance levels.

Note: support thresholds τ_μ = 0.05, τ_Q = 0.50 were selected on the validation set using a conservative score (DR value − 0.5 × override rate − 0.25 × low-support rate) and then held fixed for all reported test-set results.

Interpretation:

The case study supports confounder-aware offline RL as a viable approach, but not a claim that causal FQI is the best overall policy in this benchmark.

## 5.2 What the causal framing contributes

Make four points:

1. it enforces an honest pre-decision state design,
2. it reduces the temptation to rely on opaque proxy variables (`distance_km` is correlated with the route_risky treatment; `risk_score` embeds latent driver risk propensity; `compatibility_violation` reflects vehicle-cargo matching logic — all three are excluded from the causal state because their construction conflates pre-decision context with latent or treatment-adjacent information),
   Treat the non-causal comparator as a broader deployable proxy state, not as a causal state.
3. it supports conservative policy improvement suitable for operational decision support,
4. **empirical feature importance confirms operationally interpretable drivers**: the causal FQI Q-function is dominated by `remaining_steps` (trajectory position: importance 1.30), `time_window_tightness` (0.37), `speed_limit_kmph` (0.10), and `traffic_index` (0.06) — all features a logistics manager would immediately recognize as relevant to eco-mode timing.

Do not claim that the causal model wins overall. Instead say:

The value of the causal framing in this paper is **credibility, interpretability, and deployment discipline**, not universal empirical dominance.

## 5.3 Ablations and robustness

Use the ablation results to make a clear, graded point:

- `minimal_fqi` (5-feature baseline: hour, demand_size, time_window_tightness, traffic_index, dispatch_delay_min) reaches only `-4.810`, a gap of **+0.116** below `causal_fqi`,
- `causal_no_vehicle_id_fqi` reaches `-4.670` (gap +0.024),
- `causal_no_history_fqi` reaches `-4.687` (gap +0.007),
- `causal_fqi` reaches `-4.694`.

Interpretation:

The minimal ablation shows that the full 33-feature causal backdoor state provides measurable value (+0.116 DR) over the simplest operationally available features. Removing vehicle identity or history within the causal state has only minor effects, suggesting the causal state is robust to feature subsets while still outperforming a severely restricted baseline.

For robustness, describe the segment pattern truthfully:

- `non_causal_fqi` is the strongest policy in all four evaluated segments,
- the causal policy still improves over logged behavior in `high_traffic`, `rain_or_event`, and `tight_window`,
- in `late_day`, `causal_fqi` falls slightly below logged behavior,
- heuristic or static policies sometimes outperform the causal policy in stressed segments.

**Late-day failure diagnosis:** The late_day underperformance is associated with a moderate distribution shift: traffic_index in late-day test hours is 0.014 higher than in training (0.870 vs 0.856), and dispatch_delay_min is 0.18 minutes longer. Under these slightly more congested conditions, the causal policy's support constraints cause more fallback-to-logged actions, limiting any improvement over the behavior policy. This finding supports adding time-of-day interaction features or segment-specific thresholds in future work.

Use this exact takeaway:

The benchmark reveals **segment-dependent trade-offs and the need for conservative policy deployment**.

## 5.4 Practical interpretation

End with a managerial paragraph:

The project should be presented as a tool for recommending when eco mode is more defensible and when operators should fall back to logged behavior or simpler rules. That makes the method a credible **decision-support layer for eco-mode selection**, not a justification for autonomous control.

## 5.5 Interpretability and Deployment Safety

Add a focused paragraph (can be a subsection within 5.2 or a standalone section depending on page budget):

Permutation importance analysis of the causal FQI Q-function identifies `remaining_steps` (trajectory position: importance 1.30), `time_window_tightness` (0.37), `speed_limit_kmph` (0.10), and `traffic_index` (0.06) as the primary decision drivers. These features are immediately interpretable to logistics operations managers: eco mode becomes less attractive when little trajectory time remains, when the time window is tight, or when traffic is congested.

Contrast with the non-causal state: the non-causal FQI uses `risk_score` (predicted risk embedding latent driver characteristics), `compatibility_violation` (a vehicle-cargo constraint flag), and `distance_km` (planned route distance correlated with route choice) as additional features. While this broader state achieves stronger overall DR value, its Q-function drivers are harder to audit and explain to operators. The causal state trades a small performance margin (the gap is within overlapping confidence intervals) for a more operationally auditable decision rule.

The common support analysis confirms that the behavior policy covers virtually all test-set states: only 0.028% of test rows have propensity below τ_μ = 0.05, and the minimum propensity is 0.023. This means the override mechanism operates in a well-supported region and the DR correction term is not distorted by extreme importance weights.

# 6. Conclusion

Conclude in three sentences:

- The paper reframes eco-mode selection as an offline sequential decision problem in urban logistics.
- The synthetic urban logistics case study shows that confounder-aware offline RL can improve over logged behavior and support intervention-aware logistics analytics.
- The same results also show that stronger overall value can still come from a broader non-causal comparator, reinforcing the need for conservative interpretation and deployment.

## Limitations and future work

Keep these explicit:

- synthetic data rather than field interventions; the sim-to-real gap is unquantified,
- binary action space only — does not address routing, vehicle assignment, or speed selection,
- no live deployment; causal identification assumptions (overlap, SUTVA) are not empirically verified,
- no claim of full causal identification — the backdoor adjustment is a domain-informed heuristic; two unobserved confounders (driver skill, maintenance latency) are not blocked,
- row-level bootstrap CIs are mildly optimistic; cluster bootstrap (resampling trajectories) yields 2.9% wider intervals,
- future work should test richer action spaces, partial observability, time-of-day segment-specific policies, and real operational logs.
