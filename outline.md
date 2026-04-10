# Working Title

**Confounder-Aware Offline Reinforcement Learning for Eco-Mode Control in Urban Logistics**

# Abstract (150–200 words)

**Part 1: Operational motivation**

Position eco-mode selection as an operational control problem in urban logistics. Eco-driving or eco-mode settings can lower fuel usage and carbon emissions, but they may also affect travel speed, service timeliness, and safety outcomes.

**Part 2: Gap**

State that most logistics analytics studies focus on prediction tasks such as ETA, lateness, and route efficiency, while few studies learn action policies from logged operational decisions. Real logistics datasets are often rich in events but do not expose explicit intervention labels; this motivates a clean decision-learning study on a logged action that is directly observed.

**Part 3: Method**

Summarize the method in one sentence:

This paper proposes a confounder-aware offline reinforcement learning framework that learns eco-mode control policies from historical urban logistics logs using causal adjustment, fitted Q iteration, and off-policy evaluation.

**Part 4: Results and implication**

Summarize only 2–3 outcomes:

- improved expected operational return under a timeliness-safety-emissions reward,
- more robust policy value under traffic and weather shifts,
- interpretable eco-mode recommendations for different operating conditions.

# 1. Introduction

**Paragraph 1: Problem context**

Explain why eco-mode control matters in last-mile and urban logistics:

- pressure to reduce emissions and fuel costs,
- need to preserve service levels,
- interaction with congestion, weather, and route risk,
- operational trade-offs between sustainability and performance.

End with the core challenge:

The main challenge is not only estimating whether an order is at risk, but selecting an operational control action that balances service, safety, and sustainability under observational bias.

**Paragraph 2: Existing solutions**

Briefly summarize logistics analytics work on:

- ETA prediction,
- late-delivery prediction,
- route optimization,
- emissions-aware planning,
- heuristic dispatch rules.

Then state the limitation:

These methods primarily provide predictions or fixed rules, but they do not directly learn decision policies from logged historical actions.

**Paragraph 3: Research gap**

Use this logic:

- offline RL is attractive because online experimentation in logistics can be unsafe or impractical,
- observational action logs are confounded because operators choose eco-mode differently under congestion, urgency, and route conditions,
- naive offline RL may learn spurious action-value relationships,
- causal adjustment can improve policy relevance and robustness.

Mention the dataset motivation briefly:

Industry datasets such as LaDe are rich in event and trajectory information, but they do not directly expose intervention labels suited to prescriptive policy learning. In contrast, the present synthetic urban logistics log contains an observed control action (`eco_mode`) and measurable service, safety, and emissions outcomes, making it suitable for a clean offline decision-learning study.

**Paragraph 4: Research question and contributions**

State the research question:

**RQ:** Can a confounder-aware offline RL policy choose better eco-mode actions than logged behavior, heuristics, and non-causal offline RL for improving the timeliness-safety-emissions trade-off in urban logistics?

Then keep only 3 contributions:

1. We formulate urban logistics eco-mode selection as an offline sequential decision problem over vehicle-day trajectories.
2. We propose a confounder-aware offline RL framework that combines domain-guided causal adjustment with fitted Q iteration and support-constrained policy improvement.
3. We benchmark the resulting policy against logged behavior, heuristic rules, and non-causal offline RL using off-policy evaluation, robustness analysis, and operational interpretation.

# 2. Related Work and Research Gap

## 2.1 Prediction and optimization in logistics analytics

Review four clusters briefly:

- ETA and lateness prediction,
- route and stop-sequence optimization,
- emissions-aware transportation analytics,
- heuristic dispatch or control policies.

For each cluster, emphasize one limitation:

- predicts risk but does not choose actions,
- optimizes plans but not logged adaptive control,
- often correlational,
- limited counterfactual support.

## 2.2 Offline RL and causal decision learning

Briefly introduce:

- offline RL learns policies from logged data without online exploration,
- major risk: extrapolation and distribution shift,
- causal adjustment helps separate confounders from unstable proxies,
- support constraints matter when moving from logged behavior to recommended actions.

Explain why this matters here:

Eco-mode is an action, not a label. Logged eco-mode decisions are observational and may depend on urgency, congestion, driver context, and route characteristics.

**Benchmark table**

|**Study type**|**Task**|**Method**|**Output**|**Limitation**|
|---|---|---|---|---|
|Logistics ML|ETA/lateness prediction|RF/XGB/DL|Risk score|No action recommendation|
|Green logistics analytics|Fuel/emissions estimation|Regression/optimization|Efficiency estimate|Weak policy learning from logs|
|Offline RL|Policy learning|Batch RL|Action policy|Sensitive to confounding and shift|
|Causal RL|Intervention learning|Causal policy learning|Counterfactual action|Rare in logistics operations|
|This study|Eco-mode control|Confounder-aware offline RL|Operational action policy|—|

**Synthesis paragraph**

Prior logistics studies largely stop at prediction, planning, or heuristic control. Offline RL offers a path from logs to action recommendation, but it can overfit spurious action-outcome patterns when historical decisions are confounded. This motivates a confounder-aware offline RL framework for eco-mode control that targets actionable and operationally interpretable policy learning.

# 3. Problem Formulation and Proposed Method

## 3.1 Sequential decision formulation

Define the problem as a sequential decision process over trajectories formed by `date`-`vehicle_id`.

- **State**: pre-decision operational context at each step
  - time and location context: `day_idx`, `dow`, `hour`, `zone`
  - vehicle and task context: `vehicle_id`, `vehicle_type`, `vehicle_age_years`, `vehicle_efficiency_index`, `commodity_type`, `demand_size`, `time_window_tightness`, `service_time_min`
  - road and environment context: `speed_limit_kmph`, `intersection_density`, `road_grade_index`, `road_risk_index`, `rain`, `rain_intensity`, `temperature_c`, `visibility_km`, `event_indicator`, `roadworks_indicator`, `traffic_index`, `traffic_state`, `route_risky`
  - operational backlog/control context: `dispatch_delay_min`
  - sequential context: `step_idx`, `remaining_steps`, `rolling_mean_traffic`, `rolling_cumulative_lateness`, `rolling_incident_count`, `prior_reward_primary`, `prior_eco_mode`

- **Action**: `eco_mode ∈ {0,1}`
  - `0`: standard mode
  - `1`: eco mode

- **Reward**:

\[
r_t = -(\text{lateness\_min}_t + 10 \cdot \text{crash}_t + 2 \cdot \text{near\_miss}_t + 0.2 \cdot \text{co2\_kg}_t)
\]

Add two sensitivity rewards:

- service-heavy reward,
- sustainability-heavy reward.

- **Objective**: learn a policy \(\pi(a|s)\) that maximizes expected discounted return from historical logs without online exploration.

## 3.2 Confounder-aware offline RL framework

Describe the framework in 3 steps.

**Step 1: Domain-guided causal adjustment**

State that eco-mode decisions are observational and may depend on urgency, weather, congestion, route risk, and recent operational history. Use a small domain DAG to justify a backdoor-adjustment feature set.

Suggested causal structure:

- confounder groups: time context, vehicle context, demand pressure, weather/traffic, route risk, recent history
- action node: `eco_mode`
- outcome node: reward / timeliness / safety / emissions

Include one sentence like:

To mitigate confounding, the framework uses a domain-guided backdoor set that captures shared causes of eco-mode choice and downstream operational outcomes while excluding latent and post-action variables from the deployable state.

**Step 2: Offline policy learning**

Use two fitted Q iteration variants:

- **Non-causal FQI**: uses the broader deployable state, including opaque proxy variables such as `risk_score`, `distance_km`, and `compatibility_violation`.
- **Causal FQI**: uses the confounder-aware backdoor-adjustment state only.

Explain why offline RL is needed:

Live experimentation with eco-mode in logistics operations may degrade service or safety and is impractical in production environments.

**Step 3: Support-constrained policy improvement and evaluation**

State that the historical behavior policy \(P(a|s)\) is estimated with a calibrated classifier. The learned policy only overrides logged behavior when:

- estimated action support is adequate, and
- the Q-value gap exceeds a conservative threshold.

Evaluation is done with off-policy estimators rather than live deployment.

**Workflow figure**

Historical urban logistics logs → Confounder-aware state construction → Behavior policy estimation → Offline policy learning → Off-policy evaluation → Eco-mode recommendation

## 3.3 Baselines and evaluation logic

Use these baselines:

1. **Logged behavior**
2. **Always eco**
3. **Never eco**
4. **Heuristic risk rule**: a predicted lateness-risk threshold policy
5. **Non-causal FQI**
6. **Causal FQI**

Evaluate with:

- expected policy value,
- estimated lateness,
- estimated CO2,
- estimated on-time rate,
- robustness under operational shifts,
- fallback and low-support override rates.

# 4. Experimental Setup

## 4.1 Dataset and decision log construction

Describe the dataset using concrete values:

- 96,000 logged operational records,
- 120 dates,
- 120 vehicles,
- 15 urban zones,
- observed binary action: `eco_mode`,
- measured outcomes: `lateness_min`, `co2_kg`, `near_miss`, `crash`, `on_time`.

Explain decision-log construction:

- each trajectory is defined by `date`-`vehicle_id`,
- records are ordered by `hour` and stable row index,
- each step stores current state, action, reward, terminal flag, and next state,
- lagged sequential features are computed using only prior events within the same trajectory.

State exclusions clearly:

- do not use post-action variables such as `avg_speed_kmph`, `travel_time_min`, `fuel_liters`, `co2_kg`, `near_miss`, `crash`, `lateness_min`, `on_time` in the deployable state,
- do not use latent columns in the deployable policy model,
- keep latent variables only for optional appendix discussion.

## 4.2 Implementation details

Keep this compact:

- Python with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- temporal split by date into train / validation / test
- calibrated behavior policy model
- fitted Q iteration for causal and non-causal policies
- outcome models for reward decomposition
- fixed random seed for reproducibility

## 4.3 Evaluation protocol

State:

- held-out temporal test set,
- self-normalized IPS,
- doubly robust estimation,
- fitted Q evaluation,
- scenario tests on:
  - high traffic,
  - rain or event conditions,
  - tight time windows,
  - late-day operations.

Add one fairness sentence:

All policies are evaluated on the same historical partitions and the same logged support conditions.

# 5. Results and Discussion

## 5.1 Policy performance against baselines

Main result table:

|**Policy**|**Policy value**|**Expected lateness**|**Expected CO2**|**On-time rate**|**Fallback rate**|
|---|---|---|---|---|---|

Make only 3 points in the text:

1. causal offline RL performs best or near-best on the primary reward,
2. heuristics and static baselines expose the sustainability-service trade-off but are less adaptive,
3. non-causal FQI is more sensitive to unstable proxy signals.

## 5.2 Why the causal policy helps

Explain:

- historical eco-mode choices are confounded by urgency and operating conditions,
- causal adjustment reduces reliance on unstable correlational proxies,
- support-constrained overrides make recommendations more conservative and believable.

Include one small figure:

- causal graph, or
- policy override conditions by scenario.

## 5.3 Robustness under operational shift

Show one compact robustness figure or table covering:

- high traffic,
- rain/events,
- tight time windows,
- late-day operations.

State explicitly:

The causal policy degrades less under operational shift than the non-causal alternative while preserving better trade-off quality than static rules.

## 5.4 Practical interpretation

End with one short managerial paragraph:

The learned policy should recommend eco mode more often under slack time windows and lower congestion, while reverting toward standard mode when service urgency, route risk, or operational stress is high. This positions the method as a decision-support layer rather than a fully autonomous controller.

# 6. Conclusion

Summarize in 3 sentences:

- the paper reframes urban logistics eco-mode selection as an offline sequential decision problem,
- a confounder-aware offline RL policy improves the timeliness-safety-emissions trade-off compared with logged behavior and non-causal alternatives,
- the study supports a shift from pure logistics prediction toward intervention-aware operational analytics.

**Limitations and future work**

- synthetic rather than real operational interventions,
- binary action space,
- no live field deployment,
- future work can extend to richer control spaces, multiple simultaneous objectives, and partial observability.
