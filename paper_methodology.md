# Methodology

## 3. Problem Formulation and Proposed Method

### 3.1 Sequential Decision Formulation

We formulate urban logistics eco-mode control as an offline sequential decision problem over vehicle-day trajectories. Let the logged dataset be

$$
\mathcal{D} = \{\tau_i\}_{i=1}^{N}, \qquad
\tau_i = \{(s_{i,t}, a_{i,t}, r_{i,t}, s_{i,t+1}, d_{i,t})\}_{t=0}^{T_i-1},
$$

where each trajectory corresponds to one `date`-`vehicle_id` pair, $s_{i,t}$ is the pre-decision state at step $t$, $a_{i,t}\in\{0,1\}$ is the logged action, $r_{i,t}$ is the observed reward, $s_{i,t+1}$ is the next state, and $d_{i,t}\in\{0,1\}$ indicates trajectory termination.

The state is restricted to **pre-decision deployable covariates**. In the implementation, the state contains six groups of information:

1. **Time and location context**: `day_idx`, `dow`, `hour`, `zone`.
2. **Vehicle and task context**: `vehicle_id`, `vehicle_type`, `vehicle_age_years`, `vehicle_efficiency_index`, `commodity_type`, `demand_size`, `time_window_tightness`, `service_time_min`.
3. **Road and environmental context**: `speed_limit_kmph`, `intersection_density`, `road_grade_index`, `road_risk_index`, `rain`, `rain_intensity`, `temperature_c`, `visibility_km`, `event_indicator`, `roadworks_indicator`, `traffic_index`, `traffic_state`, `route_risky`.
4. **Operational backlog context**: `dispatch_delay_min`.
5. **Sequential context**: `step_idx`, `remaining_steps`, `rolling_mean_traffic`, `rolling_cumulative_lateness`, `rolling_incident_count`, `prior_reward_primary`, `prior_eco_mode`.
6. **Non-causal proxy extensions** for the broader comparator: `compatibility_violation`, `distance_km`, and `risk_score`.

The action space is binary:

$$
a_t = \texttt{eco\_mode}_t \in \{0,1\},
$$

where $a_t=0$ denotes standard mode and $a_t=1$ denotes eco mode.

The primary step reward balances service, safety, and sustainability:

$$
r_t
=
-
\Big(
\text{lateness\_min}_t
+
10 \cdot \text{crash}_t
+
2 \cdot \text{near\_miss}_t
+
0.2 \cdot \text{co2\_kg}_t
\Big).
$$

To test whether conclusions depend on managerial preference, we also evaluate two alternative reward functions:

$$
r_t^{\text{service}}
=
-
\Big(
1.5 \cdot \text{lateness\_min}_t
+
12 \cdot \text{crash}_t
+
3 \cdot \text{near\_miss}_t
+
0.1 \cdot \text{co2\_kg}_t
\Big),
$$

$$
r_t^{\text{sustain}}
=
-
\Big(
0.8 \cdot \text{lateness\_min}_t
+
10 \cdot \text{crash}_t
+
2 \cdot \text{near\_miss}_t
+
0.4 \cdot \text{co2\_kg}_t
\Big).
$$

The objective is to learn a policy $\pi(a \mid s)$ that maximizes the discounted return

$$
J(\pi)

=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1} \gamma^t r_t
\right],
$$

with discount factor $\gamma = 0.95$, using only historical logs and without online exploration.

### 3.2 Confounder-Aware Offline RL Framework

The proposed framework has three stages: domain-guided causal adjustment, offline policy learning, and support-constrained policy improvement with offline evaluation.

#### 3.2.1 Domain-Guided Causal Adjustment

Historical eco-mode decisions are observational rather than randomized. In practice, operators may activate eco mode differently depending on urgency, traffic, weather, route conditions, vehicle characteristics, and recent delivery history. These factors affect both action choice and downstream outcomes, creating confounding.

We encode this assumption with a domain DAG whose main structure is

$$
Z_t \rightarrow A_t, \qquad
Z_t \rightarrow Y_t, \qquad
A_t \rightarrow Y_t,
$$

where $Z_t$ denotes observed pre-decision confounders, $A_t$ is eco-mode selection, and $Y_t$ denotes reward-related outcomes such as lateness, incidents, and emissions. The confounder blocks represented in the DAG are:

- time context,
- vehicle context,
- demand pressure,
- weather and traffic,
- route risk,
- recent operational history.

Under the working backdoor assumption, conditioning on $Z_t$ blocks spurious action-outcome paths:

$$
Y_t(a) \perp A_t \mid Z_t.
$$

The implementation therefore defines a **causal backdoor state** using the observed covariates that plausibly cause both eco-mode choice and operational outcomes. This design intentionally excludes:

- post-action variables such as `avg_speed_kmph`, `travel_time_min`, `fuel_liters`, `co2_kg`, `near_miss`, `crash`, `lateness_min`, and `on_time`,
- latent simulator-only columns such as `maintenance_latent`, `driver_skill_latent`, and `driver_risk_propensity_latent`.

These exclusions matter because post-action variables would leak outcome information into the policy state, while latent variables are not available to a deployable real-world controller.

#### 3.2.2 Behavior Policy Estimation

Because offline policy improvement must remain close to historically supported decisions, we first estimate the logging policy

$$
\mu(a \mid s) = \Pr(A_t = a \mid S_t = s).
$$

In the implementation, $\mu$ is estimated using a calibrated logistic regression model fitted on the causal state:

$$
\hat{\mu}(a \mid s)
=
\text{CalibratedClassifier}\big(\text{LogisticRegression}(s)\big).
$$

This model provides estimated propensities used in both conservative policy improvement and importance-weighted evaluation.

#### 3.2.3 Offline Policy Learning via Fitted Q Iteration

We learn deterministic decision rules using fitted Q iteration (FQI). Let $Q(s,a)$ denote the expected discounted return after taking action $a$ in state $s$. For each Bellman iteration $k$, the regression target is

$$
y_i^{(k)}
=
r_i
+
\gamma (1-d_i)\max_{a' \in \{0,1\}} \hat{Q}^{(k-1)}(s'_i, a').
$$

Separate regressors are then fit for each action:

$$
\hat{Q}^{(k)}(\cdot, a)
=
\arg\min_{f \in \mathcal{F}}
\sum_{i: a_i = a}
\left(y_i^{(k)} - f(s_i)\right)^2.
$$

After the final iteration $K$, the greedy action is

$$
\pi_{\text{greedy}}(s)
=
\arg\max_{a \in \{0,1\}} \hat{Q}^{(K)}(s,a).
$$

The code uses $K=5$ iterations and gradient-boosted regression trees for the action-value regressors.

We train two main FQI variants:

1. **Non-causal FQI**

$$
\pi_{\text{NC}}(s)
=
\arg\max_a \hat{Q}_{\text{NC}}(s,a),
$$

where the state includes the broader deployable feature set, including `compatibility_violation`, `distance_km`, and `risk_score`.

2. **Causal FQI**

$$
\pi_{\text{C}}(s)
=
\arg\max_a \hat{Q}_{\text{C}}(s,a),
$$

where the state is restricted to the backdoor-guided causal feature set.

Two ablation variants are also trained for mechanism checks:

- **causal\_no\_history\_fqi**, which removes rolling-history features and prior eco-mode information,
- **causal\_no\_vehicle\_id\_fqi**, which removes `vehicle_id` to test whether gains are driven by vehicle-specific memorization.

#### 3.2.4 Support-Constrained Policy Improvement

Pure greedy improvement can recommend actions that are weakly supported by the logged data. To make recommendations more conservative, the learned policy only overrides the logged action when both support and value-gap criteria are satisfied.

Let

$$
a_t^\star = \arg\max_{a \in \{0,1\}} \hat{Q}(s_t,a),
\qquad
\Delta_t = \left|\hat{Q}(s_t,1) - \hat{Q}(s_t,0)\right|.
$$

The deployed policy is

$$
\pi_{\text{safe}}(s_t, a_t^{\text{log}})
=
\begin{cases}
a_t^\star, & \text{if } \hat{\mu}(a_t^\star \mid s_t) \ge \tau_\mu \text{ and } \Delta_t \ge \tau_Q, \\
a_t^{\text{log}}, & \text{otherwise},
\end{cases}
$$

where $a_t^{\text{log}}$ is the logged action, $\tau_\mu$ is a minimum support threshold, and $\tau_Q$ is a minimum Q-gap threshold. In the publication configuration, these are set to

$$
\tau_\mu = 0.05,
\qquad
\tau_Q = 0.50.
$$

This rule converts offline RL from unconstrained imitation-breaking optimization into a conservative decision-support layer.

### 3.3 Baselines and Evaluation Logic

The learned policies are compared against four baselines:

1. **Logged behavior**: replay the historical action sequence.
2. **Always eco**: set $a_t=1$ for every decision.
3. **Never eco**: set $a_t=0$ for every decision.
4. **Heuristic risk rule**: train a lateness-risk classifier and recommend eco mode when predicted lateness risk is below a threshold.

For the heuristic policy, let $\hat{p}_{\text{late}}(s)$ be the predicted probability of lateness. The rule is

$$
\pi_{\text{heur}}(s)
=
\mathbb{I}\!\left\{
\hat{p}_{\text{late}}(s) < c_{0.60}
\right\},
$$

where $c_{0.60}$ is the 60th percentile of predicted lateness risk within the evaluation frame. Intuitively, the heuristic favors eco mode in lower-risk situations and standard mode in higher-risk situations.

Policy comparison focuses on:

- expected policy value under the primary and sensitivity rewards,
- expected lateness,
- expected CO$_2$,
- expected crash and near-miss rates,
- expected on-time rate,
- eco-mode rate,
- override, fallback, and low-support rates,
- robustness under operational shifts.

## 4. Experimental Setup

### 4.1 Dataset and Decision Log Construction

The study uses a synthetic urban logistics dataset with:

- 96,000 logged operational records,
- 120 dates,
- 120 vehicles,
- 15 urban zones,
- observed binary actions `eco_mode`,
- measured outcomes `lateness_min`, `co2_kg`, `near_miss`, `crash`, and `on_time`.

Each trajectory is defined by the ordered pair (`date`, `vehicle_id`). Within each trajectory, records are sorted by `hour` and a stable row index. The resulting decision log stores:

$$
(s_t, a_t, r_t, s_{t+1}, d_t)
$$

for every step, where $d_t=1$ at the final step and $0$ otherwise.

Sequential features are constructed using only prior information from the same trajectory:

$$
\text{rolling\_mean\_traffic}_t
=
\frac{1}{t}\sum_{u=0}^{t-1}\text{traffic\_index}_u,
$$

$$
\text{rolling\_cumulative\_lateness}_t
=
\sum_{u=0}^{t-1}\text{lateness\_min}_u,
$$

$$
\text{rolling\_incident\_count}_t
=
\sum_{u=0}^{t-1}
(\text{near\_miss}_u + \text{crash}_u),
$$

$$
\text{prior\_reward\_primary}_t = r_{t-1},
\qquad
\text{prior\_eco\_mode}_t = a_{t-1}.
$$

Terminal next-state values are filled with neutral placeholders so that all transitions remain well-defined in tabular form.

To preserve deployability and avoid leakage, the policy state excludes post-action variables and latent simulator fields. These are retained only in the raw data or metadata, not in the deployed state used for learning.

### 4.2 Implementation Details

The pipeline is implemented in Python using `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib`. Numeric features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded.

The dataset is split temporally by date into train, validation, and test partitions with a 70/15/15 ratio. The current processed split contains:

- 67,179 training rows,
- 14,561 validation rows,
- 14,260 test rows.

The behavior policy is estimated with calibrated logistic regression. The heuristic lateness model is a gradient-boosted classifier. FQI and outcome models use histogram-based gradient-boosted regressors. All experiments use a fixed random seed of 42.

### 4.3 Offline Evaluation Protocol

All policies are evaluated on the same held-out temporal test partition and under the same logged support conditions.

#### 4.3.1 Outcome Models for Reward Decomposition

To estimate policy-level operational metrics beyond direct logged matches, we fit outcome regressors for

$$
\mathcal{Y}
=
\{
\text{lateness\_min},
\text{co2\_kg},
\text{crash},
\text{near\_miss},
\text{on\_time}
\}.
$$

For each target $y \in \mathcal{Y}$, we learn

$$
\hat{m}_y(s,a) \approx \mathbb{E}[y \mid s,a].
$$

Predicted reward under a reward specification $w$ is then reconstructed as

$$
\hat{r}_w(s,a)
=
-
\sum_{j \in \mathcal{J}} w_j \hat{m}_j(s,a),
$$

where $\mathcal{J}=\{\text{lateness\_min}, \text{crash}, \text{near\_miss}, \text{co2\_kg}\}$.

This decomposition makes it possible to report policy-level service, emissions, and safety quantities in addition to scalar reward.

#### 4.3.2 Self-Normalized IPS

As a first off-policy estimator, we use self-normalized inverse propensity scoring (SNIPS):

$$
\hat{V}_{\text{SNIPS}}(\pi)
=
\frac{
\sum_{i=1}^{n}
\frac{\mathbb{I}\{\pi(s_i)=a_i\}}{\hat{\mu}(a_i \mid s_i)} r_i
}{
\sum_{i=1}^{n}
\frac{\mathbb{I}\{\pi(s_i)=a_i\}}{\hat{\mu}(a_i \mid s_i)}
}.
$$

This estimator uses only logged matches, so it is a useful consistency check but can be high-variance when overlap is limited.

#### 4.3.3 Doubly Robust Estimation

The primary table values are reported with a doubly robust estimator that combines observed rewards with model-based predictions:

$$
\hat{V}_{\text{DR}}(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\hat{r}(s_i,\pi(s_i))
+
\frac{\mathbb{I}\{\pi(s_i)=a_i\}}{\hat{\mu}(a_i \mid s_i)}
\Big(
r_i - \hat{r}(s_i,a_i)
\Big)
\right].
$$

Here $\hat{r}(s_i,\pi(s_i))$ is the model-predicted reward under the policy action, while $\hat{r}(s_i,a_i)$ is the model-predicted reward under the logged action. This estimator is preferred because it remains informative when exact action matches are sparse, while still correcting model bias on matched samples.

#### 4.3.4 Fitted Q Evaluation

As an additional sequential diagnostic, we use fitted Q evaluation (FQE). For a fixed policy $\pi$, the Bellman target is

$$
z_i^{(k)}
=
r_i
+
\gamma(1-d_i)\hat{Q}^{(k-1)}(s'_i,\pi(s'_i)),
$$

and the evaluation model is fit by action-specific regression, analogously to FQI. The estimated policy value is then

$$
\hat{V}_{\text{FQE}}(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}\hat{Q}(s_i,\pi(s_i)).
$$

In the paper, FQE is best treated as an appendix diagnostic rather than the headline estimator.

#### 4.3.5 Bootstrap Uncertainty and Robustness Checks

Uncertainty intervals are computed by nonparametric bootstrap resampling of held-out transitions. For each bootstrap replicate $b=1,\dots,B$, we sample $n$ test rows with replacement, recompute the chosen metric, and form percentile intervals:

$$
\text{CI}_{95\%}
=
\left[
\operatorname{Quantile}_{0.025}(\{\hat{V}^{(b)}\}_{b=1}^{B}),
\operatorname{Quantile}_{0.975}(\{\hat{V}^{(b)}\}_{b=1}^{B})
\right].
$$

The publication configuration uses $B=500$ bootstrap replicates.

Robustness is evaluated on four operational slices:

- **high traffic**: top quartile of `traffic_index`,
- **rain or event**: `rain = 1` or `event_indicator = 1`,
- **tight window**: top quartile of `time_window_tightness`,
- **late day**: `hour \ge 17`.

For each segment, we report the same policy-value and operational metrics used in the overall evaluation, together with the delta relative to logged behavior.

## Summary of the Proposed Method

The full methodology can be summarized as:

1. Convert urban logistics logs into sequential trajectories over vehicle-day decisions.
2. Construct a deployable pre-decision state with causal backdoor guidance.
3. Estimate the historical behavior policy to quantify logged action support.
4. Train offline FQI policies on both broad and confounder-aware state representations.
5. Apply support-constrained policy improvement to avoid weakly supported overrides.
6. Evaluate learned policies on a held-out temporal test set using SNIPS, doubly robust estimation, FQE, bootstrap confidence intervals, and subgroup robustness analysis.

This framing positions eco-mode control as an interpretable and operationally conservative offline decision-learning problem rather than a purely predictive task.
