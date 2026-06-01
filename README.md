# Confounder-Aware Eco-Mode Decision Support in Synthetic Urban Logistics

This repository is an **IEEM-oriented manuscript and experiment package** for a **synthetic urban logistics case study** on **confounder-aware offline policy learning**.

The paper framing is:

**Confounder-Aware Eco-Mode Decision Support in Synthetic Urban Logistics**

The project studies one logistics decision only:

- `eco_mode = 0`: standard mode
- `eco_mode = 1`: eco mode

The repo should be read as a benchmark for **decision support for eco-mode selection** and for **intervention-aware logistics analytics**. It is not a claim that offline policy learning solves logistics control in general, nor that the learned policy is deployment-ready autonomous control.

## Honest Claim

The current evidence supports this conservative story:

- the project builds a credible **confounder-aware offline policy-learning** benchmark around an observed logistics action,
- `causal_fqi` has a point-estimate improvement over held-out `logged_behavior` replay,
- after excluding outcome-adjacent proxy fields, `causal_fqi` also edges out the non-causal FQI comparator,
- the confounder-aware framing helps with state design, interpretability, and conservative deployment logic,
- simple baselines and causal ablations remain strong, so `causal_fqi` is not the best overall policy,
- the dominance audit has 8 counterexamples across 11 estimator, reward, and segment surfaces, so the evidence explicitly rejects a universal causal-RL win,
- a separate controlled ground-truth benchmark shows that `causal_fqi` can win when the data-generating process has observed confounding plus delayed action effects,
- the claim-boundary audit records the formal reason strict universal dominance is impossible without assumptions,
- the synthetic setup does not justify broad generalization beyond this eco-mode case study.

## What This Project Does

The pipeline turns [causalog_synthetic_urban_logistics.csv](causalog_synthetic_urban_logistics.csv) into an offline decision-learning benchmark where each record belongs to a `date`-`vehicle_id` trajectory.

It:

- builds sequential decision logs with current-state and next-state features,
- constructs rewards that balance service, safety, and emissions,
- estimates the observed logged-action policy,
- trains heuristic, non-causal FQI, and confounder-aware FQI policies,
- evaluates them with off-policy estimators,
- produces figures and tables for an applied logistics paper.

The primary reward is:

```text
r_t = -(lateness_min + 10*crash + 2*near_miss + 0.2*co2_kg)
```

The weights are explicit lateness-minute-equivalent utility tradeoffs, not
deployment-cost estimates learned from the test set: one crash is treated as
10 lateness minutes, one near miss as 2 lateness minutes, and 1 kg CO2 as
0.2 lateness minutes. [results/reward_weight_calibration.csv](results/reward_weight_calibration.csv)
audits these weights against train/validation/test component scales.

Two sensitivity rewards are also evaluated:

- `service_heavy`
- `sustainability_heavy`

## Dataset Semantics

The implementation treats the provider email plus the JSON data dictionary as the semantic source of truth.

- one row is one dispatch decision / trip segment / decision epoch,
- one daily trajectory is defined by `(date, vehicle_id)`,
- `eco_mode` is an epoch-level controllable action and can change during the day,
- `hour` is the decision-time bucket, not departure or arrival time,
- `dispatch_delay_min` is pre-movement waiting or batching time,
- `compatibility_violation` is retained only in the broader non-causal comparator,
- `distance_km` and `risk_score` are excluded from learned policy state because the raw dictionary makes their decision-time status outcome-adjacent or ambiguous,
- `crash`, `near_miss`, `lateness_min`, `co2_kg`, `fuel_liters`, `travel_time_min`, `avg_speed_kmph`, `harsh_events`, `distance_km`, `risk_score`, and latent simulator columns are excluded from current-step deployable state.

The decision log is ordered by `date`, `vehicle_id`, `hour`, and original CSV row order. In the current dataset, about 19.4% of `(date, vehicle_id, hour)` groups contain multiple rows, so the within-hour sequence is explicitly treated as a source-row tie-break assumption rather than an observed finer timestamp.

## Current Empirical Message

Using the publication defaults selected by the support sweep,

- `min_propensity = 0.05`
- `q_gap_threshold = 0.50`

the overall doubly robust results in [results/main_results_table.csv](results/main_results_table.csv) are led by:

- `heuristic_risk_rule`: `-4.636` with 95% bootstrap CI `[-4.782, -4.495]`
- `causal_no_vehicle_id_fqi`: `-4.670` with 95% bootstrap CI `[-4.847, -4.486]`
- `causal_no_history_fqi`: `-4.687` with 95% bootstrap CI `[-4.862, -4.502]`
- `causal_fqi`: `-4.694` with 95% bootstrap CI `[-4.886, -4.530]`
- `non_causal_fqi`: `-4.712` with 95% bootstrap CI `[-4.895, -4.543]`
- `logged_behavior`: `-4.826` with 95% bootstrap CI `[-5.055, -4.577]`

The honest interpretation is:

- `causal_fqi` now outperforms `non_causal_fqi` after excluding `distance_km` and `risk_score` from learned policy state,
- `causal_fqi` remains competitive and has a point-estimate improvement over `logged_behavior`,
- `heuristic_risk_rule` and causal ablations still outperform the full confounder-aware FQI point estimate,
- the confounder-aware state family is fairly stable under the history and vehicle-identity ablations,
- the main value of the confounder-aware framing is **credible pre-decision state design and conservative decision support**, not proof that the full causal FQI policy dominates every baseline.

The subgroup results in [results/robustness.csv](results/robustness.csv) reinforce that point:

- `non_causal_fqi` remains above `causal_fqi` in `high_traffic`, `tight_window`, and `late_day`,
- `causal_fqi` is above `non_causal_fqi` in `rain_or_event`,
- `causal_fqi` has favorable point estimates relative to `logged_behavior` in `high_traffic`, `rain_or_event`, and `tight_window`,
- `causal_fqi` falls slightly below `logged_behavior` in `late_day`,
- simpler policies such as `heuristic_risk_rule` or `never_eco` can outperform the confounder-aware policy in stressed segments.

That means the repo supports a publishable logistics story about **segment-dependent trade-offs** and the need for **conservative deployment guardrails**.

The objective-alignment check in [results/dominance_audit.csv](results/dominance_audit.csv) separates the RL objective from the contextual OPE tables:

- `causal_fqi` is top only on the discounted trajectory-return FQE row (`-17.027` vs `non_causal_fqi` at `-17.053`),
- the broader confounder-aware FQI family is also top on overall SNIPS and the tight-window DR segment,
- 8 of 11 audited surfaces remain counterexamples to universal causal-RL-family dominance.

This suggests the current methodology was not simply “missing” a universal win. It was mixing a sequential RL training objective with conservative per-decision deployment evaluation, and the correct conclusion depends on which objective is being evaluated.

The controlled benchmark in [results/ground_truth_policy_audit.csv](results/ground_truth_policy_audit.csv) is the positive sanity check. In a known structural environment where the logging policy is confounded and eco-mode has delayed effects, `causal_fqi` ranks first under true interventional rollout value and beats the omitted-confounder FQI comparator. That is evidence for the mechanism under specified assumptions, not a universal theorem.

[results/claim_boundary_audit.csv](results/claim_boundary_audit.csv) is the formal claim boundary. It includes an action-irrelevant MDP counterexample, also written to [results/action_irrelevant_counterexample.csv](results/action_irrelevant_counterexample.csv): if `R(s,0)=R(s,1)` and `P(s'|s,0)=P(s'|s,1)` for all states, every policy has the same value, so no method can strictly dominate universally. Any valid theorem must therefore be conditional on action effects, overlap, correct adjustment, consistent estimation, and a positive value gap.

The theorem-assumption audit in [results/theorem_assumption_audit.csv](results/theorem_assumption_audit.csv) maps that bounded theorem template to evidence. In the controlled DGP, the assumptions are satisfied or positively checked. In the main benchmark, only overlap is empirically supported for the thresholded policies; sufficient adjustment, consistent value learning, objective alignment, and a positive comparator gap remain unverified or objective-dependent.

The strongest valid theorem template is therefore bounded: under conditional exchangeability, overlap, consistent fitted-Q estimation, objective alignment, action relevance, and a positive value gap over a misspecified comparator, causal FQI can asymptotically outperform that comparator under the shared objective. This is not a universal theorem; it is an assumption-bound statement.

[results/policy_difference_bootstrap.csv](results/policy_difference_bootstrap.csv) adds paired trajectory-bootstrap intervals for the most important gaps. The `causal_fqi` point-estimate gap over `non_causal_fqi` is `+0.018` with paired 95% CI `[-0.070, 0.115]`, and its gap over `logged_behavior` is `+0.132` with paired 95% CI `[-0.001, 0.279]`. These intervals reinforce the point-estimate, not decisive-dominance, interpretation.

[results/oracle_policy_sensitivity.csv](results/oracle_policy_sensitivity.csv) separates the deployed support-constrained policy from methodological sensitivities. Greedy `causal_fqi` without fallback reaches `-4.675`, while a non-deployable latent-state `oracle_fqi` reaches `-4.550`; the oracle row is a diagnostic upper-bound check, not a deployable policy.

## Method Summary

The implementation follows this logic:

1. Build trajectories using `date` and `vehicle_id`.
2. Add sequential features:
   - `step_idx`
   - `remaining_steps`
   - `rolling_mean_traffic`
   - `rolling_cumulative_lateness`
   - `rolling_incident_count`
   - `prior_reward_primary`
   - `prior_eco_mode`
3. Restrict the confounder-aware policy state to **pre-decision deployable covariates** motivated by a domain DAG.
4. Exclude:
   - post-action or outcome-adjacent fields like `avg_speed_kmph`, `travel_time_min`, `fuel_liters`, `co2_kg`, `crash`, `near_miss`, `lateness_min`, `on_time`, `harsh_events`, `distance_km`, `risk_score`
   - latent simulator columns from the deployable policy
5. Estimate the observed logged-action policy with a calibrated classifier.
6. Train:
   - `logged_behavior`
   - `always_eco`
   - `never_eco`
   - `heuristic_risk_rule`
   - `non_causal_fqi`
   - `causal_fqi`
   - `causal_no_history_fqi`
   - `causal_no_vehicle_id_fqi`
7. Evaluate with:
   - doubly robust estimation as the primary policy-value estimator
   - self-normalized IPS as a secondary check
   - fitted Q evaluation as a discounted trajectory-return diagnostic
   - bootstrap confidence intervals
   - subgroup robustness analysis
   - support-threshold sweeps for conservative overrides

The causal-adjustment story is intentionally modest: this is **confounder-aware offline policy learning** inside a synthetic logistics benchmark, not a claim of strong causal identification or autonomous deployment readiness.

## Project Layout

Key paper-facing files:

- [paper/paper.tex](paper/paper.tex): current IEEE-style manuscript source
- [paper/paper.pdf](paper/paper.pdf): compiled manuscript PDF
- [paper/IEEEtran.cls](paper/IEEEtran.cls): IEEE conference class used to compile the manuscript
- [figures/workflow.png](figures/workflow.png), [figures/policy_values.png](figures/policy_values.png), and [figures/robustness_heatmap.png](figures/robustness_heatmap.png): tracked manuscript figures

Key implementation files:

- [causal_rl](causal_rl): full pipeline
- [Makefile](Makefile): pipeline and verification entry points
- [Dockerfile](Dockerfile): repeatable test environment
- [requirements.txt](requirements.txt): pinned Python dependencies

Important modules:

- [causal_rl/build_dataset.py](causal_rl/build_dataset.py): builds the offline decision log
- [causal_rl/policy_learning.py](causal_rl/policy_learning.py): trains behavior, heuristic, and FQI policies
- [causal_rl/evaluate.py](causal_rl/evaluate.py): runs off-policy evaluation and robustness analysis
- [causal_rl/report.py](causal_rl/report.py): generates figures and paper-facing tables

## How To Run

From the repository root:

```bash
make all
```

This runs:

```bash
python3 -m causal_rl.build_dataset
python3 -m causal_rl.train_policies --n-jobs auto
python3 -m causal_rl.evaluate --bootstrap-reps 500 --run-ablations --n-jobs auto
python3 -m causal_rl.ground_truth_benchmark
python3 -m causal_rl.claim_boundary
python3 -m causal_rl.theorem_assumptions
python3 -m causal_rl.report
```

`--n-jobs auto` uses one fewer than the detected CPU core count. Override it with
`N_JOBS=4 make all` or `CAUSAL_RL_N_JOBS=4` if you want to reserve more CPU for
other work.

For a release-style verification:

```bash
make release-check
```

To build and check the manuscript only:

```bash
make paper-check
```

## Docker

Docker can make the Python environment repeatable when the host or cloud runner
does not already have `numpy`, `pandas`, `scikit-learn`, and the other pinned
packages installed:

```bash
make docker-test
```

This builds the image from [Dockerfile](Dockerfile)
and runs a Python compile check inside it. To open a shell in the same image:

```bash
make docker-shell
```

Important limitation: Docker still needs the dependencies at image build time.
If the target environment cannot reach PyPI, build and push the image from a
machine that has network access, or prepare an offline wheel bundle:

```bash
python3 -m pip download -r requirements.txt -d vendor/wheels
make docker-test
```

When `vendor/wheels` contains downloaded wheels, the Docker build installs with
`--no-index` from that local bundle instead of trying PyPI.

## Main Outputs

Generated artifacts are intentionally ignored and are rebuilt with `make all`:

- [artifacts/decision_log.csv](artifacts/decision_log.csv): cleaned offline decision log
- [artifacts/dataset_metadata.json](artifacts/dataset_metadata.json): metadata and confounder-aware state definitions
- [results/metrics.csv](results/metrics.csv): full policy metrics
- [results/main_results_table.csv](results/main_results_table.csv): compact policy comparison table
- [results/robustness.csv](results/robustness.csv): subgroup robustness analysis
- [results/reward_weight_calibration.csv](results/reward_weight_calibration.csv): reward-weight tradeoff and component-scale audit
- [results/reward_sensitivity.csv](results/reward_sensitivity.csv): reward sensitivity study
- [results/bootstrap_summary.csv](results/bootstrap_summary.csv): long-form confidence intervals
- [results/estimator_diagnostics.csv](results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison
- [results/causal_design_mapping.csv](results/causal_design_mapping.csv): DAG node, treatment/outcome, adjustment-set, and exclusion mapping
- [results/heuristic_diagnostics.csv](results/heuristic_diagnostics.csv): validation-frozen heuristic behavior by lateness-risk decile and segment
- [results/policy_difference_bootstrap.csv](results/policy_difference_bootstrap.csv): paired trajectory-bootstrap intervals for key DR policy gaps
- [results/oracle_policy_sensitivity.csv](results/oracle_policy_sensitivity.csv): greedy/no-fallback and non-deployable latent-oracle sensitivity checks
- [results/dominance_audit.csv](results/dominance_audit.csv): objective-alignment and universal-dominance falsification audit
- [results/ground_truth_policy_audit.csv](results/ground_truth_policy_audit.csv): controlled structural-DGP sanity check with true interventional rollout values
- [results/claim_boundary_audit.csv](results/claim_boundary_audit.csv): formal claim-boundary audit separating falsified, supported-under-assumptions, and impossible claims
- [results/action_irrelevant_counterexample.csv](results/action_irrelevant_counterexample.csv): executable counterexample showing all policies tie when rewards and transitions ignore actions
- [results/theorem_assumption_audit.csv](results/theorem_assumption_audit.csv): assumption-by-assumption audit for the bounded causal-RL theorem template
- [results/ablation_comparison.csv](results/ablation_comparison.csv): confounder-aware ablation summary
- [results/support_threshold_sweep.csv](results/support_threshold_sweep.csv): conservative override sweep
- [results/interpretation_summary.csv](results/interpretation_summary.csv): narrative cues for overall and subgroup interpretation
- [results/runtime_benchmark.csv](results/runtime_benchmark.csv): saved-model test-set inference timing; engineering context only, not a policy-value claim
- [figures/policy_values.png](figures/policy_values.png)
- [figures/robustness_heatmap.png](figures/robustness_heatmap.png)
- [figures/reward_sensitivity.png](figures/reward_sensitivity.png)
- [figures/causal_graph.png](figures/causal_graph.png)
- [figures/workflow.png](figures/workflow.png)

## Reproducibility Notes

- Python version used in the current environment: `3.12.13`
- Pinned dependencies are listed in [requirements.txt](requirements.txt)
- Full reproducible run: `make all`
- Manuscript build: `make paper`
- Release candidate regeneration plus manuscript checks: `make release-check`
- Publication-mode evaluation command:

```bash
python3 -m causal_rl.evaluate --bootstrap-reps 500 --run-ablations --n-jobs auto
```

- Saved-model inference runtime benchmark:

```bash
python3 -m causal_rl.benchmark_runtime
```

- Controlled ground-truth benchmark command:

```bash
python3 -m causal_rl.ground_truth_benchmark
```

- Claim-boundary audit command:

```bash
python3 -m causal_rl.claim_boundary
```

- Theorem-assumption audit command:

```bash
python3 -m causal_rl.theorem_assumptions
```

- Fixed random seed: `42`
- No `torch` dependency is required

## Intended Audience

This repo is set up for:

- drafting and revising an applied logistics paper,
- benchmarking offline policy-learning methods on a synthetic logistics dataset,
- generating tables, diagnostics, and figures for an IEEM-style submission,
- studying how confounding-aware state design affects operational decision support.
