# Causal Reinforcement Learning for Urban Logistics

This repository is an **IEEM-oriented manuscript and experiment package** for a **synthetic urban logistics case study** on **confounder-aware offline RL**.

The paper framing is:

**Causal Reinforcement Learning for Urban Logistics: A Confounder-Aware Offline Case Study of Eco-Mode Control**

The project studies one logistics decision only:

- `eco_mode = 0`: standard mode
- `eco_mode = 1`: eco mode

The repo should be read as a benchmark for **decision support for eco-mode selection** and for **intervention-aware logistics analytics**. It is not a claim that causal RL solves logistics control in general.

## Honest Claim

The current evidence supports this conservative story:

- the project builds a credible **confounder-aware offline RL** benchmark around an observed logistics action,
- `causal_fqi` improves over `logged_behavior`,
- the causal framing helps with state design, interpretability, and conservative deployment logic,
- `non_causal_fqi` is still the strongest overall policy in the current benchmark,
- the synthetic setup does not justify broad generalization beyond this eco-mode case study.

## What This Project Does

The pipeline turns [causalog_synthetic_urban_logistics.csv](/Users/anonymoize/Projects/Causal RL/causalog_synthetic_urban_logistics.csv) into an offline decision-learning benchmark where each record belongs to a `date`-`vehicle_id` trajectory.

It:

- builds sequential decision logs with current-state and next-state features,
- constructs rewards that balance service, safety, and emissions,
- estimates the logged behavior policy,
- trains heuristic, non-causal FQI, and causal FQI policies,
- evaluates them with off-policy estimators,
- produces figures and tables for an applied logistics paper.

The primary reward is:

```text
r_t = -(lateness_min + 10*crash + 2*near_miss + 0.2*co2_kg)
```

Two sensitivity rewards are also evaluated:

- `service_heavy`
- `sustainability_heavy`

## Current Empirical Message

Using the publication defaults selected by the support sweep,

- `min_propensity = 0.05`
- `q_gap_threshold = 0.50`

the overall doubly robust results in [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv) are led by:

- `non_causal_fqi`: `-4.457` with 95% bootstrap CI `[-4.636, -4.280]`
- `heuristic_risk_rule`: `-4.638` with 95% bootstrap CI `[-4.787, -4.500]`
- `causal_no_vehicle_id_fqi`: `-4.670` with 95% bootstrap CI `[-4.847, -4.486]`
- `causal_no_history_fqi`: `-4.687` with 95% bootstrap CI `[-4.862, -4.502]`
- `causal_fqi`: `-4.694` with 95% bootstrap CI `[-4.886, -4.530]`
- `logged_behavior`: `-4.826` with 95% bootstrap CI `[-5.055, -4.577]`

The honest interpretation is:

- `non_causal_fqi` is the strongest overall comparator in this benchmark,
- `causal_fqi` remains competitive and improves over `logged_behavior`,
- the causal-state family is fairly stable under the history and vehicle-identity ablations,
- the main value of the causal framing is **credible pre-decision state design and conservative decision support**, not proof of universal performance dominance.

The subgroup results in [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv) reinforce that point:

- `non_causal_fqi` is best in `high_traffic`, `rain_or_event`, `tight_window`, and `late_day`,
- `causal_fqi` still beats `logged_behavior` in `high_traffic`, `rain_or_event`, and `tight_window`,
- `causal_fqi` falls slightly below `logged_behavior` in `late_day`,
- simpler policies such as `heuristic_risk_rule` or `never_eco` can outperform the causal policy in stressed segments.

That means the repo supports a publishable logistics story about **segment-dependent trade-offs** and the need for **conservative deployment guardrails**.

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
3. Restrict the causal policy state to **pre-decision deployable covariates** motivated by a domain DAG.
4. Exclude:
   - post-action outcomes like `co2_kg`, `crash`, `near_miss`, `lateness_min`, `on_time`
   - latent simulator columns from the deployable policy
5. Estimate the logged behavior policy with a calibrated classifier.
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
   - fitted Q evaluation as an appendix diagnostic
   - bootstrap confidence intervals
   - subgroup robustness analysis
   - support-threshold sweeps for conservative overrides

The causal story is intentionally modest: this is **confounder-aware offline policy learning** inside a synthetic logistics benchmark, not a claim of strong causal identification or autonomous deployment readiness.

## Project Layout

Key paper-facing files:

- [outline.md](/Users/anonymoize/Projects/Causal RL/outline.md): manuscript outline aligned to the logistics case-study framing
- [paper_methodology.md](/Users/anonymoize/Projects/Causal RL/paper_methodology.md): methods section with the confounder-aware offline RL pipeline
- [IEEM_submission_assets.md](/Users/anonymoize/Projects/Causal RL/IEEM_submission_assets.md): asset map for the IEEM-style submission package

Key code and artifacts:

- [causal_rl](/Users/anonymoize/Projects/Causal RL/causal_rl): full pipeline
- [artifacts](/Users/anonymoize/Projects/Causal RL/artifacts): decision log and dataset metadata
- [models](/Users/anonymoize/Projects/Causal RL/models): saved trained models
- [results](/Users/anonymoize/Projects/Causal RL/results): metrics, robustness, ablations, and policy actions
- [figures](/Users/anonymoize/Projects/Causal RL/figures): paper-ready visuals
- [tests](/Users/anonymoize/Projects/Causal RL/tests): dataset and evaluation tests

Important modules:

- [causal_rl/build_dataset.py](/Users/anonymoize/Projects/Causal RL/causal_rl/build_dataset.py): builds the offline decision log
- [causal_rl/policy_learning.py](/Users/anonymoize/Projects/Causal RL/causal_rl/policy_learning.py): trains behavior, heuristic, and FQI policies
- [causal_rl/evaluate.py](/Users/anonymoize/Projects/Causal RL/causal_rl/evaluate.py): runs off-policy evaluation and robustness analysis
- [causal_rl/report.py](/Users/anonymoize/Projects/Causal RL/causal_rl/report.py): generates figures and paper-facing tables

## How To Run

From the repository root:

```bash
make all
```

This runs:

```bash
python3 -m causal_rl.build_dataset
python3 -m causal_rl.train_policies
python3 -m causal_rl.evaluate --bootstrap-reps 500 --run-ablations
python3 -m causal_rl.report
```

For a release-style verification:

```bash
make release-check
```

To run tests only:

```bash
pytest -q
```

## Main Outputs

Generated artifacts:

- [artifacts/decision_log.csv](/Users/anonymoize/Projects/Causal RL/artifacts/decision_log.csv): cleaned offline decision log
- [artifacts/dataset_metadata.json](/Users/anonymoize/Projects/Causal RL/artifacts/dataset_metadata.json): metadata and causal-state definitions
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): full policy metrics
- [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv): compact policy comparison table
- [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv): subgroup robustness analysis
- [results/reward_sensitivity.csv](/Users/anonymoize/Projects/Causal RL/results/reward_sensitivity.csv): reward sensitivity study
- [results/bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/bootstrap_summary.csv): long-form confidence intervals
- [results/estimator_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison
- [results/ablation_comparison.csv](/Users/anonymoize/Projects/Causal RL/results/ablation_comparison.csv): causal ablation summary
- [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv): conservative override sweep
- [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv): narrative cues for overall and subgroup interpretation
- [figures/policy_values.png](/Users/anonymoize/Projects/Causal RL/figures/policy_values.png)
- [figures/robustness_heatmap.png](/Users/anonymoize/Projects/Causal RL/figures/robustness_heatmap.png)
- [figures/reward_sensitivity.png](/Users/anonymoize/Projects/Causal RL/figures/reward_sensitivity.png)
- [figures/causal_graph.png](/Users/anonymoize/Projects/Causal RL/figures/causal_graph.png)
- [figures/workflow.png](/Users/anonymoize/Projects/Causal RL/figures/workflow.png)

## Reproducibility Notes

- Python version used in the current environment: `3.12.13`
- Pinned dependencies are listed in [requirements.txt](/Users/anonymoize/Projects/Causal RL/requirements.txt)
- Full reproducible run: `make all`
- Release candidate verification: `make release-check`
- Publication-mode evaluation command:

```bash
python3 -m causal_rl.evaluate --bootstrap-reps 500 --run-ablations
```

- Fixed random seed: `42`
- No `torch` dependency is required

Current tests cover:

- no post-action state leakage,
- trajectory construction and temporal splitting,
- binary policy outputs,
- support-constrained fallback behavior,
- ablation policy registration,
- bootstrap determinism and interval ordering,
- support-threshold sweep completeness.

## Intended Audience

This repo is set up for:

- drafting and revising an applied logistics paper,
- benchmarking offline RL policies on a synthetic logistics dataset,
- generating tables, diagnostics, and figures for an IEEM-style submission,
- studying how confounding-aware state design affects operational decision support.
