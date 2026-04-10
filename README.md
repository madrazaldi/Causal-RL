# Causal RL for Urban Logistics Eco-Mode Control

This repository contains a small end-to-end research pipeline for studying **confounder-aware offline reinforcement learning** on a synthetic urban logistics dataset.

The project reframes the original paper idea away from shipping upgrades and late-delivery intervention proxies, and instead focuses on a cleaner logged action that is directly present in the data: `eco_mode`.

The current paper framing is:

**Confounder-Aware Offline Reinforcement Learning for Eco-Mode Control in Urban Logistics**

## What This Project Does

The code turns `causalog_synthetic_urban_logistics.csv` into an offline decision-learning benchmark where each record belongs to a `date`-`vehicle_id` trajectory.

The pipeline:

- builds sequential decision logs with current-state and next-state features,
- constructs a reward that balances timeliness, safety, and emissions,
- estimates a logged behavior policy,
- trains heuristic, non-causal FQI, and causal FQI policies,
- evaluates them with off-policy estimators,
- generates paper-ready tables and figures.

The primary action is:

- `eco_mode = 0`: standard mode
- `eco_mode = 1`: eco mode

The primary reward is:

```text
r_t = -(lateness_min + 10*crash + 2*near_miss + 0.2*co2_kg)
```

Two additional reward variants are also evaluated:

- `service_heavy`
- `sustainability_heavy`

## Project Layout

Key files and directories:

- [outline.md](/Users/anonymoize/Projects/Causal RL/outline.md): revised paper outline aligned to the eco-mode framing
- [causalog_synthetic_urban_logistics.csv](/Users/anonymoize/Projects/Causal RL/causalog_synthetic_urban_logistics.csv): raw synthetic dataset
- [causal_rl](/Users/anonymoize/Projects/Causal RL/causal_rl): package containing the full pipeline
- [artifacts](/Users/anonymoize/Projects/Causal RL/artifacts): built decision log and dataset metadata
- [models](/Users/anonymoize/Projects/Causal RL/models): saved trained models
- [results](/Users/anonymoize/Projects/Causal RL/results): metrics, robustness results, reward sensitivity, and policy actions
- [figures](/Users/anonymoize/Projects/Causal RL/figures): paper-ready figures
- [tests](/Users/anonymoize/Projects/Causal RL/tests): dataset and policy tests

Important modules:

- [causal_rl/build_dataset.py](/Users/anonymoize/Projects/Causal RL/causal_rl/build_dataset.py): builds the offline decision log
- [causal_rl/policy_learning.py](/Users/anonymoize/Projects/Causal RL/causal_rl/policy_learning.py): trains behavior, heuristic, and FQI policies
- [causal_rl/evaluate.py](/Users/anonymoize/Projects/Causal RL/causal_rl/evaluate.py): runs OPE and robustness analysis
- [causal_rl/report.py](/Users/anonymoize/Projects/Causal RL/causal_rl/report.py): generates tables and figures

## Method Summary

The implementation follows this logic:

1. Build trajectories using `date` and `vehicle_id`.
2. Add sequential features such as:
   - `step_idx`
   - `remaining_steps`
   - `rolling_mean_traffic`
   - `rolling_cumulative_lateness`
   - `rolling_incident_count`
   - `prior_reward_primary`
   - `prior_eco_mode`
3. Use only **pre-decision deployable covariates** in the main policy state.
4. Exclude:
   - post-action/outcome variables like `co2_kg`, `crash`, `near_miss`, `lateness_min`, `on_time`
   - latent columns from the deployable policy
5. Estimate the logged behavior policy with a calibrated classifier.
6. Train:
   - logged behavior baseline
   - always-eco baseline
   - never-eco baseline
   - heuristic risk-rule baseline
   - non-causal fitted Q iteration
   - causal fitted Q iteration
7. Evaluate with:
   - self-normalized IPS
   - doubly robust estimation
   - fitted Q evaluation
   - subgroup robustness analysis

The causal story is intentionally modest: this is **confounder-aware offline policy learning**, not a claim of strong real-world causal identification.

## How To Run

From the repository root:

```bash
python3 -m causal_rl.build_dataset
python3 -m causal_rl.train_policies
python3 -m causal_rl.evaluate
python3 -m causal_rl.report
```

To run tests:

```bash
pytest -q
```

## Main Outputs

Generated artifacts:

- [artifacts/decision_log.csv](/Users/anonymoize/Projects/Causal RL/artifacts/decision_log.csv): cleaned offline decision log
- [artifacts/dataset_metadata.json](/Users/anonymoize/Projects/Causal RL/artifacts/dataset_metadata.json): metadata and causal-state definitions
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): main policy comparison
- [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv): subgroup robustness analysis
- [results/reward_sensitivity.csv](/Users/anonymoize/Projects/Causal RL/results/reward_sensitivity.csv): reward sensitivity study
- [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv): compact paper-facing result table
- [figures/policy_values.png](/Users/anonymoize/Projects/Causal RL/figures/policy_values.png)
- [figures/robustness_heatmap.png](/Users/anonymoize/Projects/Causal RL/figures/robustness_heatmap.png)
- [figures/reward_sensitivity.png](/Users/anonymoize/Projects/Causal RL/figures/reward_sensitivity.png)
- [figures/causal_graph.png](/Users/anonymoize/Projects/Causal RL/figures/causal_graph.png)
- [figures/workflow.png](/Users/anonymoize/Projects/Causal RL/figures/workflow.png)

## Current First-Run Results

On the current synthetic dataset and current feature/reward design, the top doubly robust policy values in [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv) are:

- `non_causal_fqi`: `-4.519`
- `heuristic_risk_rule`: `-4.628`
- `causal_fqi`: `-4.653`
- `logged_behavior`: `-4.747`

This means the current implementation is working end to end, but it **does not yet support a strong paper claim that the causal policy outperforms the non-causal one**.

That is an important research caveat, not a code bug.

## Recommended Next Steps

If the goal is to turn this into a stronger conference paper, the next improvements should focus on:

- refining the causal backdoor feature set,
- tightening the support-constraint rule,
- revisiting the reward weights,
- improving the subgroup analysis and interpretation,
- adding stronger ablations that explain when causal adjustment helps,
- optionally revisiting the synthetic data generation assumptions if the paper needs clearer causal-policy gains.

## Reproducibility Notes

- Python stack used by the current code: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- No `torch` dependency is required for this v1 pipeline
- Tests currently cover:
  - no post-action state leakage,
  - trajectory construction and date-based splitting,
  - binary policy outputs,
  - support-constrained fallback behavior

## Intended Audience

This repo is set up for:

- paper drafting and experiment iteration,
- offline RL benchmarking on the synthetic logistics data,
- generating tables/figures for an IEEM-style applied analytics paper,
- future refinement of the causal-RL story before submission.
