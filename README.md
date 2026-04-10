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
- [Makefile](/Users/anonymoize/Projects/Causal RL/Makefile): reproducible one-command run targets
- [requirements.txt](/Users/anonymoize/Projects/Causal RL/requirements.txt): pinned Python dependency manifest
- [IEEM_submission_assets.md](/Users/anonymoize/Projects/Causal RL/IEEM_submission_assets.md): section-by-section asset map for the current manuscript outline
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
   - `causal_no_history_fqi` ablation
   - `causal_no_vehicle_id_fqi` ablation
7. Evaluate with:
   - doubly robust estimation as the primary policy-value estimator
   - self-normalized IPS as a secondary check
   - fitted Q evaluation as an appendix diagnostic
   - bootstrap confidence intervals for overall and subgroup results
   - subgroup robustness analysis with delta-vs-logged-behavior ranking
   - support-threshold sweeps for conservative policy override selection

The causal story is intentionally modest: this is **confounder-aware offline policy learning**, not a claim of strong real-world causal identification.

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

To run the release check:

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
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): main policy comparison
- [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv): subgroup robustness analysis
- [results/reward_sensitivity.csv](/Users/anonymoize/Projects/Causal RL/results/reward_sensitivity.csv): reward sensitivity study
- [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv): compact paper-facing result table
- [results/bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/bootstrap_summary.csv): long-form bootstrap confidence intervals
- [results/estimator_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison
- [results/ablation_comparison.csv](/Users/anonymoize/Projects/Causal RL/results/ablation_comparison.csv): causal ablation summary
- [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv): support-constraint sweep used to select the publication default
- [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv): narrative summary of overall and subgroup winners
- [figures/policy_values.png](/Users/anonymoize/Projects/Causal RL/figures/policy_values.png)
- [figures/robustness_heatmap.png](/Users/anonymoize/Projects/Causal RL/figures/robustness_heatmap.png)
- [figures/reward_sensitivity.png](/Users/anonymoize/Projects/Causal RL/figures/reward_sensitivity.png)
- [figures/causal_graph.png](/Users/anonymoize/Projects/Causal RL/figures/causal_graph.png)
- [figures/workflow.png](/Users/anonymoize/Projects/Causal RL/figures/workflow.png)

## Current Publication-Mode Results

Using the conservative support rule selected by the sweep,

- `min_propensity = 0.05`
- `q_gap_threshold = 0.50`

the top doubly robust policy values in [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv) are:

- `non_causal_fqi`: `-4.457` with 95% bootstrap CI `[-4.636, -4.280]`
- `heuristic_risk_rule`: `-4.638` with 95% bootstrap CI `[-4.787, -4.500]`
- `causal_no_vehicle_id_fqi`: `-4.670` with 95% bootstrap CI `[-4.847, -4.486]`
- `causal_no_history_fqi`: `-4.687` with 95% bootstrap CI `[-4.862, -4.502]`
- `causal_fqi`: `-4.694` with 95% bootstrap CI `[-4.886, -4.530]`
- `logged_behavior`: `-4.826` with 95% bootstrap CI `[-5.055, -4.577]`

The current evidence supports a conservative paper claim:

- confounder-aware offline RL improves over logged behavior,
- the causal feature design is competitive and interpretable,
- but the current synthetic setup still does **not** justify claiming that causal FQI dominates the non-causal comparator.

The subgroup outputs also show an important operational nuance: in `high_traffic`, [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv) flags `never_eco` as the best-performing policy, which is useful deployment guidance rather than a failure of the framework.

## Publication-Oriented Additions

The repo now includes the main publication-readiness upgrades that were previously only listed as next steps:

- bootstrap confidence intervals for overall and subgroup estimates,
- targeted causal ablations for history and vehicle identity,
- a support-threshold sweep with a documented conservative default,
- subgroup ranking and delta-vs-logged-behavior outputs,
- estimator diagnostics that justify using DR as the main table value,
- a pinned dependency manifest and `make all` reproducibility path,
- an IEEM submission asset map tied to [outline.md](/Users/anonymoize/Projects/Causal RL/outline.md).

Remaining future work is now optional rather than blocking:

- revisiting reward weights if a different managerial objective is preferred,
- refining the synthetic data generator only if the paper requires a stronger causal-policy-gain story,
- expanding the appendix with more case studies or qualitative action traces.

## Reproducibility Notes

- Python version used in the current environment: `3.12.13`
- Pinned dependencies are listed in [requirements.txt](/Users/anonymoize/Projects/Causal RL/requirements.txt)
- Current stack versions:
  - `numpy==2.4.3`
  - `pandas==3.0.2`
  - `scikit-learn==1.8.0`
  - `matplotlib==3.10.8`
  - `seaborn==0.13.2`
  - `joblib==1.5.2`
- Full reproducible run: `make all`
- Release candidate verification: `make release-check`
- Publication-mode evaluation command:

```bash
python3 -m causal_rl.evaluate --bootstrap-reps 500 --run-ablations
```

- Fixed random seed: `42`
- No `torch` dependency is required for this v1 pipeline
- Tests currently cover:
  - no post-action state leakage,
  - trajectory construction and date-based splitting,
  - binary policy outputs,
  - support-constrained fallback behavior,
  - ablation policy registration,
  - bootstrap determinism and interval ordering,
  - support-threshold sweep completeness

## Intended Audience

This repo is set up for:

- paper drafting and experiment iteration,
- offline RL benchmarking on the synthetic logistics data,
- generating tables, diagnostics, and figures for an IEEM-style applied analytics paper,
- future refinement of the causal-RL story before submission.
