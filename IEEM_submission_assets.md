# IEEM Submission Asset Map

This note maps the repo outputs to the manuscript in [outline.md](/Users/anonymoize/Projects/Causal RL/outline.md) under the **synthetic urban logistics case study** framing.

## Recommended Main-Paper Assets

- Section 3 (`Problem Formulation and Proposed Method`)
  - [figures/causal_graph.png](/Users/anonymoize/Projects/Causal RL/figures/causal_graph.png): confounder-aware state design for urban logistics eco-mode control.
  - [figures/workflow.png](/Users/anonymoize/Projects/Causal RL/figures/workflow.png): offline decision-support workflow for the logistics case study.

- Section 4 (`Experimental Setup`)
  - [artifacts/dataset_metadata.json](/Users/anonymoize/Projects/Causal RL/artifacts/dataset_metadata.json): split dates, state definitions, exclusions, and seed.
  - [README.md](/Users/anonymoize/Projects/Causal RL/README.md): reproducibility commands and the honest project framing.

- Section 5 (`Results and Discussion`)
  - [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv): main policy comparison with doubly robust estimates and bootstrap intervals.
  - [figures/policy_values.png](/Users/anonymoize/Projects/Causal RL/figures/policy_values.png): overall policy-value comparison for the synthetic urban logistics case study.
  - [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv): segment-level values, delta vs logged behavior, and within-segment ranking.
  - [figures/robustness_heatmap.png](/Users/anonymoize/Projects/Causal RL/figures/robustness_heatmap.png): operational-slice comparison across traffic, rain/event, tight-window, and late-day conditions.
  - [results/reward_sensitivity.csv](/Users/anonymoize/Projects/Causal RL/results/reward_sensitivity.csv): reward-weight sensitivity comparison.
  - [figures/reward_sensitivity.png](/Users/anonymoize/Projects/Causal RL/figures/reward_sensitivity.png): reward sensitivity figure for the case-study narrative.

- Section 5.5 (`Interpretability and Deployment Safety`) — **NEW**
  - [results/feature_importance.csv](/Users/anonymoize/Projects/Causal RL/results/feature_importance.csv): permutation importance for causal FQI Q-function (averaged over both actions). Top features: `remaining_steps` (1.30), `time_window_tightness` (0.37), `speed_limit_kmph` (0.10), `traffic_index` (0.06).
  - [figures/feature_importance.png](/Users/anonymoize/Projects/Causal RL/figures/feature_importance.png): top-10 feature importance bar chart; use in Section 5.5 to support the interpretability claim.

## Recommended Appendix / Diagnostic Assets

- [results/bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/bootstrap_summary.csv): long-form confidence interval output for overall and subgroup metrics (row-level bootstrap, B=500).
- [results/cluster_bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/cluster_bootstrap_summary.csv): trajectory-level bootstrap CIs (2.9% wider on average than row-level); use to demonstrate temporal robustness.
- [results/estimator_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison; use this to justify DR as the primary estimator and to explain FQE scale difference.
- [results/ablation_comparison.csv](/Users/anonymoize/Projects/Causal RL/results/ablation_comparison.csv): sensitivity of the causal-state family; now includes `minimal_fqi` (+0.116 gap vs causal_fqi), which quantifies the floor contribution of the full causal state.
- [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv): support-constraint sweep showing **validation-set selection** of τ_μ=0.05, τ_Q=0.50; includes both `val_policy_value_dr` and `test_policy_value_dr` columns.
- [results/common_support_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/common_support_diagnostics.csv): positivity check; only 0.028% of test rows have propensity below τ_μ=0.05, minimum propensity = 0.023.
- [figures/common_support_hist.png](/Users/anonymoize/Projects/Causal RL/figures/common_support_hist.png): propensity distribution visualization; confirms excellent overlap for DR estimation.
- [results/fqe_convergence.csv](/Users/anonymoize/Projects/Causal RL/results/fqe_convergence.csv): FQE convergence across 10 iterations (mean |ΔQ| per step); use to support the FQE scale explanation in the appendix.
- [figures/fqe_convergence.png](/Users/anonymoize/Projects/Causal RL/figures/fqe_convergence.png): FQE convergence line plot.
- [results/late_day_diagnosis.csv](/Users/anonymoize/Projects/Causal RL/results/late_day_diagnosis.csv): feature distribution comparison (train vs test, hour ≥ 17) and eco recommendation rate by hour; use to explain the late_day causal_fqi underperformance.
- [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv): concise narrative cues for the overall winner and for segments where causal recommendations need tighter guardrails.
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): full policy metrics with diagnostic columns.
- [results/policy_actions.csv](/Users/anonymoize/Projects/Causal RL/results/policy_actions.csv): step-level actions for reproducibility and qualitative case inspection.

## Recommended Framing

- Present the repo as an **applied logistics case study** and **confounder-aware offline RL benchmark**, not as proof that causal FQI is the best overall policy.
- Use doubly robust estimates as the main table values.
- State explicitly that `non_causal_fqi` is the strongest overall policy in the current benchmark at `-4.457` with 95% CI `[-4.636, -4.280]`.
- State explicitly that `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) while trailing the broader non-causal comparator; the confidence intervals overlap, so the difference is not decisive.
- State explicitly that **support thresholds τ_μ=0.05 and τ_Q=0.50 were selected on the validation set**, not the test set, to avoid implicit test-set tuning.
- Treat IPS as a consistency check and FQE as an appendix diagnostic. Explain the FQE scale difference (discounted trajectory accumulation vs per-step DR) proactively rather than leaving it unexplained.
- Use the `minimal_fqi` ablation to argue that the full causal state adds +0.116 DR over the simplest 5-feature baseline, supporting the state design contribution.
- Use `feature_importance.csv` to support the interpretability claim: top features (`remaining_steps`, `time_window_tightness`, `traffic_index`) are operationally auditable.
- Present robustness results as evidence of **segment-dependent trade-offs** and the need for conservative deployment, especially because the causal policy trails logged behavior in `late_day`. Use `late_day_diagnosis.csv` to explain the failure (moderate traffic_index distribution shift of +0.014 and dispatch_delay_min shift of +0.18 min).
- Keep the logistics framing tied strictly to `eco_mode`; do not broaden the manuscript to routing, dispatching, or fleet-wide control.
