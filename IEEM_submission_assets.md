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

## Recommended Appendix / Diagnostic Assets

- [results/bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/bootstrap_summary.csv): long-form confidence interval output for overall and subgroup metrics.
- [results/estimator_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison; use this to justify DR as the primary estimator.
- [results/ablation_comparison.csv](/Users/anonymoize/Projects/Causal RL/results/ablation_comparison.csv): sensitivity of the causal-state family to removing history or vehicle identity.
- [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv): support-constraint sweep used to justify the conservative default thresholds.
- [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv): concise narrative cues for the overall winner and for segments where causal recommendations need tighter guardrails.
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): full policy metrics with diagnostic columns.
- [results/policy_actions.csv](/Users/anonymoize/Projects/Causal RL/results/policy_actions.csv): step-level actions for reproducibility and qualitative case inspection.

## Recommended Framing

- Present the repo as an **applied logistics case study** and **confounder-aware offline RL benchmark**, not as proof that causal FQI is the best overall policy.
- Use doubly robust estimates as the main table values.
- State explicitly that `non_causal_fqi` is the strongest overall policy in the current benchmark at `-4.457` with 95% CI `[-4.636, -4.280]`.
- State explicitly that `causal_fqi` improves over `logged_behavior` (`-4.694` vs `-4.826`) while trailing the broader non-causal comparator.
- Treat IPS as a consistency check and FQE as an appendix diagnostic unless its scale is separately reconciled.
- Document the conservative support defaults as `min_propensity=0.05` and `q_gap_threshold=0.50`, justified by [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv).
- Present robustness results as evidence of **segment-dependent trade-offs** and the need for conservative deployment, especially because the causal policy trails logged behavior in `late_day`.
- Keep the logistics framing tied strictly to `eco_mode`; do not broaden the manuscript to routing, dispatching, or fleet-wide control.
