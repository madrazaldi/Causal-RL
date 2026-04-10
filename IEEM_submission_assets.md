# IEEM Submission Asset Map

This note maps the repo outputs to the current manuscript structure in [outline.md](/Users/anonymoize/Projects/Causal RL/outline.md).

## Recommended Main-Paper Assets

- Section 3 (`Problem Formulation and Proposed Method`)
  - [figures/causal_graph.png](/Users/anonymoize/Projects/Causal RL/figures/causal_graph.png): domain DAG illustrating the backdoor-guided state design.
  - [figures/workflow.png](/Users/anonymoize/Projects/Causal RL/figures/workflow.png): end-to-end offline RL workflow.

- Section 4 (`Experimental Setup`)
  - [artifacts/dataset_metadata.json](/Users/anonymoize/Projects/Causal RL/artifacts/dataset_metadata.json): split dates, state definitions, exclusions, and seed.
  - [README.md](/Users/anonymoize/Projects/Causal RL/README.md): reproducibility commands and package versions.

- Section 5 (`Results`)
  - [results/main_results_table.csv](/Users/anonymoize/Projects/Causal RL/results/main_results_table.csv): main policy comparison table with doubly robust estimates and bootstrap intervals.
  - [figures/policy_values.png](/Users/anonymoize/Projects/Causal RL/figures/policy_values.png): doubly robust policy values with uncertainty bars.
  - [results/robustness.csv](/Users/anonymoize/Projects/Causal RL/results/robustness.csv): subgroup-level values, delta vs logged behavior, and within-segment ranking.
  - [figures/robustness_heatmap.png](/Users/anonymoize/Projects/Causal RL/figures/robustness_heatmap.png): scenario robustness heatmap.
  - [results/reward_sensitivity.csv](/Users/anonymoize/Projects/Causal RL/results/reward_sensitivity.csv): reward-weight sensitivity comparison.
  - [figures/reward_sensitivity.png](/Users/anonymoize/Projects/Causal RL/figures/reward_sensitivity.png): reward sensitivity figure.

## Recommended Appendix / Diagnostic Assets

- [results/bootstrap_summary.csv](/Users/anonymoize/Projects/Causal RL/results/bootstrap_summary.csv): long-form confidence interval output for overall and subgroup metrics.
- [results/estimator_diagnostics.csv](/Users/anonymoize/Projects/Causal RL/results/estimator_diagnostics.csv): DR, IPS, plugin, and FQE comparison; use this to justify DR as the primary estimator.
- [results/ablation_comparison.csv](/Users/anonymoize/Projects/Causal RL/results/ablation_comparison.csv): causal ablation table for history and vehicle-identity removal.
- [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv): support-constraint sweep used to justify the default fallback thresholds.
- [results/interpretation_summary.csv](/Users/anonymoize/Projects/Causal RL/results/interpretation_summary.csv): concise narrative cues for segments where simpler policies outperform the causal learner.
- [results/metrics.csv](/Users/anonymoize/Projects/Causal RL/results/metrics.csv): full policy metrics with diagnostic columns.
- [results/policy_actions.csv](/Users/anonymoize/Projects/Causal RL/results/policy_actions.csv): step-level actions for reproducibility and case study inspection.

## Recommended Framing

- Present the repo as a confounder-aware offline RL benchmark and analysis pipeline, not as proof that causal FQI dominates every comparator.
- Use doubly robust estimates as the main table values.
- Treat IPS as a consistency check and FQE as an appendix diagnostic unless its scale is separately reconciled.
- Document the selected conservative support defaults as `min_propensity=0.05` and `q_gap_threshold=0.50`, justified by [results/support_threshold_sweep.csv](/Users/anonymoize/Projects/Causal RL/results/support_threshold_sweep.csv).
- Highlight subgroup cases where simpler policies win as operational findings about deployment conservatism, not as invalidation of the causal framing.
