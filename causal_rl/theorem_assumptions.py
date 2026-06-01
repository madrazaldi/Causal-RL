from __future__ import annotations

import argparse

import pandas as pd

from .config import (
    CLAIM_BOUNDARY_PATH,
    COMMON_SUPPORT_PATH,
    DOMINANCE_AUDIT_PATH,
    GROUND_TRUTH_BENCHMARK_PATH,
    RESULTS_DIR,
    THEOREM_ASSUMPTION_PATH,
)


def build_theorem_assumption_audit(
    claim_boundary_path=CLAIM_BOUNDARY_PATH,
    dominance_path=DOMINANCE_AUDIT_PATH,
    ground_truth_path=GROUND_TRUTH_BENCHMARK_PATH,
    support_path=COMMON_SUPPORT_PATH,
) -> pd.DataFrame:
    claim_boundary = pd.read_csv(claim_boundary_path)
    dominance = pd.read_csv(dominance_path)
    ground_truth = pd.read_csv(ground_truth_path)
    support = pd.read_csv(support_path)

    support_logged = support.loc[support["policy"] == "logged_behavior"].iloc[0]
    ground_truth_top = ground_truth.sort_values("rank").iloc[0]
    causal_ground_truth = ground_truth.loc[ground_truth["policy"] == "causal_fqi"].iloc[0]
    omitted_ground_truth = ground_truth.loc[ground_truth["policy"] == "omitted_confounder_fqi"].iloc[0]
    controlled_gap = float(
        causal_ground_truth["true_discounted_return"]
        - omitted_ground_truth["true_discounted_return"]
    )
    universal_counterexamples = int(dominance["counterexample_count"].iloc[0])
    theorem_row = claim_boundary.loc[
        claim_boundary["claim_id"] == "bounded_causal_rl_theorem_template"
    ].iloc[0]

    rows = [
        {
            "assumption_id": "A1_action_relevance",
            "assumption": "Actions affect rewards or transitions in a way that creates a nonzero policy value gap.",
            "controlled_dgp_status": "satisfied_by_construction",
            "main_benchmark_status": "partly_supported_not_proven",
            "evidence": (
                "Controlled DGP includes delayed action effects and causal_fqi beats omitted_confounder_fqi by "
                f"{controlled_gap:.3f}; main benchmark has policy value differences but no true interventional rollout."
            ),
            "if_missing": "No strict method can dominate in action-irrelevant decision processes.",
        },
        {
            "assumption_id": "A2_sufficient_adjustment",
            "assumption": "The causal state blocks action-outcome confounding for deployable decisions.",
            "controlled_dgp_status": "satisfied_by_construction",
            "main_benchmark_status": "not_verified",
            "evidence": (
                "Controlled DGP exposes the confounders to causal_fqi; main benchmark excludes latent simulator fields, "
                "so strict backdoor identification is not established."
            ),
            "if_missing": "Omitted confounding can make causal policy estimates biased.",
        },
        {
            "assumption_id": "A3_overlap_support",
            "assumption": "Every evaluated policy action has adequate logged support in the evaluated state region.",
            "controlled_dgp_status": "satisfied_by_simulation",
            "main_benchmark_status": "empirically_supported_for_thresholded_policy",
            "evidence": (
                f"Main benchmark logged propensities have p5={support_logged['propensity_p5']:.3f}; "
                f"{support_logged['pct_below_tau_mu']:.2f}% of test rows are below tau_mu."
            ),
            "if_missing": "Offline evaluation and policy improvement become extrapolative.",
        },
        {
            "assumption_id": "A4_consistent_value_learning",
            "assumption": "FQI and nuisance models consistently estimate the relevant value or reward functions.",
            "controlled_dgp_status": "approximated_by_positive_control",
            "main_benchmark_status": "not_proven",
            "evidence": (
                f"Controlled DGP ranks {ground_truth_top['policy']} first by true rollout; "
                "main benchmark relies on learned OPE and FQE diagnostics rather than ground-truth values."
            ),
            "if_missing": "Ranking can reflect estimator/model error rather than policy quality.",
        },
        {
            "assumption_id": "A5_objective_alignment",
            "assumption": "The training objective and the reported evaluation objective measure the same notion of policy quality.",
            "controlled_dgp_status": "satisfied",
            "main_benchmark_status": "violated_or_mixed",
            "evidence": (
                "Main benchmark trains with a sequential Bellman objective but reports per-decision contextual DR as the headline; "
                f"dominance audit records {universal_counterexamples} counterexamples across audited surfaces."
            ),
            "if_missing": "A policy can win under FQE while losing under per-decision DR or segment metrics.",
        },
        {
            "assumption_id": "A6_positive_comparator_gap",
            "assumption": "The causal-RL policy has a positive value gap over the relevant comparator under the chosen objective.",
            "controlled_dgp_status": "satisfied",
            "main_benchmark_status": "objective_dependent",
            "evidence": (
                f"Controlled DGP causal_fqi gap over omitted-confounder FQI is {controlled_gap:.3f}; "
                "main benchmark gap is positive for FQE and some surfaces but not universal."
            ),
            "if_missing": "At best, methods tie; strict universal win claims fail.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit["bounded_theorem_status"] = str(theorem_row["status"])
    audit["all_assumptions_verified_in_main_benchmark"] = int(
        (audit["main_benchmark_status"] == "empirically_supported_for_thresholded_policy").all()
    )
    audit["all_assumptions_satisfied_in_controlled_dgp"] = int(
        audit["controlled_dgp_status"].isin(
            ["satisfied", "satisfied_by_construction", "satisfied_by_simulation", "approximated_by_positive_control"]
        ).all()
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the theorem-assumption audit for bounded causal-RL claims.")
    parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_theorem_assumption_audit()
    audit.to_csv(THEOREM_ASSUMPTION_PATH, index=False)
    print(f"Saved theorem-assumption audit to {THEOREM_ASSUMPTION_PATH}")
    print(audit[["assumption_id", "controlled_dgp_status", "main_benchmark_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
