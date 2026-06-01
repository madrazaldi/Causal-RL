from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from .config import (
    ACTION_IRRELEVANT_COUNTEREXAMPLE_PATH,
    CLAIM_BOUNDARY_PATH,
    DOMINANCE_AUDIT_PATH,
    GROUND_TRUTH_BENCHMARK_PATH,
    RESULTS_DIR,
)


def build_action_irrelevant_counterexample(
    policy_names: Sequence[str] = (
        "causal_fqi",
        "omitted_confounder_fqi",
        "heuristic_risk_rule",
    ),
    *,
    horizon: int = 5,
    gamma: float = 0.95,
    reward_per_step: float = -1.0,
) -> pd.DataFrame:
    """Construct the minimal counterexample to strict universal dominance.

    In this one-state MDP, both actions have the same reward and transition.
    The value of every policy is therefore identical, regardless of whether
    the policy is causal, non-causal, RL-based, heuristic, or static.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if not policy_names:
        raise ValueError("policy_names must be non-empty")

    common_value = sum((gamma**step) * reward_per_step for step in range(horizon))
    rows = [
        {
            "policy": policy_name,
            "true_discounted_return": common_value,
            "action_reward_gap": 0.0,
            "transition_gap": 0.0,
            "strictly_dominates_all": 0,
        }
        for policy_name in policy_names
    ]
    return pd.DataFrame(rows)


def build_claim_boundary_audit(
    dominance_path=DOMINANCE_AUDIT_PATH,
    ground_truth_path=GROUND_TRUTH_BENCHMARK_PATH,
    counterexample_path=ACTION_IRRELEVANT_COUNTEREXAMPLE_PATH,
) -> pd.DataFrame:
    dominance = pd.read_csv(dominance_path)
    ground_truth = pd.read_csv(ground_truth_path)
    action_irrelevant = build_action_irrelevant_counterexample()

    counterexamples = int(dominance["counterexample_count"].iloc[0])
    audited_surfaces = int(len(dominance))
    universal_family_win = int(dominance["universal_causal_rl_family_win"].iloc[0])
    action_irrelevant_gap = (
        float(action_irrelevant["true_discounted_return"].max())
        - float(action_irrelevant["true_discounted_return"].min())
    )

    top_ground_truth = ground_truth.sort_values("rank").iloc[0]
    causal_ground_truth = ground_truth.loc[ground_truth["policy"] == "causal_fqi"].iloc[0]
    omitted_ground_truth = ground_truth.loc[ground_truth["policy"] == "omitted_confounder_fqi"].iloc[0]
    controlled_gap = float(
        causal_ground_truth["true_discounted_return"]
        - omitted_ground_truth["true_discounted_return"]
    )

    rows = [
        {
            "claim_id": "observational_benchmark_universal_causal_rl_win",
            "claim_text": "The confounder-aware causal-RL family wins on every audited surface in the main observational benchmark.",
            "status": "falsified",
            "evidence": (
                f"{counterexamples} counterexamples across {audited_surfaces} estimator, reward, and segment surfaces; "
                f"universal_causal_rl_family_win={universal_family_win}."
            ),
            "artifact": str(dominance_path),
            "methodological_consequence": (
                "Do not claim universal dominance from the observational benchmark; report objective-dependent trade-offs."
            ),
        },
        {
            "claim_id": "controlled_dgp_causal_fqi_advantage",
            "claim_text": "Causal FQI can win when the DGP has observed confounding, delayed action effects, adequate support, and true rollout evaluation.",
            "status": "supported_under_constructed_assumptions",
            "evidence": (
                f"{top_ground_truth['policy']} ranks first; causal_fqi beats omitted_confounder_fqi by "
                f"{controlled_gap:.3f} true discounted-return units."
            ),
            "artifact": str(ground_truth_path),
            "methodological_consequence": (
                "Use as a positive control showing the implementation can recover the intended causal-RL advantage."
            ),
        },
        {
            "claim_id": "strict_universal_dominance_over_all_mdps",
            "claim_text": "Causality plus RL strictly wins universally over all possible decision processes and comparators.",
            "status": "impossible_by_counterexample",
            "evidence": (
                "Consider an action-irrelevant MDP with R(s,0)=R(s,1) and P(s'|s,0)=P(s'|s,1) for all states. "
                f"Every policy has the same value (computed value gap={action_irrelevant_gap:.3f}), "
                "so no method can strictly dominate universally."
            ),
            "artifact": str(counterexample_path),
            "methodological_consequence": (
                "Any valid theorem must be conditional on assumptions such as action effects, overlap, correct adjustment, "
                "consistent estimation, and a positive value gap."
            ),
        },
        {
            "claim_id": "bounded_causal_rl_theorem_template",
            "claim_text": "Under sufficient adjustment, overlap, consistent FQI estimation, sequential action effects, and a positive value gap, causal FQI can outperform misspecified comparators asymptotically.",
            "status": "plausible_but_assumption_bound",
            "evidence": (
                "The controlled DGP is a constructive sanity check; the main observational benchmark does not verify all theorem assumptions."
            ),
            "artifact": f"{ground_truth_path}; {dominance_path}",
            "methodological_consequence": (
                "Frame the paper around assumption-bound causal decision support, not unqualified universal dominance."
            ),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the claim-boundary audit for causal-RL dominance claims.")
    parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counterexample = build_action_irrelevant_counterexample()
    counterexample.to_csv(ACTION_IRRELEVANT_COUNTEREXAMPLE_PATH, index=False)
    audit = build_claim_boundary_audit()
    audit.to_csv(CLAIM_BOUNDARY_PATH, index=False)
    print(f"Saved action-irrelevant counterexample to {ACTION_IRRELEVANT_COUNTEREXAMPLE_PATH}")
    print(f"Saved claim-boundary audit to {CLAIM_BOUNDARY_PATH}")
    print(audit[["claim_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
