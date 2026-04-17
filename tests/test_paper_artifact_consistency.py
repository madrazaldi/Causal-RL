from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_rl.config import (
    CLUSTER_BOOTSTRAP_PATH,
    COMMON_SUPPORT_PATH,
    LATENT_COLUMNS,
    MAIN_RESULTS_TABLE_PATH,
    ROOT,
    BOOTSTRAP_SUMMARY_PATH,
)


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _load_main_results() -> pd.DataFrame:
    return pd.read_csv(MAIN_RESULTS_TABLE_PATH).set_index("policy")


def _compute_cluster_ci_wider_pct() -> float:
    row_boot = pd.read_csv(BOOTSTRAP_SUMMARY_PATH).fillna({"segment": ""})
    cluster_boot = pd.read_csv(CLUSTER_BOOTSTRAP_PATH).fillna({"segment": ""})
    for df in (row_boot, cluster_boot):
        df["width"] = df["ci_high"] - df["ci_low"]

    merged = row_boot.merge(
        cluster_boot,
        on=["scope", "segment", "policy", "reward_name", "metric"],
        suffixes=("_row", "_cluster"),
    )
    subset = merged[
        (merged["scope"] == "overall")
        & (merged["reward_name"] == "primary")
        & (merged["metric"] == "policy_value_dr")
    ].copy()
    subset["pct_wider"] = (subset["width_cluster"] / subset["width_row"] - 1.0) * 100.0
    return float(subset["pct_wider"].mean())


def test_headline_policy_claims_match_regenerated_results() -> None:
    results = _load_main_results()
    readme = _read_text("README.md")
    outline = _read_text("outline.md")
    ieem = _read_text("IEEM_submission_assets.md")

    non_causal = results.loc["non_causal_fqi"]
    causal = results.loc["causal_fqi"]
    logged = results.loc["logged_behavior"]

    readme_line = (
        f"`non_causal_fqi`: `{non_causal['policy_value_dr']:.3f}` with 95% bootstrap CI "
        f"`[{non_causal['policy_value_dr_ci_low']:.3f}, {non_causal['policy_value_dr_ci_high']:.3f}]`"
    )
    assert readme_line in readme

    outline_line = (
        f"`non_causal_fqi` is the strongest overall policy by doubly robust value at "
        f"`{non_causal['policy_value_dr']:.3f}` with 95% CI "
        f"`[{non_causal['policy_value_dr_ci_low']:.3f}, {non_causal['policy_value_dr_ci_high']:.3f}]`"
    )
    assert outline_line in outline

    ieem_line = (
        f"`non_causal_fqi` is the strongest overall policy in the current benchmark at "
        f"`{non_causal['policy_value_dr']:.3f}` with 95% CI "
        f"`[{non_causal['policy_value_dr_ci_low']:.3f}, {non_causal['policy_value_dr_ci_high']:.3f}]`"
    )
    assert ieem_line in ieem

    causal_vs_logged = (
        f"`causal_fqi` improves over `logged_behavior` "
        f"(`{causal['policy_value_dr']:.3f}` vs `{logged['policy_value_dr']:.3f}`)"
    )
    assert causal_vs_logged in outline
    assert causal_vs_logged in ieem


def test_latent_exclusion_language_matches_config() -> None:
    methodology = _read_text("paper_methodology.md")
    outline = _read_text("outline.md")
    report_source = _read_text("causal_rl/report.py")

    for latent in LATENT_COLUMNS:
        assert latent in methodology
        assert latent in outline
        assert latent in report_source


def test_cluster_bootstrap_claim_matches_artifacts() -> None:
    methodology = _read_text("paper_methodology.md")
    outline = _read_text("outline.md")
    ieem = _read_text("IEEM_submission_assets.md")

    pct_wider = _compute_cluster_ci_wider_pct()
    pct_text = f"{pct_wider:.1f}% wider"

    for text in (methodology, outline, ieem):
        assert "overall primary-reward doubly robust policy-value metric" in text
        assert pct_text in text


def test_common_support_percentage_uses_percent_units() -> None:
    ieem = _read_text("IEEM_submission_assets.md")
    support_df = pd.read_csv(COMMON_SUPPORT_PATH)
    pct_below = float(support_df.loc[support_df["policy"] == "logged_behavior", "pct_below_tau_mu"].iloc[0] * 100.0)
    assert f"{pct_below:.2f}% of test rows have propensity below τ_μ=0.05" in ieem
