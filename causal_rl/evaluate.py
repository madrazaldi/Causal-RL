from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import (
    ABLATION_COMPARISON_PATH,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SUMMARY_PATH,
    ESTIMATOR_DIAGNOSTICS_PATH,
    INTERPRETATION_SUMMARY_PATH,
    LEARNED_POLICY_REGISTRY,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MIN_PROPENSITY,
    MODELS_DIR,
    POLICY_ACTIONS_PATH,
    POLICY_SUMMARY_PATH,
    Q_GAP_THRESHOLD,
    REWARD_COLUMNS,
    REWARD_SENSITIVITY_PATH,
    REWARD_SPECS,
    RESULTS_DIR,
    ROBUSTNESS_PATH,
    ROBUSTNESS_SEGMENTS,
    SEED,
    SUPPORT_SWEEP_PATH,
    SUPPORT_SWEEP_PROPENSITIES,
    SUPPORT_SWEEP_Q_GAPS,
)
from .config import ACTION_COLUMN, DECISION_LOG_PATH
from .ope import FittedQEvaluator, doubly_robust, self_normalized_ips

CORE_POLICY_NAMES = [
    "logged_behavior",
    "always_eco",
    "never_eco",
    "heuristic_risk_rule",
    "non_causal_fqi",
    "causal_fqi",
]
ABLATION_POLICY_NAMES = [
    "causal_no_history_fqi",
    "causal_no_vehicle_id_fqi",
]
BOOTSTRAP_METRICS = [
    "policy_value_plugin",
    "policy_value_ips",
    "policy_value_dr",
    "estimated_lateness_min",
    "estimated_co2_kg",
    "estimated_crash",
    "estimated_near_miss",
    "estimated_on_time",
]
SIMPLE_POLICY_NAMES = {"logged_behavior", "always_eco", "never_eco", "heuristic_risk_rule"}


def stable_seed(*parts: object) -> int:
    value = SEED
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = (value * 131 + byte) % (2**32)
    return value


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(DECISION_LOG_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if POLICY_SUMMARY_PATH.exists():
        with open(POLICY_SUMMARY_PATH, "r", encoding="utf-8") as fh:
            policy_summary = json.load(fh)
        metadata.setdefault("trained_state_columns", {})
        metadata.setdefault("ablation_definitions", {})
        for policy_name, spec in policy_summary.get("policies", {}).items():
            if "state_columns" in spec:
                metadata["trained_state_columns"][policy_name] = spec["state_columns"]
            if "ablation" in spec:
                metadata["ablation_definitions"][policy_name] = spec["ablation"]
    models = {
        "behavior": joblib.load(MODELS_DIR / "behavior_model.joblib"),
        "heuristic": joblib.load(MODELS_DIR / "heuristic_model.joblib"),
        "reward_outcome": joblib.load(MODELS_DIR / "reward_outcome_model.joblib"),
        "outcome_targets": joblib.load(MODELS_DIR / "outcome_targets.joblib"),
    }
    for policy_name in LEARNED_POLICY_REGISTRY:
        model_path = MODELS_DIR / f"{policy_name}.joblib"
        if model_path.exists():
            models[policy_name] = joblib.load(model_path)
    return df, metadata, models


def heuristic_actions(model, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    late_prob = model.predict_proba(df[columns])[:, 1]
    threshold = np.quantile(late_prob, 0.60)
    return (late_prob < threshold).astype(int)


def predict_reward_components(
    outcome_models: dict,
    df: pd.DataFrame,
    state_columns: list[str],
    policy_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    features = df[state_columns].copy()
    features[ACTION_COLUMN] = policy_actions
    return {target: model.predict(features) for target, model in outcome_models.items()}


def reward_from_components(component_predictions: dict[str, np.ndarray], reward_name: str) -> np.ndarray:
    weights = REWARD_SPECS[reward_name]
    reward = np.zeros(len(next(iter(component_predictions.values()))), dtype=float)
    for field, weight in weights.items():
        reward -= weight * component_predictions[field]
    return reward


def prepare_policy_cache(df: pd.DataFrame, metadata: dict, models: dict, policy_output: pd.DataFrame) -> dict:
    causal_cols = metadata["causal_state_columns"]
    action = policy_output["policy_action"].to_numpy(dtype=int)
    logged_actions = df["action"].to_numpy(dtype=int)
    behavior_prop = models["behavior"].predict_proba(df[causal_cols])
    logged_propensity = behavior_prop[np.arange(len(df)), logged_actions]

    policy_components = predict_reward_components(models["outcome_targets"], df, causal_cols, action)
    logged_components = predict_reward_components(models["outcome_targets"], df, causal_cols, logged_actions)

    return {
        "action": action,
        "logged_actions": logged_actions,
        "logged_propensity": logged_propensity,
        "matched": action == logged_actions,
        "used_fallback": policy_output.get("used_fallback", pd.Series(np.zeros(len(df), dtype=int))).to_numpy(dtype=float),
        "low_support": policy_output.get("low_support", pd.Series(np.zeros(len(df), dtype=int))).to_numpy(dtype=float),
        "policy_components": policy_components,
        "logged_components": logged_components,
        "policy_rewards": {
            reward_name: reward_from_components(policy_components, reward_name) for reward_name in REWARD_COLUMNS
        },
        "logged_rewards": {
            reward_name: reward_from_components(logged_components, reward_name) for reward_name in REWARD_COLUMNS
        },
    }


def subset_cache(cache: dict, mask: np.ndarray | pd.Series | None) -> dict:
    if mask is None:
        return cache
    mask_array = np.asarray(mask, dtype=bool)
    return {
        "action": cache["action"][mask_array],
        "logged_actions": cache["logged_actions"][mask_array],
        "logged_propensity": cache["logged_propensity"][mask_array],
        "matched": cache["matched"][mask_array],
        "used_fallback": cache["used_fallback"][mask_array],
        "low_support": cache["low_support"][mask_array],
        "policy_components": {
            key: values[mask_array] for key, values in cache["policy_components"].items()
        },
        "logged_components": {
            key: values[mask_array] for key, values in cache["logged_components"].items()
        },
        "policy_rewards": {
            key: values[mask_array] for key, values in cache["policy_rewards"].items()
        },
        "logged_rewards": {
            key: values[mask_array] for key, values in cache["logged_rewards"].items()
        },
    }


def summarize_point_metrics(cache: dict, reward_values: np.ndarray, reward_name: str) -> dict[str, float]:
    reward_values = np.asarray(reward_values, dtype=float)
    matched = cache["matched"]
    point_metrics = {
        "direct_match_reward": float(reward_values[matched].mean()) if matched.any() else np.nan,
        "match_rate": float(matched.mean()),
        "override_rate": float((~matched).mean()),
        "policy_value_plugin": float(np.mean(cache["policy_rewards"][reward_name])),
        "policy_value_ips": self_normalized_ips(
            reward_values,
            cache["logged_actions"],
            cache["action"],
            cache["logged_propensity"],
        ),
        "policy_value_dr": doubly_robust(
            reward_values,
            cache["logged_actions"],
            cache["action"],
            cache["logged_propensity"],
            cache["logged_rewards"][reward_name],
            cache["policy_rewards"][reward_name],
        ),
        "estimated_lateness_min": float(np.mean(cache["policy_components"]["lateness_min"])),
        "estimated_co2_kg": float(np.mean(cache["policy_components"]["co2_kg"])),
        "estimated_crash": float(np.mean(cache["policy_components"]["crash"])),
        "estimated_near_miss": float(np.mean(cache["policy_components"]["near_miss"])),
        "estimated_on_time": float(np.mean(cache["policy_components"]["on_time"])),
        "eco_rate": float(np.mean(cache["action"])),
        "fallback_rate": float(np.mean(cache["used_fallback"])),
        "low_support_rate": float(np.mean(cache["low_support"])),
    }
    return point_metrics


def compute_policy_metrics(
    cache: dict,
    reward_values: np.ndarray,
    reward_name: str,
    policy_name: str,
    mask: np.ndarray | pd.Series | None = None,
    policy_value_fqe: float | None = None,
) -> dict:
    masked_cache = subset_cache(cache, mask)
    masked_rewards = np.asarray(reward_values, dtype=float)
    if mask is not None:
        masked_rewards = masked_rewards[np.asarray(mask, dtype=bool)]
    metrics = summarize_point_metrics(masked_cache, masked_rewards, reward_name)
    metrics.update(
        {
            "policy": policy_name,
            "reward_name": reward_name,
            "policy_value_fqe": float(policy_value_fqe) if policy_value_fqe is not None else np.nan,
        }
    )
    return metrics


def bootstrap_confidence_intervals(
    cache: dict,
    reward_values: np.ndarray,
    reward_name: str,
    policy_name: str,
    bootstrap_reps: int,
    scope: str,
    segment: str | None = None,
    mask: np.ndarray | pd.Series | None = None,
) -> list[dict]:
    if bootstrap_reps <= 0:
        return []

    masked_cache = subset_cache(cache, mask)
    masked_rewards = np.asarray(reward_values, dtype=float)
    if mask is not None:
        masked_rewards = masked_rewards[np.asarray(mask, dtype=bool)]
    n_rows = len(masked_rewards)
    if n_rows == 0:
        return []

    point_metrics = summarize_point_metrics(masked_cache, masked_rewards, reward_name)
    distributions = {metric: np.empty(bootstrap_reps, dtype=float) for metric in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(stable_seed(policy_name, reward_name, scope, segment or "overall"))

    for rep in range(bootstrap_reps):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        sample_cache = {
            "action": masked_cache["action"][sample_idx],
            "logged_actions": masked_cache["logged_actions"][sample_idx],
            "logged_propensity": masked_cache["logged_propensity"][sample_idx],
            "matched": masked_cache["matched"][sample_idx],
            "used_fallback": masked_cache["used_fallback"][sample_idx],
            "low_support": masked_cache["low_support"][sample_idx],
            "policy_components": {
                key: values[sample_idx] for key, values in masked_cache["policy_components"].items()
            },
            "logged_components": {
                key: values[sample_idx] for key, values in masked_cache["logged_components"].items()
            },
            "policy_rewards": {
                key: values[sample_idx] for key, values in masked_cache["policy_rewards"].items()
            },
            "logged_rewards": {
                key: values[sample_idx] for key, values in masked_cache["logged_rewards"].items()
            },
        }
        sample_metrics = summarize_point_metrics(sample_cache, masked_rewards[sample_idx], reward_name)
        for metric in BOOTSTRAP_METRICS:
            distributions[metric][rep] = sample_metrics[metric]

    rows = []
    for metric in BOOTSTRAP_METRICS:
        ci_low, ci_high = np.quantile(distributions[metric], [0.025, 0.975])
        rows.append(
            {
                "scope": scope,
                "segment": segment or "",
                "policy": policy_name,
                "reward_name": reward_name,
                "metric": metric,
                "point_estimate": point_metrics[metric],
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_reps": bootstrap_reps,
            }
        )
    return rows


def get_policy_outputs(
    df: pd.DataFrame,
    metadata: dict,
    models: dict,
    run_ablations: bool,
    min_propensity: float = MIN_PROPENSITY,
    q_gap_threshold: float = Q_GAP_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    causal_cols = metadata["causal_state_columns"]
    learned_policy_names = ["non_causal_fqi", "causal_fqi"]
    if run_ablations:
        learned_policy_names.extend(ABLATION_POLICY_NAMES)

    outputs = {
        "logged_behavior": pd.DataFrame({"policy_action": df["action"].astype(int)}),
        "always_eco": pd.DataFrame({"policy_action": np.ones(len(df), dtype=int)}),
        "never_eco": pd.DataFrame({"policy_action": np.zeros(len(df), dtype=int)}),
        "heuristic_risk_rule": pd.DataFrame(
            {"policy_action": heuristic_actions(models["heuristic"], df, causal_cols)}
        ),
    }
    for policy_name in learned_policy_names:
        outputs[policy_name] = models[policy_name].policy_action(
            df,
            models["behavior"],
            causal_cols,
            min_propensity=min_propensity,
            q_gap_threshold=q_gap_threshold,
        )
    return outputs


def subset_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    if "quantile" in rule:
        if "traffic_index.quantile(0.75)" in rule:
            threshold = df["traffic_index"].quantile(0.75)
            return df["traffic_index"] >= threshold
        if "time_window_tightness.quantile(0.75)" in rule:
            threshold = df["time_window_tightness"].quantile(0.75)
            return df["time_window_tightness"] >= threshold
    if "rain == 1" in rule:
        return (df["rain"] == 1) | (df["event_indicator"] == 1)
    if "hour >= 17" in rule:
        return df["hour"] >= 17
    raise ValueError(f"Unsupported robustness rule: {rule}")


def policy_callable(policy_name: str, models: dict, metadata: dict):
    causal_cols = metadata["causal_state_columns"]

    def _callable(next_state_df: pd.DataFrame) -> np.ndarray:
        if policy_name == "logged_behavior":
            if "action" in next_state_df.columns:
                return next_state_df["action"].to_numpy(dtype=int)
            return np.zeros(len(next_state_df), dtype=int)
        if policy_name == "always_eco":
            return np.ones(len(next_state_df), dtype=int)
        if policy_name == "never_eco":
            return np.zeros(len(next_state_df), dtype=int)
        if policy_name == "heuristic_risk_rule":
            return heuristic_actions(models["heuristic"], next_state_df, causal_cols)
        if policy_name in LEARNED_POLICY_REGISTRY:
            policy_df = next_state_df.copy()
            for column in models[policy_name].state_columns:
                if column not in policy_df.columns:
                    policy_df[column] = 0
            return models[policy_name].policy_action(
                policy_df,
                models["behavior"],
                causal_cols,
                fallback_to_logged=False,
                min_propensity=MIN_PROPENSITY,
                q_gap_threshold=Q_GAP_THRESHOLD,
            )["policy_action"].to_numpy(dtype=int)
        raise ValueError(f"Unknown policy: {policy_name}")

    return _callable


def build_support_sweep_df(
    test_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    run_ablations: bool,
) -> pd.DataFrame:
    learned_policy_names = ["non_causal_fqi", "causal_fqi"]
    if run_ablations:
        learned_policy_names.extend(ABLATION_POLICY_NAMES)

    reward_values = test_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float)
    rows = []
    for policy_name in learned_policy_names:
        for min_propensity in SUPPORT_SWEEP_PROPENSITIES:
            for q_gap_threshold in SUPPORT_SWEEP_Q_GAPS:
                policy_output = models[policy_name].policy_action(
                    test_df,
                    models["behavior"],
                    metadata["causal_state_columns"],
                    min_propensity=min_propensity,
                    q_gap_threshold=q_gap_threshold,
                )
                cache = prepare_policy_cache(test_df, metadata, models, policy_output)
                summary = summarize_point_metrics(cache, reward_values, "primary")
                rows.append(
                    {
                        "policy": policy_name,
                        "min_propensity": min_propensity,
                        "q_gap_threshold": q_gap_threshold,
                        "policy_value_dr": summary["policy_value_dr"],
                        "policy_value_ips": summary["policy_value_ips"],
                        "policy_value_plugin": summary["policy_value_plugin"],
                        "override_rate": summary["override_rate"],
                        "fallback_rate": summary["fallback_rate"],
                        "low_support_rate": summary["low_support_rate"],
                        "match_rate": summary["match_rate"],
                        "eco_rate": summary["eco_rate"],
                        "is_default": int(
                            np.isclose(min_propensity, MIN_PROPENSITY)
                            and np.isclose(q_gap_threshold, Q_GAP_THRESHOLD)
                        ),
                    }
                )
    sweep_df = pd.DataFrame(rows)
    if sweep_df.empty:
        return sweep_df

    pair_summary = (
        sweep_df.groupby(["min_propensity", "q_gap_threshold"], as_index=False)
        .agg(
            mean_policy_value_dr=("policy_value_dr", "mean"),
            mean_override_rate=("override_rate", "mean"),
            mean_fallback_rate=("fallback_rate", "mean"),
            mean_low_support_rate=("low_support_rate", "mean"),
        )
    )
    pair_summary["conservative_score"] = (
        pair_summary["mean_policy_value_dr"]
        - 0.5 * pair_summary["mean_override_rate"]
        - 0.25 * pair_summary["mean_low_support_rate"]
    )
    best_pair = pair_summary.sort_values(
        ["conservative_score", "mean_override_rate", "mean_fallback_rate"],
        ascending=[False, True, True],
    ).iloc[0]
    sweep_df["selected_for_main_paper"] = (
        np.isclose(sweep_df["min_propensity"], best_pair["min_propensity"])
        & np.isclose(sweep_df["q_gap_threshold"], best_pair["q_gap_threshold"])
    ).astype(int)
    return sweep_df


def add_bootstrap_columns(df: pd.DataFrame, bootstrap_df: pd.DataFrame, index_columns: list[str]) -> pd.DataFrame:
    if bootstrap_df.empty or df.empty:
        return df
    relevant = bootstrap_df.copy()
    wide = (
        relevant.pivot_table(index=index_columns, columns="metric", values=["ci_low", "ci_high"], aggfunc="first")
        .sort_index(axis=1)
    )
    wide.columns = [f"{metric}_{bound}" for bound, metric in wide.columns]
    wide = wide.reset_index()
    return df.merge(wide, on=index_columns, how="left")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def build_ablation_comparison(metrics_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    ablation_policies = [
        "causal_fqi",
        "causal_no_history_fqi",
        "causal_no_vehicle_id_fqi",
        "non_causal_fqi",
    ]
    available = metrics_df[metrics_df["policy"].isin(ablation_policies)].copy()
    if available.empty:
        return available

    causal_row = available.loc[available["policy"] == "causal_fqi"].iloc[0]
    notes = {
        "causal_fqi": "Reference confounder-aware policy with the full backdoor-guided state.",
        "non_causal_fqi": "Uses broader deployable proxies, including risk and compatibility features, as the non-causal comparator.",
    }
    for policy_name, spec in metadata.get("ablation_definitions", {}).items():
        notes[policy_name] = spec["note"]

    available["delta_vs_causal_fqi"] = available["policy_value_dr"] - causal_row["policy_value_dr"]
    available["ablation_note"] = available["policy"].map(notes).fillna("")
    available["removed_columns"] = available["policy"].map(
        {
            name: ",".join(spec["removed_columns"])
            for name, spec in metadata.get("ablation_definitions", {}).items()
        }
    ).fillna("")
    return available[
        [
            "policy",
            "policy_value_dr",
            "policy_value_dr_ci_low",
            "policy_value_dr_ci_high",
            "fallback_rate",
            "override_rate",
            "delta_vs_causal_fqi",
            "removed_columns",
            "ablation_note",
        ]
    ].sort_values("policy_value_dr", ascending=False)


def build_estimator_diagnostics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = metrics_df.copy()
    diagnostics["primary_estimator"] = "policy_value_dr"
    diagnostics["secondary_estimator"] = "policy_value_ips"
    diagnostics["plugin_gap_vs_dr"] = diagnostics["policy_value_plugin"] - diagnostics["policy_value_dr"]
    diagnostics["ips_gap_vs_dr"] = diagnostics["policy_value_ips"] - diagnostics["policy_value_dr"]
    diagnostics["fqe_gap_vs_dr"] = diagnostics["policy_value_fqe"] - diagnostics["policy_value_dr"]
    return diagnostics[
        [
            "policy",
            "primary_estimator",
            "secondary_estimator",
            "policy_value_dr",
            "policy_value_dr_ci_low",
            "policy_value_dr_ci_high",
            "policy_value_ips",
            "policy_value_plugin",
            "policy_value_fqe",
            "ips_gap_vs_dr",
            "plugin_gap_vs_dr",
            "fqe_gap_vs_dr",
        ]
    ].sort_values("policy_value_dr", ascending=False)


def build_interpretation_summary(metrics_df: pd.DataFrame, robustness_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_scope_rows(scope_df: pd.DataFrame, scope: str, segment: str) -> None:
        if scope_df.empty:
            return
        best_row = scope_df.sort_values("policy_value_dr", ascending=False).iloc[0]
        logged_row = scope_df.loc[scope_df["policy"] == "logged_behavior"].iloc[0]
        causal_row = scope_df.loc[scope_df["policy"] == "causal_fqi"].iloc[0]
        simpler_policy_beats_causal = int(best_row["policy"] in SIMPLE_POLICY_NAMES and best_row["policy"] != "causal_fqi")
        if simpler_policy_beats_causal:
            insight = (
                f"{best_row['policy']} outperforms causal_fqi in {segment}, suggesting a simpler operating rule is more stable in this slice."
            )
        else:
            insight = (
                f"{best_row['policy']} delivers the strongest doubly robust value in {segment}, supporting the current deployment framing."
            )
        rows.append(
            {
                "scope": scope,
                "segment": segment,
                "best_policy": best_row["policy"],
                "best_policy_value_dr": best_row["policy_value_dr"],
                "delta_vs_logged_behavior": best_row["policy_value_dr"] - logged_row["policy_value_dr"],
                "delta_vs_causal_fqi": best_row["policy_value_dr"] - causal_row["policy_value_dr"],
                "simpler_policy_beats_causal": simpler_policy_beats_causal,
                "insight": insight,
            }
        )

    add_scope_rows(metrics_df, "overall", "overall")
    for segment, segment_df in robustness_df.groupby("segment"):
        add_scope_rows(segment_df, "segment", segment)
    return pd.DataFrame(rows)


def evaluate_all(
    bootstrap_reps: int = BOOTSTRAP_REPS,
    run_ablations: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata, models = load_inputs()
    train_val_df = df[df["split"].isin(["train", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()
    policy_outputs = get_policy_outputs(test_df, metadata, models, run_ablations=run_ablations)

    actions_frame = test_df[["trajectory_id", "t", "split", "action", "reward"]].copy()
    metric_rows = []
    robustness_rows = []
    reward_rows = []
    bootstrap_rows = []

    for policy_name, policy_output in policy_outputs.items():
        actions_frame[f"{policy_name}_action"] = policy_output["policy_action"].to_numpy(dtype=int)
        primary_reward = test_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float)
        policy_action_values = policy_output["policy_action"].to_numpy(dtype=int)
        fqe_value = FittedQEvaluator(metadata["causal_state_columns"]).fit(
            train_val_df.copy(), policy_callable(policy_name, models, metadata), reward_column="reward"
        ).evaluate_policy_value(test_df, policy_action_values)

        cache = prepare_policy_cache(test_df, metadata, models, policy_output)
        metric_rows.append(
            compute_policy_metrics(
                cache,
                primary_reward,
                "primary",
                policy_name,
                policy_value_fqe=fqe_value,
            )
        )
        bootstrap_rows.extend(
            bootstrap_confidence_intervals(
                cache,
                primary_reward,
                "primary",
                policy_name,
                bootstrap_reps=bootstrap_reps,
                scope="overall",
            )
        )

        for segment_name, rule in ROBUSTNESS_SEGMENTS.items():
            mask = subset_mask(test_df, rule)
            if not mask.any():
                continue
            robustness_rows.append(
                {
                    **compute_policy_metrics(
                        cache,
                        primary_reward,
                        "primary",
                        policy_name,
                        mask=mask.to_numpy(),
                        policy_value_fqe=fqe_value,
                    ),
                    "segment": segment_name,
                }
            )
            bootstrap_rows.extend(
                bootstrap_confidence_intervals(
                    cache,
                    primary_reward,
                    "primary",
                    policy_name,
                    bootstrap_reps=bootstrap_reps,
                    scope="segment",
                    segment=segment_name,
                    mask=mask.to_numpy(),
                )
            )

        for reward_name, reward_column in REWARD_COLUMNS.items():
            reward_rows.append(
                compute_policy_metrics(
                    cache,
                    test_df[reward_column].to_numpy(dtype=float),
                    reward_name,
                    policy_name,
                )
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("policy_value_dr", ascending=False).reset_index(drop=True)
    reward_df = pd.DataFrame(reward_rows).sort_values(["reward_name", "policy_value_dr"], ascending=[True, False])
    robustness_df = pd.DataFrame(robustness_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    if not robustness_df.empty:
        logged_by_segment = robustness_df.loc[robustness_df["policy"] == "logged_behavior", ["segment", "policy_value_dr"]]
        logged_by_segment = logged_by_segment.rename(columns={"policy_value_dr": "logged_behavior_dr"})
        robustness_df = robustness_df.merge(logged_by_segment, on="segment", how="left")
        robustness_df["delta_vs_logged_behavior"] = robustness_df["policy_value_dr"] - robustness_df["logged_behavior_dr"]
        robustness_df["segment_rank"] = robustness_df.groupby("segment")["policy_value_dr"].rank(
            ascending=False, method="dense"
        )
        robustness_df = robustness_df.drop(columns=["logged_behavior_dr"]).sort_values(
            ["segment", "policy_value_dr"], ascending=[True, False]
        )

    overall_bootstrap_df = bootstrap_df[bootstrap_df["scope"] == "overall"].copy()
    segment_bootstrap_df = bootstrap_df[bootstrap_df["scope"] == "segment"].copy()
    metrics_df = add_bootstrap_columns(metrics_df, overall_bootstrap_df, ["policy", "reward_name"])
    robustness_df = add_bootstrap_columns(robustness_df, segment_bootstrap_df, ["policy", "reward_name", "segment"])
    metrics_df = ensure_columns(
        metrics_df,
        [
            "policy_value_dr_ci_low",
            "policy_value_dr_ci_high",
            "estimated_lateness_min_ci_low",
            "estimated_lateness_min_ci_high",
            "estimated_co2_kg_ci_low",
            "estimated_co2_kg_ci_high",
            "estimated_on_time_ci_low",
            "estimated_on_time_ci_high",
        ],
    )
    robustness_df = ensure_columns(
        robustness_df,
        [
            "policy_value_dr_ci_low",
            "policy_value_dr_ci_high",
            "estimated_lateness_min_ci_low",
            "estimated_lateness_min_ci_high",
            "estimated_co2_kg_ci_low",
            "estimated_co2_kg_ci_high",
            "estimated_on_time_ci_low",
            "estimated_on_time_ci_high",
        ],
    )

    support_sweep_df = build_support_sweep_df(test_df, metadata, models, run_ablations=run_ablations)
    ablation_df = build_ablation_comparison(metrics_df, metadata)
    diagnostics_df = build_estimator_diagnostics(metrics_df)
    interpretation_df = build_interpretation_summary(metrics_df, robustness_df)

    actions_frame.to_csv(POLICY_ACTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    robustness_df.to_csv(ROBUSTNESS_PATH, index=False)
    reward_df.to_csv(REWARD_SENSITIVITY_PATH, index=False)
    bootstrap_df.to_csv(BOOTSTRAP_SUMMARY_PATH, index=False)
    diagnostics_df.to_csv(ESTIMATOR_DIAGNOSTICS_PATH, index=False)
    support_sweep_df.to_csv(SUPPORT_SWEEP_PATH, index=False)
    ablation_df.to_csv(ABLATION_COMPARISON_PATH, index=False)
    interpretation_df.to_csv(INTERPRETATION_SUMMARY_PATH, index=False)
    metrics_df[
        [
            "policy",
            "policy_value_dr",
            "policy_value_dr_ci_low",
            "policy_value_dr_ci_high",
            "policy_value_ips",
            "policy_value_plugin",
            "estimated_lateness_min",
            "estimated_lateness_min_ci_low",
            "estimated_lateness_min_ci_high",
            "estimated_co2_kg",
            "estimated_co2_kg_ci_low",
            "estimated_co2_kg_ci_high",
            "estimated_on_time",
            "estimated_on_time_ci_low",
            "estimated_on_time_ci_high",
            "eco_rate",
            "override_rate",
            "fallback_rate",
        ]
    ].to_csv(MAIN_RESULTS_TABLE_PATH, index=False)
    return metrics_df, robustness_df, reward_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline RL policies with bootstrap and ablation outputs.")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--run-ablations", dest="run_ablations", action="store_true", default=True)
    parser.add_argument("--skip-ablations", dest="run_ablations", action="store_false")
    args = parser.parse_args()

    metrics_df, robustness_df, reward_df = evaluate_all(
        bootstrap_reps=args.bootstrap_reps,
        run_ablations=args.run_ablations,
    )
    print(f"Saved metrics to {METRICS_PATH} with {len(metrics_df)} policies")
    print(f"Saved robustness results to {ROBUSTNESS_PATH} with {len(robustness_df)} rows")
    print(f"Saved reward sensitivity results to {REWARD_SENSITIVITY_PATH} with {len(reward_df)} rows")
    print(f"Saved bootstrap intervals to {BOOTSTRAP_SUMMARY_PATH}")
    print(f"Saved support sweep to {SUPPORT_SWEEP_PATH}")
    print(f"Saved ablation comparison to {ABLATION_COMPARISON_PATH}")
    print(f"Saved interpretation summary to {INTERPRETATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
