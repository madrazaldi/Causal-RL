from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import (
    ABLATION_COMPARISON_PATH,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SUMMARY_PATH,
    CAUSAL_BACKDOOR_COLUMNS,
    CLUSTER_BOOTSTRAP_PATH,
    COMMON_SUPPORT_PATH,
    DOMINANCE_AUDIT_PATH,
    ESTIMATOR_DIAGNOSTICS_PATH,
    FQE_CONVERGENCE_PATH,
    FQE_N_REPEATS,
    FEATURE_IMPORTANCE_PATH,
    HEURISTIC_DIAGNOSTICS_PATH,
    INTERPRETATION_SUMMARY_PATH,
    LEARNED_POLICY_REGISTRY,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MIN_PROPENSITY,
    MODELS_DIR,
    ORACLE_SENSITIVITY_PATH,
    POLICY_ACTIONS_PATH,
    POLICY_DIFFERENCE_BOOTSTRAP_PATH,
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
from .parallel import limit_inner_threads, resolve_n_jobs

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
    "minimal_fqi",
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
CAUSAL_RL_POLICY_NAMES = {
    "causal_fqi",
    "causal_no_history_fqi",
    "causal_no_vehicle_id_fqi",
}
KEY_POLICY_DIFFERENCES = [
    ("causal_fqi", "non_causal_fqi", "causal_vs_non_causal"),
    ("causal_fqi", "logged_behavior", "causal_vs_logged"),
    ("heuristic_risk_rule", "causal_fqi", "heuristic_vs_causal"),
    ("causal_fqi", "minimal_fqi", "causal_vs_minimal"),
    ("causal_no_vehicle_id_fqi", "causal_fqi", "no_vehicle_vs_causal"),
]


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
        if "heuristic_policy" in policy_summary:
            metadata["heuristic_policy"] = policy_summary["heuristic_policy"]
    models = {
        "behavior": joblib.load(MODELS_DIR / "behavior_model.joblib"),
        "heuristic": joblib.load(MODELS_DIR / "heuristic_model.joblib"),
        "outcome_targets": joblib.load(MODELS_DIR / "outcome_targets.joblib"),
    }
    behavior_models_path = MODELS_DIR / "behavior_models.joblib"
    if behavior_models_path.exists():
        models["behavior_models"] = joblib.load(behavior_models_path)
    outcome_targets_by_state_path = MODELS_DIR / "outcome_targets_by_state.joblib"
    if outcome_targets_by_state_path.exists():
        models["outcome_targets_by_state"] = joblib.load(outcome_targets_by_state_path)
    for policy_name in LEARNED_POLICY_REGISTRY:
        model_path = MODELS_DIR / f"{policy_name}.joblib"
        if model_path.exists():
            models[policy_name] = joblib.load(model_path)
    oracle_model_path = MODELS_DIR / "oracle_fqi.joblib"
    if oracle_model_path.exists():
        models["oracle_fqi"] = joblib.load(oracle_model_path)
    return df, metadata, models


def nuisance_group_for_policy(policy_name: str) -> str:
    if policy_name == "oracle_fqi":
        return "oracle"
    return "non_causal" if policy_name == "non_causal_fqi" else "causal"


def nuisance_state_columns(policy_name: str, metadata: dict) -> list[str]:
    if nuisance_group_for_policy(policy_name) == "oracle":
        return metadata.get(
            "oracle_state_columns",
            metadata.get("trained_state_columns", {}).get("oracle_fqi", metadata["causal_state_columns"]),
        )
    if nuisance_group_for_policy(policy_name) == "non_causal":
        return metadata.get("non_causal_state_columns", metadata["causal_state_columns"])
    return metadata["causal_state_columns"]


def behavior_model_for_policy(policy_name: str, models: dict):
    behavior_models = models.get("behavior_models", {})
    return behavior_models.get(nuisance_group_for_policy(policy_name), models["behavior"])


def outcome_models_for_policy(policy_name: str, models: dict) -> dict:
    outcome_targets_by_state = models.get("outcome_targets_by_state", {})
    return outcome_targets_by_state.get(nuisance_group_for_policy(policy_name), models["outcome_targets"])


def heuristic_risk_threshold(metadata: dict) -> float:
    heuristic_policy = metadata.get("heuristic_policy", {})
    if "risk_threshold" not in heuristic_policy:
        raise KeyError(
            "Missing heuristic_policy.risk_threshold. Re-run `python3 -m causal_rl.train_policies` "
            "so the heuristic cutoff is calibrated before test evaluation."
        )
    return float(heuristic_policy["risk_threshold"])


def heuristic_actions(model, df: pd.DataFrame, columns: list[str], threshold: float) -> np.ndarray:
    late_prob = model.predict_proba(df[columns])[:, 1]
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


def prepare_policy_cache(
    df: pd.DataFrame,
    metadata: dict,
    models: dict,
    policy_output: pd.DataFrame,
    policy_name: str = "causal",
) -> dict:
    state_cols = nuisance_state_columns(policy_name, metadata)
    action = policy_output["policy_action"].to_numpy(dtype=int)
    logged_actions = df["action"].to_numpy(dtype=int)
    behavior_model = behavior_model_for_policy(policy_name, models)
    behavior_prop = behavior_model.predict_proba(df[state_cols])
    logged_propensity = behavior_prop[np.arange(len(df)), logged_actions]

    outcome_models = outcome_models_for_policy(policy_name, models)
    policy_components = predict_reward_components(outcome_models, df, state_cols, action)
    logged_components = predict_reward_components(outcome_models, df, state_cols, logged_actions)

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


def sample_cache(cache: dict, row_indices: np.ndarray) -> dict:
    return {
        "action": cache["action"][row_indices],
        "logged_actions": cache["logged_actions"][row_indices],
        "logged_propensity": cache["logged_propensity"][row_indices],
        "matched": cache["matched"][row_indices],
        "used_fallback": cache["used_fallback"][row_indices],
        "low_support": cache["low_support"][row_indices],
        "policy_components": {key: values[row_indices] for key, values in cache["policy_components"].items()},
        "logged_components": {key: values[row_indices] for key, values in cache["logged_components"].items()},
        "policy_rewards": {key: values[row_indices] for key, values in cache["policy_rewards"].items()},
        "logged_rewards": {key: values[row_indices] for key, values in cache["logged_rewards"].items()},
    }


def build_cluster_index(trajectory_ids: np.ndarray) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    unique_trajs = np.unique(trajectory_ids)
    row_lookup = {traj: np.flatnonzero(trajectory_ids == traj) for traj in unique_trajs}
    return unique_trajs, row_lookup


def sample_cluster_rows(
    unique_trajs: np.ndarray,
    row_lookup: dict[object, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_trajs = rng.choice(unique_trajs, size=len(unique_trajs), replace=True)
    return np.concatenate([row_lookup[traj] for traj in sampled_trajs])


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
        sample_cache_for_rep = sample_cache(masked_cache, sample_idx)
        sample_metrics = summarize_point_metrics(sample_cache_for_rep, masked_rewards[sample_idx], reward_name)
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
    heuristic_threshold = heuristic_risk_threshold(metadata)
    learned_policy_names = ["non_causal_fqi", "causal_fqi"]
    if run_ablations:
        learned_policy_names.extend(ABLATION_POLICY_NAMES)

    outputs = {
        "logged_behavior": pd.DataFrame({"policy_action": df["action"].astype(int)}),
        "always_eco": pd.DataFrame({"policy_action": np.ones(len(df), dtype=int)}),
        "never_eco": pd.DataFrame({"policy_action": np.zeros(len(df), dtype=int)}),
        "heuristic_risk_rule": pd.DataFrame(
            {"policy_action": heuristic_actions(models["heuristic"], df, causal_cols, heuristic_threshold)}
        ),
    }
    for policy_name in learned_policy_names:
        behavior_state_cols = nuisance_state_columns(policy_name, metadata)
        outputs[policy_name] = models[policy_name].policy_action(
            df,
            behavior_model_for_policy(policy_name, models),
            behavior_state_cols,
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
    behavior_state_cols = nuisance_state_columns(policy_name, metadata)

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
            heuristic_threshold = heuristic_risk_threshold(metadata)
            return heuristic_actions(models["heuristic"], next_state_df, causal_cols, heuristic_threshold)
        if policy_name in LEARNED_POLICY_REGISTRY or policy_name == "oracle_fqi":
            policy_df = next_state_df.copy()
            for column in models[policy_name].state_columns:
                if column not in policy_df.columns:
                    policy_df[column] = 0
            for column in behavior_state_cols:
                if column not in policy_df.columns:
                    policy_df[column] = 0
            return models[policy_name].policy_action(
                policy_df,
                behavior_model_for_policy(policy_name, models),
                behavior_state_cols,
                fallback_to_logged=False,
                min_propensity=MIN_PROPENSITY,
                q_gap_threshold=Q_GAP_THRESHOLD,
            )["policy_action"].to_numpy(dtype=int)
        raise ValueError(f"Unknown policy: {policy_name}")

    return _callable


def _run_sweep_on_df(
    eval_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    policy_names: list[str],
    value_col_prefix: str,
    n_jobs: int | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the support-threshold grid over eval_df and return (rows_df, pair_summary)."""
    reward_values = eval_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float)
    specs = [
        (policy_name, min_propensity, q_gap_threshold)
        for policy_name in policy_names
        for min_propensity in SUPPORT_SWEEP_PROPENSITIES
        for q_gap_threshold in SUPPORT_SWEEP_Q_GAPS
    ]

    def _build_row(policy_name: str, min_propensity: float, q_gap_threshold: float) -> dict:
        behavior_state_cols = nuisance_state_columns(policy_name, metadata)
        policy_output = models[policy_name].policy_action(
            eval_df,
            behavior_model_for_policy(policy_name, models),
            behavior_state_cols,
            min_propensity=min_propensity,
            q_gap_threshold=q_gap_threshold,
        )
        cache = prepare_policy_cache(eval_df, metadata, models, policy_output, policy_name)
        summary = summarize_point_metrics(cache, reward_values, "primary")
        return {
            "policy": policy_name,
            "min_propensity": min_propensity,
            "q_gap_threshold": q_gap_threshold,
            f"{value_col_prefix}_policy_value_dr": summary["policy_value_dr"],
            f"{value_col_prefix}_policy_value_ips": summary["policy_value_ips"],
            f"{value_col_prefix}_policy_value_plugin": summary["policy_value_plugin"],
            f"{value_col_prefix}_override_rate": summary["override_rate"],
            f"{value_col_prefix}_fallback_rate": summary["fallback_rate"],
            f"{value_col_prefix}_low_support_rate": summary["low_support_rate"],
            f"{value_col_prefix}_match_rate": summary["match_rate"],
            f"{value_col_prefix}_eco_rate": summary["eco_rate"],
        }

    jobs = min(resolve_n_jobs(n_jobs), len(specs))
    if jobs == 1:
        rows = [_build_row(*spec) for spec in specs]
    else:
        with limit_inner_threads(jobs):
            rows = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_build_row)(*spec) for spec in specs
            )
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return rows_df, pd.DataFrame()
    pair_summary = (
        rows_df.groupby(["min_propensity", "q_gap_threshold"], as_index=False)
        .agg(
            mean_policy_value_dr=(f"{value_col_prefix}_policy_value_dr", "mean"),
            mean_override_rate=(f"{value_col_prefix}_override_rate", "mean"),
            mean_fallback_rate=(f"{value_col_prefix}_fallback_rate", "mean"),
            mean_low_support_rate=(f"{value_col_prefix}_low_support_rate", "mean"),
        )
    )
    pair_summary["conservative_score"] = (
        pair_summary["mean_policy_value_dr"]
        - 0.5 * pair_summary["mean_override_rate"]
        - 0.25 * pair_summary["mean_low_support_rate"]
    )
    return rows_df, pair_summary


def build_support_sweep_df(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    run_ablations: bool,
    n_jobs: int | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run support-threshold sweep on val_df for selection; evaluate selected pair on test_df.

    Returns (sweep_df, best_thresholds) where best_thresholds contains the val-selected
    min_propensity and q_gap_threshold to use for main paper results.
    """
    learned_policy_names = ["non_causal_fqi", "causal_fqi"]
    if run_ablations:
        learned_policy_names.extend(ABLATION_POLICY_NAMES)

    val_rows_df, val_pair_summary = _run_sweep_on_df(
        val_df, metadata, models, learned_policy_names, "val", n_jobs=n_jobs
    )
    test_rows_df, _ = _run_sweep_on_df(
        test_df, metadata, models, learned_policy_names, "test", n_jobs=n_jobs
    )

    if val_rows_df.empty:
        return pd.DataFrame(), {"min_propensity": MIN_PROPENSITY, "q_gap_threshold": Q_GAP_THRESHOLD}

    best_pair = val_pair_summary.sort_values(
        ["conservative_score", "mean_override_rate", "mean_fallback_rate"],
        ascending=[False, True, True],
    ).iloc[0]
    best_thresholds = {
        "min_propensity": float(best_pair["min_propensity"]),
        "q_gap_threshold": float(best_pair["q_gap_threshold"]),
    }

    merge_keys = ["policy", "min_propensity", "q_gap_threshold"]
    sweep_df = val_rows_df.merge(test_rows_df, on=merge_keys, how="left")
    sweep_df["is_default"] = (
        np.isclose(sweep_df["min_propensity"], MIN_PROPENSITY)
        & np.isclose(sweep_df["q_gap_threshold"], Q_GAP_THRESHOLD)
    ).astype(int)
    sweep_df["selected_for_main_paper"] = (
        np.isclose(sweep_df["min_propensity"], best_pair["min_propensity"])
        & np.isclose(sweep_df["q_gap_threshold"], best_pair["q_gap_threshold"])
    ).astype(int)
    return sweep_df, best_thresholds


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
        "non_causal_fqi": (
            "Uses the remaining pre-dispatch compatibility proxy as the non-causal comparator; "
            "distance_km and risk_score are excluded as outcome-adjacent or decision-time-ambiguous fields."
        ),
        "minimal_fqi": (
            "Minimal 5-feature baseline (hour, demand_size, time_window_tightness, "
            "traffic_index, dispatch_delay_min). Quantifies the value of the full "
            "confounder-aware backdoor-style state over the simplest operationally available features."
        ),
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


def policy_family(policy_name: str) -> str:
    if policy_name in CAUSAL_RL_POLICY_NAMES:
        return "causal_rl_family"
    if policy_name == "non_causal_fqi":
        return "non_causal_fqi"
    if policy_name == "minimal_fqi":
        return "minimal_fqi"
    if policy_name == "heuristic_risk_rule":
        return "heuristic"
    if policy_name == "logged_behavior":
        return "logged_behavior"
    if policy_name in {"always_eco", "never_eco"}:
        return "static_rule"
    return "other"


def build_dominance_audit(
    metrics_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Audit whether causal FQI dominance survives across objectives and estimators.

    Higher policy values are better for all included metrics. The audit is a
    falsification surface for broad dominance claims: one row where the
    confounder-aware FQI family is not top is a counterexample to a universal
    "causality + RL wins" statement for the current benchmark.
    """
    rows: list[dict] = []

    def add_row(
        source_df: pd.DataFrame,
        *,
        comparison_scope: str,
        objective_alignment: str,
        estimator: str,
        metric_column: str,
        reward_name: str = "primary",
        segment: str = "",
    ) -> None:
        if source_df.empty or metric_column not in source_df.columns:
            return
        valid = source_df[["policy", metric_column]].dropna().copy()
        if valid.empty or "causal_fqi" not in set(valid["policy"]):
            return
        if valid[valid["policy"].isin(CAUSAL_RL_POLICY_NAMES)].empty:
            return

        valid["rank"] = valid[metric_column].rank(ascending=False, method="dense").astype(int)
        causal_family = valid[valid["policy"].isin(CAUSAL_RL_POLICY_NAMES)]
        ordered = valid.sort_values([metric_column, "policy"], ascending=[False, True]).reset_index(drop=True)
        top_row = ordered.iloc[0]
        causal_row = valid.loc[valid["policy"] == "causal_fqi"].iloc[0]
        family_row = causal_family.sort_values([metric_column, "policy"], ascending=[False, True]).iloc[0]

        non_causal_rows = valid.loc[valid["policy"] == "non_causal_fqi"]
        non_causal_value = float(non_causal_rows[metric_column].iloc[0]) if not non_causal_rows.empty else np.nan
        causal_fqi_value = float(causal_row[metric_column])
        top_value = float(top_row[metric_column])
        family_value = float(family_row[metric_column])
        causal_is_top = causal_row["policy"] == top_row["policy"]
        family_is_top = family_row["policy"] == top_row["policy"]

        rows.append(
            {
                "comparison_scope": comparison_scope,
                "objective_alignment": objective_alignment,
                "estimator": estimator,
                "metric": metric_column,
                "reward_name": reward_name,
                "segment": segment,
                "top_policy": top_row["policy"],
                "top_policy_family": policy_family(str(top_row["policy"])),
                "top_value": top_value,
                "causal_fqi_value": causal_fqi_value,
                "causal_fqi_rank": int(causal_row["rank"]),
                "causal_fqi_gap_to_top": causal_fqi_value - top_value,
                "best_causal_rl_policy": family_row["policy"],
                "best_causal_rl_value": family_value,
                "best_causal_rl_rank": int(family_row["rank"]),
                "best_causal_rl_gap_to_top": family_value - top_value,
                "non_causal_fqi_value": non_causal_value,
                "causal_fqi_beats_non_causal_fqi": (
                    int(causal_fqi_value > non_causal_value)
                    if not np.isnan(non_causal_value)
                    else np.nan
                ),
                "causal_fqi_is_top": int(causal_is_top),
                "causal_rl_family_is_top": int(family_is_top),
                "universal_win_counterexample": int(not family_is_top),
                "claim_result": (
                    "supports_full_causal_fqi"
                    if causal_is_top
                    else "supports_causal_rl_family"
                    if family_is_top
                    else "counterexample_to_universal_causal_rl_win"
                ),
            }
        )

    estimator_specs = [
        ("per_decision_contextual_ope", "dr", "policy_value_dr"),
        ("per_decision_contextual_ope", "snips", "policy_value_ips"),
        ("per_decision_contextual_ope", "plugin", "policy_value_plugin"),
        ("discounted_trajectory_return", "fqe", "policy_value_fqe"),
    ]
    for objective_alignment, estimator, metric_column in estimator_specs:
        add_row(
            metrics_df,
            comparison_scope="overall",
            objective_alignment=objective_alignment,
            estimator=estimator,
            metric_column=metric_column,
        )

    if not reward_df.empty:
        for reward_name, reward_group in reward_df.groupby("reward_name"):
            add_row(
                reward_group,
                comparison_scope="reward_sensitivity",
                objective_alignment="per_decision_contextual_ope",
                estimator="dr",
                metric_column="policy_value_dr",
                reward_name=str(reward_name),
            )

    if not robustness_df.empty:
        for segment, segment_group in robustness_df.groupby("segment"):
            add_row(
                segment_group,
                comparison_scope="operational_segment",
                objective_alignment="per_decision_contextual_ope",
                estimator="dr",
                metric_column="policy_value_dr",
                reward_name="primary",
                segment=str(segment),
            )

    audit_df = pd.DataFrame(rows)
    if audit_df.empty:
        return audit_df
    audit_df["universal_full_causal_fqi_win"] = int(audit_df["causal_fqi_is_top"].all())
    audit_df["universal_causal_rl_family_win"] = int(audit_df["causal_rl_family_is_top"].all())
    audit_df["counterexample_count"] = int(audit_df["universal_win_counterexample"].sum())
    return audit_df


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
                f"{best_row['policy']} outperforms causal_fqi in {segment}, suggesting a simpler operating rule is more stable than the confounder-aware policy in this slice."
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


def cluster_bootstrap_confidence_intervals(
    cache: dict,
    reward_values: np.ndarray,
    reward_name: str,
    policy_name: str,
    bootstrap_reps: int,
    trajectory_ids: np.ndarray,
    scope: str = "overall",
    segment: str | None = None,
    mask: np.ndarray | pd.Series | None = None,
) -> list[dict]:
    """Bootstrap by resampling full trajectories (clusters) instead of individual rows."""
    if bootstrap_reps <= 0:
        return []
    masked_cache = subset_cache(cache, mask)
    masked_rewards = np.asarray(reward_values, dtype=float)
    masked_traj_ids = np.asarray(trajectory_ids)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        masked_rewards = masked_rewards[mask_array]
        masked_traj_ids = masked_traj_ids[mask_array]
    if len(masked_rewards) == 0:
        return []

    unique_trajs, row_lookup = build_cluster_index(masked_traj_ids)
    point_metrics = summarize_point_metrics(masked_cache, masked_rewards, reward_name)
    distributions = {metric: np.empty(bootstrap_reps, dtype=float) for metric in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(stable_seed(policy_name, reward_name, scope, segment or "overall", "cluster"))

    for rep in range(bootstrap_reps):
        row_indices = sample_cluster_rows(unique_trajs, row_lookup, rng)
        sample_cache_for_rep = sample_cache(masked_cache, row_indices)
        sample_metrics = summarize_point_metrics(sample_cache_for_rep, masked_rewards[row_indices], reward_name)
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
                "bootstrap_method": "cluster",
            }
        )
    return rows


def paired_policy_difference_bootstrap(
    policy_caches: dict[str, dict],
    reward_values: np.ndarray,
    reward_name: str,
    trajectory_ids: np.ndarray,
    bootstrap_reps: int,
    pairs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Estimate paired policy-value differences with trajectory-level resampling."""
    if bootstrap_reps <= 0:
        return pd.DataFrame()
    selected_pairs = pairs or KEY_POLICY_DIFFERENCES
    reward_values = np.asarray(reward_values, dtype=float)
    trajectory_ids = np.asarray(trajectory_ids)
    unique_trajs, row_lookup = build_cluster_index(trajectory_ids)
    if len(unique_trajs) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    for left_policy, right_policy, comparison in selected_pairs:
        if left_policy not in policy_caches or right_policy not in policy_caches:
            continue
        left_cache = policy_caches[left_policy]
        right_cache = policy_caches[right_policy]
        left_point = summarize_point_metrics(left_cache, reward_values, reward_name)["policy_value_dr"]
        right_point = summarize_point_metrics(right_cache, reward_values, reward_name)["policy_value_dr"]
        deltas = np.empty(bootstrap_reps, dtype=float)
        rng = np.random.default_rng(stable_seed(comparison, reward_name, "paired_cluster_difference"))
        for rep in range(bootstrap_reps):
            row_indices = sample_cluster_rows(unique_trajs, row_lookup, rng)
            sample_rewards = reward_values[row_indices]
            left_sample = sample_cache(left_cache, row_indices)
            right_sample = sample_cache(right_cache, row_indices)
            left_value = summarize_point_metrics(left_sample, sample_rewards, reward_name)["policy_value_dr"]
            right_value = summarize_point_metrics(right_sample, sample_rewards, reward_name)["policy_value_dr"]
            deltas[rep] = left_value - right_value
        ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
        rows.append(
            {
                "comparison": comparison,
                "left_policy": left_policy,
                "right_policy": right_policy,
                "reward_name": reward_name,
                "metric": "policy_value_dr",
                "left_policy_value_dr": left_point,
                "right_policy_value_dr": right_point,
                "paired_difference": left_point - right_point,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "probability_difference_gt_zero": float(np.mean(deltas > 0.0)),
                "bootstrap_reps": bootstrap_reps,
                "bootstrap_method": "paired_cluster",
            }
        )
    return pd.DataFrame(rows)


def build_policy_sensitivity_diagnostics(
    test_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    supported_policy_cache: dict,
) -> pd.DataFrame:
    """Compare support-constrained causal FQI to greedy and latent-oracle variants."""
    reward_values = test_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float)
    rows: list[dict] = []
    supported_actions = supported_policy_cache["action"]

    variants: list[tuple[str, str, dict, bool, str]] = [
        (
            "causal_fqi_supported",
            "causal_fqi",
            supported_policy_cache,
            True,
            "validation-selected support and Q-gap fallback",
        )
    ]

    if "causal_fqi" in models:
        causal_greedy_output = pd.DataFrame({"policy_action": models["causal_fqi"].greedy_action(test_df)})
        variants.append(
            (
                "causal_fqi_greedy_no_fallback",
                "causal_fqi",
                prepare_policy_cache(test_df, metadata, models, causal_greedy_output, "causal_fqi"),
                True,
                "greedy learned action with no fallback",
            )
        )

    if "oracle_fqi" in models:
        oracle_greedy_output = pd.DataFrame({"policy_action": models["oracle_fqi"].greedy_action(test_df)})
        variants.append(
            (
                "oracle_fqi_greedy_no_fallback",
                "oracle_fqi",
                prepare_policy_cache(test_df, metadata, models, oracle_greedy_output, "oracle_fqi"),
                False,
                "non-deployable latent-state oracle; no fallback",
            )
        )

    for variant, base_policy, cache, deployable, note in variants:
        metrics = summarize_point_metrics(cache, reward_values, "primary")
        rows.append(
            {
                "variant": variant,
                "base_policy": base_policy,
                "deployable": int(deployable),
                "uses_latent_state": int(base_policy == "oracle_fqi"),
                "nuisance_state_family": nuisance_group_for_policy(base_policy),
                "policy_value_dr": metrics["policy_value_dr"],
                "policy_value_ips": metrics["policy_value_ips"],
                "policy_value_plugin": metrics["policy_value_plugin"],
                "match_rate": metrics["match_rate"],
                "override_rate": metrics["override_rate"],
                "fallback_rate": metrics["fallback_rate"],
                "low_support_rate": metrics["low_support_rate"],
                "eco_rate": metrics["eco_rate"],
                "action_disagreement_vs_causal_supported": float(np.mean(cache["action"] != supported_actions)),
                "note": note,
            }
        )

    sensitivity_df = pd.DataFrame(rows)
    if not sensitivity_df.empty:
        reference_value = float(
            sensitivity_df.loc[
                sensitivity_df["variant"] == "causal_fqi_supported",
                "policy_value_dr",
            ].iloc[0]
        )
        sensitivity_df["delta_vs_causal_fqi_supported"] = sensitivity_df["policy_value_dr"] - reference_value
    return sensitivity_df


def compute_common_support_diagnostics(
    policy_caches: dict[str, dict],
) -> pd.DataFrame:
    """Compute propensity overlap diagnostics per policy from pre-computed caches."""
    rows = []
    for policy_name, cache in policy_caches.items():
        prop = cache["logged_propensity"]
        rows.append(
            {
                "policy": policy_name,
                "n_rows": len(prop),
                "propensity_min": float(np.min(prop)),
                "propensity_max": float(np.max(prop)),
                "propensity_mean": float(np.mean(prop)),
                "propensity_p5": float(np.quantile(prop, 0.05)),
                "pct_below_tau_mu": float(np.mean(prop < MIN_PROPENSITY) * 100),
                "effective_override_count": int(np.sum(~cache["matched"])),
                "pct_override_low_support": float(
                    np.mean(~cache["matched"] & (prop < MIN_PROPENSITY)) * 100
                ),
                "eco_rate": float(np.mean(cache["action"])),
            }
        )
    return pd.DataFrame(rows)


def compute_late_day_failure_diagnosis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    policy_caches: dict[str, dict],
) -> pd.DataFrame:
    """Diagnose why confounder-aware causal_fqi underperforms in late_day operations."""
    key_features = ["traffic_index", "time_window_tightness", "dispatch_delay_min", "rolling_mean_traffic"]
    train_late = train_df[train_df["hour"] >= 17]
    test_late = test_df[test_df["hour"] >= 17]

    rows = []
    for feat in key_features:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue
        rows.append(
            {
                "diagnostic_type": "feature_distribution",
                "feature_or_hour": feat,
                "train_mean": float(train_late[feat].mean()) if len(train_late) > 0 else np.nan,
                "train_std": float(train_late[feat].std()) if len(train_late) > 0 else np.nan,
                "test_mean": float(test_late[feat].mean()) if len(test_late) > 0 else np.nan,
                "test_std": float(test_late[feat].std()) if len(test_late) > 0 else np.nan,
                "delta_mean": float(test_late[feat].mean() - train_late[feat].mean())
                if len(train_late) > 0 and len(test_late) > 0
                else np.nan,
            }
        )

    for hour in range(17, 22):
        hour_mask = test_df["hour"] == hour
        if not hour_mask.any():
            continue
        row: dict = {
            "diagnostic_type": "eco_rate_by_hour",
            "feature_or_hour": str(hour),
            "train_mean": np.nan,
            "train_std": np.nan,
            "test_mean": np.nan,
            "test_std": np.nan,
            "delta_mean": np.nan,
        }
        for policy_name in ["causal_fqi", "logged_behavior"]:
            if policy_name in policy_caches:
                eco = float(np.mean(policy_caches[policy_name]["action"][hour_mask.to_numpy()]))
                row[f"{policy_name}_eco_rate"] = eco
        rows.append(row)

    return pd.DataFrame(rows)


def compute_heuristic_diagnostics(
    test_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    policy_outputs: dict[str, pd.DataFrame],
    policy_caches: dict[str, dict],
) -> pd.DataFrame:
    """Explain the validation-frozen heuristic through risk bins and action agreement."""
    causal_cols = metadata["causal_state_columns"]
    threshold = heuristic_risk_threshold(metadata)
    risk_quantile = float(metadata.get("heuristic_policy", {}).get("risk_quantile", np.nan))
    lateness_risk = models["heuristic"].predict_proba(test_df[causal_cols])[:, 1]
    action_by_policy = {
        policy_name: output["policy_action"].to_numpy(dtype=int)
        for policy_name, output in policy_outputs.items()
    }
    heuristic_actions_array = action_by_policy["heuristic_risk_rule"]
    logged_actions = test_df["action"].to_numpy(dtype=int)
    heuristic_cache = policy_caches.get("heuristic_risk_rule", {})
    policy_components = heuristic_cache.get("policy_components", {})

    def _masked_mean(values: np.ndarray | pd.Series, mask: np.ndarray) -> float:
        values_array = np.asarray(values)
        return float(np.mean(values_array[mask])) if mask.any() else np.nan

    def _component_mean(component: str, mask: np.ndarray) -> float:
        if component not in policy_components:
            return np.nan
        return _masked_mean(policy_components[component], mask)

    def _policy_eco_rate(policy_name: str, mask: np.ndarray) -> float:
        if policy_name not in action_by_policy:
            return np.nan
        return _masked_mean(action_by_policy[policy_name], mask)

    def _agreement(other_actions: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(heuristic_actions_array[mask] == other_actions[mask])) if mask.any() else np.nan

    def _row(diagnostic_type: str, group: str, mask: np.ndarray) -> dict:
        return {
            "diagnostic_type": diagnostic_type,
            "group": group,
            "n_rows": int(np.sum(mask)),
            "risk_threshold": threshold,
            "risk_quantile": risk_quantile,
            "lateness_risk_mean": _masked_mean(lateness_risk, mask),
            "lateness_risk_min": float(np.min(lateness_risk[mask])) if mask.any() else np.nan,
            "lateness_risk_max": float(np.max(lateness_risk[mask])) if mask.any() else np.nan,
            "heuristic_eco_rate": _masked_mean(heuristic_actions_array, mask),
            "logged_behavior_eco_rate": _masked_mean(logged_actions, mask),
            "causal_fqi_eco_rate": _policy_eco_rate("causal_fqi", mask),
            "non_causal_fqi_eco_rate": _policy_eco_rate("non_causal_fqi", mask),
            "agreement_with_logged_behavior": _agreement(logged_actions, mask),
            "agreement_with_causal_fqi": (
                _agreement(action_by_policy["causal_fqi"], mask)
                if "causal_fqi" in action_by_policy
                else np.nan
            ),
            "agreement_with_non_causal_fqi": (
                _agreement(action_by_policy["non_causal_fqi"], mask)
                if "non_causal_fqi" in action_by_policy
                else np.nan
            ),
            "estimated_lateness_min": _component_mean("lateness_min", mask),
            "estimated_co2_kg": _component_mean("co2_kg", mask),
            "estimated_crash": _component_mean("crash", mask),
            "estimated_near_miss": _component_mean("near_miss", mask),
            "estimated_on_time": _component_mean("on_time", mask),
        }

    rows = [_row("summary", "overall", np.ones(len(test_df), dtype=bool))]

    decile_labels = pd.qcut(
        pd.Series(lateness_risk).rank(method="first"),
        q=10,
        labels=False,
    ).to_numpy()
    for decile in range(10):
        rows.append(_row("risk_decile", f"decile_{decile + 1:02d}", decile_labels == decile))

    for segment_name, rule in ROBUSTNESS_SEGMENTS.items():
        rows.append(_row("segment", segment_name, subset_mask(test_df, rule).to_numpy()))

    return pd.DataFrame(rows)


def compute_feature_importance_summary(
    models: dict,
    train_df: pd.DataFrame,
    n_repeats: int = FQE_N_REPEATS,
    n_jobs: int | str | None = None,
) -> pd.DataFrame:
    """Compute permutation importance for the confounder-aware causal_fqi Q-function."""
    if "causal_fqi" not in models:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    causal_fqi = models["causal_fqi"]
    jobs = min(resolve_n_jobs(n_jobs), 2)
    if jobs == 1:
        fi_a1 = causal_fqi.get_feature_importances(train_df, action=1, n_repeats=n_repeats)
        fi_a0 = causal_fqi.get_feature_importances(train_df, action=0, n_repeats=n_repeats)
    else:
        fi_a1, fi_a0 = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(causal_fqi.get_feature_importances)(train_df, action=action, n_repeats=n_repeats)
            for action in (1, 0)
        )
    merged = fi_a1.merge(fi_a0, on="feature", suffixes=("_a1", "_a0"))
    merged["importance_mean"] = (merged["importance_mean_a1"] + merged["importance_mean_a0"]) / 2
    merged["importance_std"] = (merged["importance_std_a1"] + merged["importance_std_a0"]) / 2
    return (
        merged[["feature", "importance_mean", "importance_std"]]
        .sort_values("importance_mean", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


def evaluate_policy_block(
    policy_name: str,
    policy_output: pd.DataFrame,
    test_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    bootstrap_reps: int,
    trajectory_ids: np.ndarray,
) -> dict:
    primary_reward = test_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float)
    policy_action_values = policy_output["policy_action"].to_numpy(dtype=int)
    fqe_evaluator = FittedQEvaluator(nuisance_state_columns(policy_name, metadata))
    fqe_evaluator.fit(train_val_df, policy_callable(policy_name, models, metadata), reward_column="reward")
    fqe_value = fqe_evaluator.evaluate_policy_value(test_df, policy_action_values)

    cache = prepare_policy_cache(test_df, metadata, models, policy_output, policy_name)
    metric_row = compute_policy_metrics(
        cache,
        primary_reward,
        "primary",
        policy_name,
        policy_value_fqe=fqe_value,
    )
    bootstrap_rows = bootstrap_confidence_intervals(
        cache,
        primary_reward,
        "primary",
        policy_name,
        bootstrap_reps=bootstrap_reps,
        scope="overall",
    )
    cluster_bootstrap_rows = cluster_bootstrap_confidence_intervals(
        cache,
        primary_reward,
        "primary",
        policy_name,
        bootstrap_reps=bootstrap_reps,
        trajectory_ids=trajectory_ids,
        scope="overall",
    )

    robustness_rows = []
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

    reward_rows = [
        compute_policy_metrics(
            cache,
            test_df[reward_column].to_numpy(dtype=float),
            reward_name,
            policy_name,
        )
        for reward_name, reward_column in REWARD_COLUMNS.items()
    ]

    return {
        "policy_name": policy_name,
        "actions": policy_action_values,
        "metric_row": metric_row,
        "robustness_rows": robustness_rows,
        "reward_rows": reward_rows,
        "bootstrap_rows": bootstrap_rows,
        "cluster_bootstrap_rows": cluster_bootstrap_rows,
        "cache": cache,
        "fqe_history": fqe_evaluator.convergence_history,
    }


def evaluate_all(
    bootstrap_reps: int = BOOTSTRAP_REPS,
    run_ablations: bool = True,
    n_jobs: int | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = resolve_n_jobs(n_jobs)
    df, metadata, models = load_inputs()
    train_val_df = df[df["split"].isin(["train", "val"])].copy()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    support_sweep_df, best_thresholds = build_support_sweep_df(
        val_df, test_df, metadata, models, run_ablations=run_ablations, n_jobs=jobs
    )
    policy_outputs = get_policy_outputs(
        test_df,
        metadata,
        models,
        run_ablations=run_ablations,
        min_propensity=best_thresholds["min_propensity"],
        q_gap_threshold=best_thresholds["q_gap_threshold"],
    )

    actions_frame = test_df[["trajectory_id", "t", "split", "action", "reward"]].copy()
    metric_rows = []
    robustness_rows = []
    reward_rows = []
    bootstrap_rows = []
    cluster_bootstrap_rows = []
    policy_caches: dict[str, dict] = {}
    fqe_convergence_history: dict[str, list[float]] = {}
    trajectory_ids = test_df["trajectory_id"].to_numpy()

    policy_items = list(policy_outputs.items())
    policy_jobs = min(jobs, len(policy_items))
    if policy_jobs == 1:
        policy_blocks = [
            evaluate_policy_block(
                policy_name,
                policy_output,
                test_df,
                train_val_df,
                metadata,
                models,
                bootstrap_reps,
                trajectory_ids,
            )
            for policy_name, policy_output in policy_items
        ]
    else:
        with limit_inner_threads(policy_jobs):
            policy_blocks = Parallel(n_jobs=policy_jobs, prefer="threads")(
                delayed(evaluate_policy_block)(
                    policy_name,
                    policy_output,
                    test_df,
                    train_val_df,
                    metadata,
                    models,
                    bootstrap_reps,
                    trajectory_ids,
                )
                for policy_name, policy_output in policy_items
            )

    for block in policy_blocks:
        policy_name = block["policy_name"]
        actions_frame[f"{policy_name}_action"] = block["actions"]
        metric_rows.append(block["metric_row"])
        robustness_rows.extend(block["robustness_rows"])
        reward_rows.extend(block["reward_rows"])
        bootstrap_rows.extend(block["bootstrap_rows"])
        cluster_bootstrap_rows.extend(block["cluster_bootstrap_rows"])
        policy_caches[policy_name] = block["cache"]
        fqe_convergence_history[policy_name] = block["fqe_history"]

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

    ablation_df = build_ablation_comparison(metrics_df, metadata)
    diagnostics_df = build_estimator_diagnostics(metrics_df)
    interpretation_df = build_interpretation_summary(metrics_df, robustness_df)
    dominance_audit_df = build_dominance_audit(metrics_df, reward_df, robustness_df)

    common_support_df = compute_common_support_diagnostics(policy_caches)
    late_day_df = compute_late_day_failure_diagnosis(train_df, test_df, policy_caches)
    heuristic_diagnostics_df = compute_heuristic_diagnostics(
        test_df,
        metadata,
        models,
        policy_outputs,
        policy_caches,
    )
    feature_importance_df = compute_feature_importance_summary(models, train_df, n_jobs=jobs)
    cluster_bootstrap_df = pd.DataFrame(cluster_bootstrap_rows)
    policy_difference_df = paired_policy_difference_bootstrap(
        policy_caches,
        test_df[REWARD_COLUMNS["primary"]].to_numpy(dtype=float),
        "primary",
        trajectory_ids,
        bootstrap_reps,
    )
    oracle_sensitivity_df = (
        build_policy_sensitivity_diagnostics(test_df, metadata, models, policy_caches["causal_fqi"])
        if "causal_fqi" in policy_caches
        else pd.DataFrame()
    )
    fqe_convergence_df = pd.DataFrame(
        [
            {"policy": policy_name, "iteration": i, "mean_abs_q_change": delta}
            for policy_name, history in fqe_convergence_history.items()
            for i, delta in enumerate(history)
        ]
    )

    actions_frame.to_csv(POLICY_ACTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    robustness_df.to_csv(ROBUSTNESS_PATH, index=False)
    reward_df.to_csv(REWARD_SENSITIVITY_PATH, index=False)
    bootstrap_df.to_csv(BOOTSTRAP_SUMMARY_PATH, index=False)
    diagnostics_df.to_csv(ESTIMATOR_DIAGNOSTICS_PATH, index=False)
    support_sweep_df.to_csv(SUPPORT_SWEEP_PATH, index=False)
    ablation_df.to_csv(ABLATION_COMPARISON_PATH, index=False)
    interpretation_df.to_csv(INTERPRETATION_SUMMARY_PATH, index=False)
    dominance_audit_df.to_csv(DOMINANCE_AUDIT_PATH, index=False)
    common_support_df.to_csv(COMMON_SUPPORT_PATH, index=False)
    late_day_df.to_csv(RESULTS_DIR / "late_day_diagnosis.csv", index=False)
    heuristic_diagnostics_df.to_csv(HEURISTIC_DIAGNOSTICS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    cluster_bootstrap_df.to_csv(CLUSTER_BOOTSTRAP_PATH, index=False)
    policy_difference_df.to_csv(POLICY_DIFFERENCE_BOOTSTRAP_PATH, index=False)
    oracle_sensitivity_df.to_csv(ORACLE_SENSITIVITY_PATH, index=False)
    fqe_convergence_df.to_csv(FQE_CONVERGENCE_PATH, index=False)
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
    parser = argparse.ArgumentParser(description="Evaluate offline policy-learning methods with bootstrap and ablation outputs.")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--run-ablations", dest="run_ablations", action="store_true", default=True)
    parser.add_argument("--skip-ablations", dest="run_ablations", action="store_false")
    parser.add_argument("--n-jobs", default=None, help="Parallel workers: auto, -1, or a positive integer.")
    args = parser.parse_args()

    n_jobs = resolve_n_jobs(args.n_jobs)
    metrics_df, robustness_df, reward_df = evaluate_all(
        bootstrap_reps=args.bootstrap_reps,
        run_ablations=args.run_ablations,
        n_jobs=n_jobs,
    )
    print(f"Saved metrics to {METRICS_PATH} with {len(metrics_df)} policies")
    print(f"Saved robustness results to {ROBUSTNESS_PATH} with {len(robustness_df)} rows")
    print(f"Saved reward sensitivity results to {REWARD_SENSITIVITY_PATH} with {len(reward_df)} rows")
    print(f"Saved bootstrap intervals to {BOOTSTRAP_SUMMARY_PATH}")
    print(f"Saved cluster bootstrap intervals to {CLUSTER_BOOTSTRAP_PATH}")
    print(f"Saved paired policy-difference bootstrap intervals to {POLICY_DIFFERENCE_BOOTSTRAP_PATH}")
    print(f"Saved oracle/no-fallback policy sensitivity to {ORACLE_SENSITIVITY_PATH}")
    print(f"Saved support sweep to {SUPPORT_SWEEP_PATH} (thresholds selected on val set)")
    print(f"Saved ablation comparison to {ABLATION_COMPARISON_PATH}")
    print(f"Saved interpretation summary to {INTERPRETATION_SUMMARY_PATH}")
    print(f"Saved dominance audit to {DOMINANCE_AUDIT_PATH}")
    print(f"Saved common support diagnostics to {COMMON_SUPPORT_PATH}")
    print(f"Saved heuristic diagnostics to {HEURISTIC_DIAGNOSTICS_PATH}")
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved FQE convergence to {FQE_CONVERGENCE_PATH}")
    print(f"Evaluation used n_jobs={n_jobs}")


if __name__ == "__main__":
    main()
