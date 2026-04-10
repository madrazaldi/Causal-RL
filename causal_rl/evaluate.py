from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .config import (
    ACTION_COLUMN,
    CAUSAL_BACKDOOR_COLUMNS,
    DECISION_LOG_PATH,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    OUTCOME_MODEL_TARGETS,
    POLICY_ACTIONS_PATH,
    REWARD_COLUMNS,
    REWARD_SENSITIVITY_PATH,
    RESULTS_DIR,
    ROBUSTNESS_PATH,
    ROBUSTNESS_SEGMENTS,
)
from .ope import FittedQEvaluator, doubly_robust, self_normalized_ips


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(DECISION_LOG_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    models = {
        "behavior": joblib.load(MODELS_DIR / "behavior_model.joblib"),
        "heuristic": joblib.load(MODELS_DIR / "heuristic_model.joblib"),
        "reward_outcome": joblib.load(MODELS_DIR / "reward_outcome_model.joblib"),
        "outcome_targets": joblib.load(MODELS_DIR / "outcome_targets.joblib"),
        "causal_fqi": joblib.load(MODELS_DIR / "causal_fqi.joblib"),
        "non_causal_fqi": joblib.load(MODELS_DIR / "non_causal_fqi.joblib"),
    }
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


def get_policy_outputs(df: pd.DataFrame, metadata: dict, models: dict) -> dict[str, pd.DataFrame]:
    causal_cols = metadata["causal_state_columns"]

    outputs = {
        "logged_behavior": pd.DataFrame({"policy_action": df["action"].astype(int)}),
        "always_eco": pd.DataFrame({"policy_action": np.ones(len(df), dtype=int)}),
        "never_eco": pd.DataFrame({"policy_action": np.zeros(len(df), dtype=int)}),
        "heuristic_risk_rule": pd.DataFrame(
            {"policy_action": heuristic_actions(models["heuristic"], df, causal_cols)}
        ),
        "non_causal_fqi": models["non_causal_fqi"].policy_action(df, models["behavior"], causal_cols),
        "causal_fqi": models["causal_fqi"].policy_action(df, models["behavior"], causal_cols),
    }
    return outputs


def compute_policy_metrics(
    test_df: pd.DataFrame,
    metadata: dict,
    models: dict,
    policy_name: str,
    policy_output: pd.DataFrame,
    reward_column: str = REWARD_COLUMNS["primary"],
    policy_value_fqe: float | None = None,
) -> dict:
    causal_cols = metadata["causal_state_columns"]
    action = policy_output["policy_action"].to_numpy(dtype=int)
    behavior_prop = models["behavior"].predict_proba(test_df[causal_cols])
    logged_actions = test_df["action"].to_numpy(dtype=int)
    logged_propensity = behavior_prop[np.arange(len(test_df)), logged_actions]

    reward_features_logged = test_df[causal_cols].copy()
    reward_features_logged[ACTION_COLUMN] = logged_actions
    q_logged = models["reward_outcome"].predict(reward_features_logged)

    reward_features_policy = test_df[causal_cols].copy()
    reward_features_policy[ACTION_COLUMN] = action
    q_policy = models["reward_outcome"].predict(reward_features_policy)

    component_predictions = predict_reward_components(models["outcome_targets"], test_df, causal_cols, action)

    matched = action == logged_actions
    metrics = {
        "policy": policy_name,
        "reward_name": reward_column,
        "direct_match_reward": float(test_df.loc[matched, reward_column].mean()) if matched.any() else np.nan,
        "match_rate": float(matched.mean()),
        "policy_value_plugin": float(np.mean(q_policy)),
        "policy_value_ips": self_normalized_ips(
            test_df[reward_column].to_numpy(dtype=float), logged_actions, action, logged_propensity
        ),
        "policy_value_dr": doubly_robust(
            test_df[reward_column].to_numpy(dtype=float),
            logged_actions,
            action,
            logged_propensity,
            q_logged,
            q_policy,
        ),
        "policy_value_fqe": float(policy_value_fqe) if policy_value_fqe is not None else np.nan,
        "estimated_lateness_min": float(np.mean(component_predictions["lateness_min"])),
        "estimated_co2_kg": float(np.mean(component_predictions["co2_kg"])),
        "estimated_crash": float(np.mean(component_predictions["crash"])),
        "estimated_near_miss": float(np.mean(component_predictions["near_miss"])),
        "estimated_on_time": float(np.mean(component_predictions["on_time"])),
        "eco_rate": float(np.mean(action)),
    }
    if "used_fallback" in policy_output:
        metrics["fallback_rate"] = float(policy_output["used_fallback"].mean())
        metrics["low_support_rate"] = float(policy_output["low_support"].mean())
    else:
        metrics["fallback_rate"] = 0.0
        metrics["low_support_rate"] = 0.0
    return metrics


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
    non_causal_cols = metadata["non_causal_state_columns"]

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
        if policy_name == "causal_fqi":
            return models["causal_fqi"].policy_action(
                next_state_df, models["behavior"], causal_cols, fallback_to_logged=False
            )["policy_action"].to_numpy(dtype=int)
        if policy_name == "non_causal_fqi":
            policy_df = next_state_df.copy()
            for column in non_causal_cols:
                if column not in policy_df.columns:
                    policy_df[column] = 0
            return models["non_causal_fqi"].policy_action(
                policy_df, models["behavior"], causal_cols, fallback_to_logged=False
            )["policy_action"].to_numpy(dtype=int)
        raise ValueError(f"Unknown policy: {policy_name}")

    return _callable


def evaluate_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata, models = load_inputs()
    train_val_df = df[df["split"].isin(["train", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()
    policy_outputs = get_policy_outputs(test_df, metadata, models)

    actions_frame = test_df[["trajectory_id", "t", "split", "action", "reward"]].copy()
    metric_rows = []
    robustness_rows = []
    reward_rows = []

    for name, policy_output in policy_outputs.items():
        actions_frame[f"{name}_action"] = policy_output["policy_action"].to_numpy(dtype=int)
        policy_action_values = policy_output["policy_action"].to_numpy(dtype=int)
        fqe_value = FittedQEvaluator(metadata["causal_state_columns"]).fit(
            train_val_df.copy(), policy_callable(name, models, metadata), reward_column="reward"
        ).evaluate_policy_value(test_df, policy_action_values)

        metrics = compute_policy_metrics(
            test_df, metadata, models, name, policy_output, policy_value_fqe=fqe_value
        )
        metric_rows.append(metrics)

        for segment_name, rule in ROBUSTNESS_SEGMENTS.items():
            mask = subset_mask(test_df, rule)
            segment_df = test_df.loc[mask].copy()
            segment_policy = policy_output.iloc[np.where(mask.to_numpy())[0]].copy()
            if len(segment_df) == 0:
                continue
            row = compute_policy_metrics(
                segment_df, metadata, models, name, segment_policy, policy_value_fqe=fqe_value
            )
            row["segment"] = segment_name
            robustness_rows.append(row)

        for reward_name, reward_column in REWARD_COLUMNS.items():
            reward_metrics = compute_policy_metrics(
                test_df.assign(reward=test_df[reward_column]),
                metadata,
                models,
                name,
                policy_output,
                reward_column,
                policy_value_fqe=fqe_value,
            )
            reward_rows.append(
                {
                    "reward_name": reward_name,
                    "policy": name,
                    "policy_value_plugin": reward_metrics["policy_value_plugin"],
                    "policy_value_ips": reward_metrics["policy_value_ips"],
                    "policy_value_dr": reward_metrics["policy_value_dr"],
                    "policy_value_fqe": reward_metrics["policy_value_fqe"],
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("policy_value_dr", ascending=False)
    robustness_df = pd.DataFrame(robustness_rows)
    reward_df = pd.DataFrame(reward_rows)

    actions_frame.to_csv(POLICY_ACTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    robustness_df.to_csv(ROBUSTNESS_PATH, index=False)
    reward_df.to_csv(REWARD_SENSITIVITY_PATH, index=False)
    metrics_df[
        [
            "policy",
            "policy_value_plugin",
            "policy_value_ips",
            "policy_value_dr",
            "policy_value_fqe",
            "estimated_lateness_min",
            "estimated_co2_kg",
            "estimated_on_time",
            "eco_rate",
            "fallback_rate",
        ]
    ].to_csv(MAIN_RESULTS_TABLE_PATH, index=False)
    return metrics_df, robustness_df, reward_df


def main() -> None:
    metrics_df, robustness_df, reward_df = evaluate_all()
    print(f"Saved metrics to {METRICS_PATH} with {len(metrics_df)} policies")
    print(f"Saved robustness results to {ROBUSTNESS_PATH} with {len(robustness_df)} rows")
    print(f"Saved reward sensitivity results to {REWARD_SENSITIVITY_PATH} with {len(reward_df)} rows")


if __name__ == "__main__":
    main()
