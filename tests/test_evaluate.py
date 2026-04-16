from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_rl.config import SUPPORT_SWEEP_PROPENSITIES, SUPPORT_SWEEP_Q_GAPS
from causal_rl.evaluate import (
    BOOTSTRAP_METRICS,
    bootstrap_confidence_intervals,
    build_support_sweep_df,
    cluster_bootstrap_confidence_intervals,
    get_policy_outputs,
    prepare_policy_cache,
)


def make_eval_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "hour": [8, 9, 10, 11, 12, 13],
            "traffic_index": [0.2, 0.8, 0.4, 0.7, 0.3, 0.9],
            "action": [1, 0, 1, 0, 1, 0],
        }
    )
    df["reward_primary"] = np.array([-2.0, -5.0, -3.0, -4.5, -2.5, -5.5])
    df["reward_service_heavy"] = df["reward_primary"] - 0.5
    df["reward_sustainability_heavy"] = df["reward_primary"] + 0.25
    return df


class DummyBehaviorModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eco_prob = np.where(X["traffic_index"].to_numpy(dtype=float) < 0.5, 0.78, 0.22)
        return np.column_stack([1.0 - eco_prob, eco_prob])


class DummyHeuristicModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        late_prob = np.clip(0.2 + 0.7 * X["traffic_index"].to_numpy(dtype=float), 0.05, 0.95)
        return np.column_stack([1.0 - late_prob, late_prob])


@dataclass
class DummyOutcomeModel:
    target: str

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        action = X["eco_mode"].to_numpy(dtype=float)
        traffic = X["traffic_index"].to_numpy(dtype=float)
        hour = X["hour"].to_numpy(dtype=float)
        if self.target == "lateness_min":
            return 1.0 + 2.0 * traffic + 0.4 * (1.0 - action) + 0.01 * hour
        if self.target == "co2_kg":
            return 3.4 - 0.35 * action + 0.15 * traffic
        if self.target == "crash":
            return 0.03 + 0.01 * traffic + 0.005 * (1.0 - action)
        if self.target == "near_miss":
            return 0.08 + 0.02 * traffic + 0.01 * (1.0 - action)
        if self.target == "on_time":
            return 0.86 - 0.18 * traffic + 0.02 * action
        raise ValueError(self.target)


@dataclass
class DummyPolicy:
    name: str

    def policy_action(
        self,
        df: pd.DataFrame,
        behavior_model,
        behavior_state_columns: list[str],
        fallback_to_logged: bool = True,
        min_propensity: float = 0.10,
        q_gap_threshold: float = 0.25,
    ) -> pd.DataFrame:
        traffic = df["traffic_index"].to_numpy(dtype=float)
        if self.name == "non_causal_fqi":
            greedy = (df["hour"].to_numpy(dtype=int) % 2 == 0).astype(int)
            q_gap = np.abs(traffic - 0.45) + 0.20
        elif self.name == "causal_no_history_fqi":
            greedy = (traffic < 0.40).astype(int)
            q_gap = np.abs(traffic - 0.40) + 0.05
        elif self.name == "causal_no_vehicle_id_fqi":
            greedy = (traffic < 0.70).astype(int)
            q_gap = np.abs(traffic - 0.65) + 0.12
        else:
            greedy = (traffic < 0.55).astype(int)
            q_gap = np.abs(traffic - 0.55) + 0.10

        prop = behavior_model.predict_proba(df[behavior_state_columns])
        chosen_propensity = prop[np.arange(len(df)), greedy]
        low_support = chosen_propensity < min_propensity
        low_gap = q_gap < q_gap_threshold
        used_fallback = low_support | low_gap
        final_action = greedy.copy()
        if fallback_to_logged:
            final_action[used_fallback] = df["action"].to_numpy(dtype=int)[used_fallback]

        return pd.DataFrame(
            {
                "policy_action": final_action,
                "greedy_action": greedy,
                "q0": 1.0 - traffic,
                "q1": traffic,
                "q_gap": q_gap,
                "chosen_propensity": chosen_propensity,
                "low_support": low_support.astype(int),
                "low_gap": low_gap.astype(int),
                "used_fallback": used_fallback.astype(int),
            }
        )


def make_models() -> dict:
    return {
        "behavior": DummyBehaviorModel(),
        "heuristic": DummyHeuristicModel(),
        "outcome_targets": {
            target: DummyOutcomeModel(target)
            for target in ["lateness_min", "co2_kg", "crash", "near_miss", "on_time"]
        },
        "non_causal_fqi": DummyPolicy("non_causal_fqi"),
        "causal_fqi": DummyPolicy("causal_fqi"),
        "causal_no_history_fqi": DummyPolicy("causal_no_history_fqi"),
        "causal_no_vehicle_id_fqi": DummyPolicy("causal_no_vehicle_id_fqi"),
        "minimal_fqi": DummyPolicy("minimal_fqi"),
    }


def make_metadata() -> dict:
    return {
        "causal_state_columns": ["hour", "traffic_index"],
        "ablation_definitions": {
            "causal_no_history_fqi": {
                "removed_columns": ["rolling_mean_traffic", "prior_eco_mode"],
                "note": "Drops history features.",
            },
            "causal_no_vehicle_id_fqi": {
                "removed_columns": ["vehicle_id"],
                "note": "Drops vehicle identity.",
            },
            "minimal_fqi": {
                "removed_columns": ["many_columns"],
                "note": "Minimal 5-feature baseline.",
            },
        },
    }


def test_get_policy_outputs_includes_ablation_variants() -> None:
    outputs = get_policy_outputs(make_eval_frame(), make_metadata(), make_models(), run_ablations=True)

    for policy_name in ["causal_fqi", "non_causal_fqi", "causal_no_history_fqi", "causal_no_vehicle_id_fqi", "minimal_fqi"]:
        assert policy_name in outputs
        assert "policy_action" in outputs[policy_name].columns


def test_bootstrap_confidence_intervals_are_deterministic_and_ordered() -> None:
    df = make_eval_frame()
    metadata = make_metadata()
    models = make_models()
    outputs = get_policy_outputs(df, metadata, models, run_ablations=True)
    cache = prepare_policy_cache(df, metadata, models, outputs["causal_fqi"])

    rows_first = bootstrap_confidence_intervals(
        cache,
        df["reward_primary"].to_numpy(dtype=float),
        "primary",
        "causal_fqi",
        bootstrap_reps=25,
        scope="overall",
    )
    rows_second = bootstrap_confidence_intervals(
        cache,
        df["reward_primary"].to_numpy(dtype=float),
        "primary",
        "causal_fqi",
        bootstrap_reps=25,
        scope="overall",
    )

    assert rows_first == rows_second
    assert len(rows_first) == len(BOOTSTRAP_METRICS)
    for row in rows_first:
        assert row["ci_low"] <= row["ci_high"]


def test_support_sweep_outputs_all_threshold_pairs() -> None:
    val_df = make_eval_frame()
    test_df = make_eval_frame()
    sweep_df, best_thresholds = build_support_sweep_df(
        val_df, test_df, make_metadata(), make_models(), run_ablations=True
    )
    expected_policies = {"non_causal_fqi", "causal_fqi", "causal_no_history_fqi", "causal_no_vehicle_id_fqi", "minimal_fqi"}

    assert set(sweep_df["policy"]) == expected_policies
    assert len(sweep_df) == len(expected_policies) * len(SUPPORT_SWEEP_PROPENSITIES) * len(SUPPORT_SWEEP_Q_GAPS)
    assert sweep_df.groupby(["policy", "min_propensity", "q_gap_threshold"]).size().eq(1).all()
    for column in [
        "val_policy_value_dr",
        "test_policy_value_dr",
        "val_override_rate",
        "selected_for_main_paper",
    ]:
        assert column in sweep_df.columns
    assert "min_propensity" in best_thresholds
    assert "q_gap_threshold" in best_thresholds


def test_cluster_bootstrap_produces_valid_output() -> None:
    df = make_eval_frame()
    df["trajectory_id"] = [0, 0, 0, 1, 1, 1]
    metadata = make_metadata()
    models = make_models()
    outputs = get_policy_outputs(df, metadata, models, run_ablations=False)
    cache = prepare_policy_cache(df, metadata, models, outputs["causal_fqi"])
    trajectory_ids = df["trajectory_id"].to_numpy()
    rows = cluster_bootstrap_confidence_intervals(
        cache,
        df["reward_primary"].to_numpy(dtype=float),
        "primary",
        "causal_fqi",
        bootstrap_reps=20,
        trajectory_ids=trajectory_ids,
        scope="overall",
    )
    assert len(rows) == len(BOOTSTRAP_METRICS)
    for row in rows:
        assert row["ci_low"] <= row["ci_high"]
        assert row.get("bootstrap_method") == "cluster"
