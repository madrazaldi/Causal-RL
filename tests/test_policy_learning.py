from __future__ import annotations

import numpy as np
import pandas as pd

from causal_rl.policy_learning import FittedQPolicy


class DummyBehaviorModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full((len(X), 2), 0.5)
        probs[:, 1] = 0.02
        probs[:, 0] = 0.98
        return probs


def make_policy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for idx in range(40):
        hour = 8 + (idx % 6)
        traffic = float(idx % 5) / 4.0
        vehicle_type = "cooling" if idx % 3 == 0 else "common"
        action = int(idx % 2 == 0)
        reward = -(traffic + 0.5 * action)
        rows.append(
            {
                "hour": hour,
                "traffic_index": traffic,
                "vehicle_type": vehicle_type,
                "action": action,
                "reward_primary": reward,
                "done": int(idx % 4 == 3),
                "next_state_hour": hour + 1,
                "next_state_traffic_index": min(1.0, traffic + 0.1),
                "next_state_vehicle_type": vehicle_type,
            }
        )
    return pd.DataFrame(rows)


def test_fitted_q_policy_returns_binary_actions() -> None:
    df = make_policy_frame()
    policy = FittedQPolicy(["hour", "traffic_index", "vehicle_type"], iterations=2).fit(df)
    actions = policy.greedy_action(df)
    assert set(np.unique(actions)).issubset({0, 1})


def test_support_constraint_falls_back_to_logged_action() -> None:
    df = make_policy_frame()
    policy = FittedQPolicy(["hour", "traffic_index", "vehicle_type"], iterations=2).fit(df)
    result = policy.policy_action(
        df,
        DummyBehaviorModel(),
        behavior_state_columns=["hour", "traffic_index", "vehicle_type"],
        fallback_to_logged=True,
        min_propensity=0.1,
        q_gap_threshold=0.0,
    )
    assert "used_fallback" in result.columns
    assert (result.loc[result["used_fallback"] == 1, "policy_action"].to_numpy() == df.loc[result["used_fallback"] == 1, "action"].to_numpy()).all()
