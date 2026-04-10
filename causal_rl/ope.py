from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import FQI_ITERATIONS, GAMMA, SEED
from .policy_learning import build_preprocessor


def self_normalized_ips(
    rewards: np.ndarray,
    logged_actions: np.ndarray,
    policy_actions: np.ndarray,
    logged_propensity: np.ndarray,
) -> float:
    weights = (policy_actions == logged_actions).astype(float) / np.clip(logged_propensity, 1e-6, 1.0)
    normalizer = np.sum(weights)
    if normalizer <= 0:
        return float("nan")
    return float(np.sum(weights * rewards) / normalizer)


def doubly_robust(
    rewards: np.ndarray,
    logged_actions: np.ndarray,
    policy_actions: np.ndarray,
    logged_propensity: np.ndarray,
    q_logged: np.ndarray,
    q_policy: np.ndarray,
) -> float:
    correction = (policy_actions == logged_actions).astype(float) / np.clip(logged_propensity, 1e-6, 1.0)
    dr = q_policy + correction * (rewards - q_logged)
    return float(np.mean(dr))


class FittedQEvaluator:
    def __init__(self, state_columns: list[str], gamma: float = GAMMA, iterations: int = FQI_ITERATIONS) -> None:
        self.state_columns = state_columns
        self.gamma = gamma
        self.iterations = iterations
        self.preprocessor: ColumnTransformer | None = None
        self.models: dict[int, object] = {}

    def fit(self, df: pd.DataFrame, policy_fn, reward_column: str = "reward") -> "FittedQEvaluator":
        self.preprocessor = build_preprocessor(df, self.state_columns)
        X_state = self.preprocessor.fit_transform(df[self.state_columns])
        next_state_columns = [f"next_state_{col}" for col in self.state_columns]
        next_df = df[next_state_columns].rename(columns={f"next_state_{c}": c for c in self.state_columns})
        X_next = self.preprocessor.transform(next_df)
        actions = df["action"].to_numpy(dtype=int)
        rewards = df[reward_column].to_numpy(dtype=float)
        done = df["done"].to_numpy(dtype=int)
        policy_next = policy_fn(next_df)
        q_next = np.zeros(len(df), dtype=float)

        for iteration in range(self.iterations):
            targets = rewards + self.gamma * (1.0 - done) * q_next
            for action in (0, 1):
                mask = actions == action
                model = HistGradientBoostingRegressor(
                    max_depth=6,
                    max_iter=120,
                    learning_rate=0.05,
                    random_state=SEED + 1000 + iteration * 7 + action,
                )
                model.fit(X_state[mask], targets[mask])
                self.models[action] = model
            q_next = np.column_stack([self.models[a].predict(X_next) for a in (0, 1)])[np.arange(len(df)), policy_next]
        return self

    def evaluate_policy_value(self, df: pd.DataFrame, policy_actions: np.ndarray) -> float:
        if self.preprocessor is None:
            raise RuntimeError("Call fit before evaluate_policy_value.")
        X = self.preprocessor.transform(df[self.state_columns])
        q_values = np.column_stack([self.models[a].predict(X) for a in (0, 1)])
        return float(np.mean(q_values[np.arange(len(df)), policy_actions]))
