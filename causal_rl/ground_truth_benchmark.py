from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import GAMMA, GROUND_TRUTH_BENCHMARK_PATH, RESULTS_DIR, SEED
from .policy_learning import FittedQPolicy


STATE_COLUMNS = [
    "t",
    "remaining_steps",
    "traffic",
    "weather",
    "urgency",
    "delay",
    "prior_action",
    "rolling_reward",
]
OMITTED_CONFOUNDER_COLUMNS = [
    "t",
    "remaining_steps",
    "traffic",
    "prior_action",
]


StateDict = dict[str, np.ndarray]
PolicyFn = Callable[[pd.DataFrame], np.ndarray]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def initial_state(n_trajectories: int, horizon: int, rng: np.random.Generator) -> StateDict:
    return {
        "t": np.zeros(n_trajectories, dtype=float),
        "remaining_steps": np.full(n_trajectories, horizon - 1, dtype=float),
        "traffic": rng.beta(2.5, 2.2, size=n_trajectories),
        "weather": rng.beta(1.8, 4.2, size=n_trajectories),
        "urgency": rng.beta(2.0, 2.8, size=n_trajectories),
        "delay": rng.gamma(shape=1.3, scale=0.25, size=n_trajectories),
        "prior_action": np.zeros(n_trajectories, dtype=float),
        "rolling_reward": np.zeros(n_trajectories, dtype=float),
    }


def behavior_actions(state: StateDict, rng: np.random.Generator) -> np.ndarray:
    """Confounded logging policy: eco mode is more likely in easy states."""
    logits = (
        1.10
        - 2.10 * state["urgency"]
        - 1.35 * state["delay"]
        - 0.85 * state["traffic"]
        + 0.65 * (1.0 - state["weather"])
        + 0.30 * state["prior_action"]
    )
    propensity = np.clip(sigmoid(logits), 0.04, 0.96)
    return (rng.random(len(propensity)) < propensity).astype(int)


def deterministic_behavior_actions(state_frame: pd.DataFrame) -> np.ndarray:
    logits = (
        1.10
        - 2.10 * state_frame["urgency"].to_numpy(dtype=float)
        - 1.35 * state_frame["delay"].to_numpy(dtype=float)
        - 0.85 * state_frame["traffic"].to_numpy(dtype=float)
        + 0.65 * (1.0 - state_frame["weather"].to_numpy(dtype=float))
        + 0.30 * state_frame["prior_action"].to_numpy(dtype=float)
    )
    return (sigmoid(logits) >= 0.5).astype(int)


def immediate_reward_mean(state_frame: pd.DataFrame, action: np.ndarray) -> np.ndarray:
    traffic = state_frame["traffic"].to_numpy(dtype=float)
    weather = state_frame["weather"].to_numpy(dtype=float)
    urgency = state_frame["urgency"].to_numpy(dtype=float)
    delay = state_frame["delay"].to_numpy(dtype=float)
    action_float = action.astype(float)

    emissions = (
        3.20
        + 1.15 * traffic
        + 0.35 * delay
        - 1.00 * action_float
        + 0.35 * action_float * traffic
    )
    lateness = (
        0.45 * delay
        + 1.15 * urgency
        + 0.75 * traffic
        + 0.35 * weather
        + 1.10 * action_float * urgency
        + 0.55 * action_float * traffic
        - 0.95 * action_float * (1.0 - urgency) * (1.0 - traffic)
    )
    incident_risk = (
        0.04
        + 0.14 * traffic
        + 0.08 * weather
        + 0.04 * delay
        - 0.03 * action_float
    )
    return -(lateness + 0.20 * emissions + 2.00 * incident_risk)


def myopic_structural_actions(state_frame: pd.DataFrame) -> np.ndarray:
    reward_0 = immediate_reward_mean(state_frame, np.zeros(len(state_frame), dtype=int))
    reward_1 = immediate_reward_mean(state_frame, np.ones(len(state_frame), dtype=int))
    return (reward_1 > reward_0).astype(int)


def transition(
    state: StateDict,
    action: np.ndarray,
    rng: np.random.Generator,
    horizon: int,
) -> tuple[np.ndarray, StateDict]:
    state_frame = pd.DataFrame(state)
    mean_reward = immediate_reward_mean(state_frame, action)
    reward_noise = rng.normal(0.0, 0.08, size=len(action))
    reward = mean_reward + reward_noise

    action_float = action.astype(float)
    lateness_pressure = -mean_reward - 0.65
    next_delay = np.clip(
        0.45 * state["delay"]
        + 0.36 * np.maximum(lateness_pressure, 0.0)
        + 0.10 * action_float * state["urgency"]
        - 0.42 * action_float * (1.0 - state["urgency"]) * (1.0 - state["traffic"])
        + rng.normal(0.0, 0.05, size=len(action)),
        0.0,
        4.0,
    )
    next_traffic = np.clip(
        0.62 * state["traffic"]
        + 0.18 * state["weather"]
        + 0.14 * state["urgency"]
        - 0.08 * action_float * (1.0 - state["weather"])
        + rng.normal(0.0, 0.08, size=len(action)),
        0.0,
        1.0,
    )
    next_weather = np.clip(
        0.70 * state["weather"] + rng.beta(1.5, 5.0, size=len(action)) * 0.30,
        0.0,
        1.0,
    )
    next_urgency = np.clip(
        0.58 * state["urgency"]
        + 0.24 * next_delay
        + 0.10 * next_traffic
        - 0.10 * action_float * (1.0 - state["urgency"])
        + rng.normal(0.0, 0.06, size=len(action)),
        0.0,
        1.0,
    )
    next_t = state["t"] + 1.0
    next_state = {
        "t": next_t,
        "remaining_steps": np.maximum(horizon - next_t - 1.0, 0.0),
        "traffic": next_traffic,
        "weather": next_weather,
        "urgency": next_urgency,
        "delay": next_delay,
        "prior_action": action.astype(float),
        "rolling_reward": (state["rolling_reward"] * state["t"] + reward) / np.maximum(next_t, 1.0),
    }
    return reward, next_state


def simulate_logged_data(
    n_trajectories: int = 3000,
    horizon: int = 6,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = initial_state(n_trajectories, horizon, rng)
    rows = []
    for step in range(horizon):
        action = behavior_actions(state, rng)
        reward, next_state = transition(state, action, rng, horizon)
        done = np.full(n_trajectories, int(step == horizon - 1), dtype=int)
        for idx in range(n_trajectories):
            row = {
                "trajectory_id": idx,
                "t": step,
                "action": int(action[idx]),
                "done": int(done[idx]),
                "reward": float(reward[idx]),
                "reward_primary": float(reward[idx]),
            }
            for column in STATE_COLUMNS:
                row[column] = float(state[column][idx])
                row[f"next_state_{column}"] = float(next_state[column][idx])
            rows.append(row)
        state = next_state
    return pd.DataFrame(rows)


def evaluate_interventional_policy(
    policy_fn: PolicyFn,
    n_trajectories: int = 3000,
    horizon: int = 6,
    seed: int = SEED + 100,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    state = initial_state(n_trajectories, horizon, rng)
    returns = np.zeros(n_trajectories, dtype=float)
    eco_rates = []
    for step in range(horizon):
        state_frame = pd.DataFrame(state)
        action = policy_fn(state_frame).astype(int)
        eco_rates.append(float(np.mean(action)))
        reward, state = transition(state, action, rng, horizon)
        returns += (GAMMA**step) * reward
    return float(np.mean(returns)), float(np.std(returns, ddof=1) / np.sqrt(n_trajectories)), float(np.mean(eco_rates))


def run_ground_truth_benchmark(
    n_train_trajectories: int = 3000,
    n_eval_trajectories: int = 3000,
    horizon: int = 6,
    fqi_iterations: int = 4,
    seed: int = SEED,
) -> pd.DataFrame:
    train_df = simulate_logged_data(n_train_trajectories, horizon, seed)
    causal_fqi = FittedQPolicy(STATE_COLUMNS, iterations=fqi_iterations, name="ground_truth_causal_fqi")
    causal_fqi.fit(train_df)
    omitted_fqi = FittedQPolicy(
        OMITTED_CONFOUNDER_COLUMNS,
        iterations=fqi_iterations,
        name="ground_truth_omitted_confounder_fqi",
    )
    omitted_fqi.fit(train_df)

    policy_functions: dict[str, PolicyFn] = {
        "causal_fqi": lambda frame: causal_fqi.greedy_action(frame),
        "omitted_confounder_fqi": lambda frame: omitted_fqi.greedy_action(frame),
        "myopic_structural_rule": myopic_structural_actions,
        "logged_behavior_rule": deterministic_behavior_actions,
        "always_eco": lambda frame: np.ones(len(frame), dtype=int),
        "never_eco": lambda frame: np.zeros(len(frame), dtype=int),
    }
    rows = []
    for offset, (policy_name, policy_fn) in enumerate(policy_functions.items()):
        mean_value, se_value, eco_rate = evaluate_interventional_policy(
            policy_fn,
            n_trajectories=n_eval_trajectories,
            horizon=horizon,
            seed=seed + 1000 + offset,
        )
        rows.append(
            {
                "policy": policy_name,
                "true_discounted_return": mean_value,
                "true_discounted_return_se": se_value,
                "eco_rate": eco_rate,
                "n_train_trajectories": n_train_trajectories,
                "n_eval_trajectories": n_eval_trajectories,
                "horizon": horizon,
                "fqi_iterations": fqi_iterations,
            }
        )

    results = pd.DataFrame(rows).sort_values("true_discounted_return", ascending=False).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)
    top_policy = str(results.iloc[0]["policy"])
    causal_row = results.loc[results["policy"] == "causal_fqi"].iloc[0]
    omitted_row = results.loc[results["policy"] == "omitted_confounder_fqi"].iloc[0]
    results["causal_fqi_top"] = int(top_policy == "causal_fqi")
    results["causal_fqi_beats_omitted_confounder_fqi"] = int(
        causal_row["true_discounted_return"] > omitted_row["true_discounted_return"]
    )
    results["benchmark_claim"] = (
        "controlled_dgp_supports_causal_fqi"
        if top_policy == "causal_fqi"
        else "controlled_dgp_does_not_make_causal_fqi_top"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled ground-truth policy benchmark.")
    parser.add_argument("--n-train-trajectories", type=int, default=3000)
    parser.add_argument("--n-eval-trajectories", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--fqi-iterations", type=int, default=4)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_ground_truth_benchmark(
        n_train_trajectories=args.n_train_trajectories,
        n_eval_trajectories=args.n_eval_trajectories,
        horizon=args.horizon,
        fqi_iterations=args.fqi_iterations,
    )
    results.to_csv(GROUND_TRUTH_BENCHMARK_PATH, index=False)
    print(f"Saved controlled ground-truth benchmark to {GROUND_TRUTH_BENCHMARK_PATH}")
    print(results[["policy", "true_discounted_return", "rank", "benchmark_claim"]].to_string(index=False))


if __name__ == "__main__":
    main()
