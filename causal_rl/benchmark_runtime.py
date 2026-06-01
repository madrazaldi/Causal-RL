from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RUNTIME_BENCHMARK_PATH
from .evaluate import (
    ABLATION_POLICY_NAMES,
    behavior_model_for_policy,
    heuristic_actions,
    heuristic_risk_threshold,
    load_inputs,
    nuisance_state_columns,
)


DEFAULT_POLICY_NAMES = [
    "logged_behavior",
    "always_eco",
    "never_eco",
    "heuristic_risk_rule",
    "non_causal_fqi",
    "causal_fqi",
    *ABLATION_POLICY_NAMES,
]


def _policy_actions(policy_name: str, df: pd.DataFrame, metadata: dict, models: dict) -> np.ndarray:
    if policy_name == "logged_behavior":
        return df["action"].to_numpy(dtype=int)
    if policy_name == "always_eco":
        return np.ones(len(df), dtype=int)
    if policy_name == "never_eco":
        return np.zeros(len(df), dtype=int)
    if policy_name == "heuristic_risk_rule":
        threshold = heuristic_risk_threshold(metadata)
        return heuristic_actions(models["heuristic"], df, metadata["causal_state_columns"], threshold)

    behavior_state_cols = nuisance_state_columns(policy_name, metadata)
    output = models[policy_name].policy_action(
        df,
        behavior_model_for_policy(policy_name, models),
        behavior_state_cols,
    )
    return output["policy_action"].to_numpy(dtype=int)


def benchmark_saved_model_inference(
    repeats: int = 7,
    policy_names: list[str] | None = None,
) -> pd.DataFrame:
    df, metadata, models = load_inputs()
    test_df = df[df["split"] == "test"].copy()
    selected_policies = policy_names or DEFAULT_POLICY_NAMES
    rows: list[dict] = []

    for policy_name in selected_policies:
        if policy_name not in {"logged_behavior", "always_eco", "never_eco", "heuristic_risk_rule"} and policy_name not in models:
            continue
        elapsed: list[float] = []
        actions = None
        for _ in range(repeats):
            start = time.perf_counter()
            actions = _policy_actions(policy_name, test_df, metadata, models)
            elapsed.append(time.perf_counter() - start)

        elapsed_array = np.asarray(elapsed, dtype=float)
        mean_seconds = float(np.mean(elapsed_array))
        best_seconds = float(np.min(elapsed_array))
        rows.append(
            {
                "policy": policy_name,
                "benchmark_scope": "test_saved_model_policy_action",
                "n_rows": len(test_df),
                "repeats": repeats,
                "mean_seconds": mean_seconds,
                "best_seconds": best_seconds,
                "std_seconds": float(np.std(elapsed_array, ddof=0)),
                "rows_per_second_mean": float(len(test_df) / mean_seconds) if mean_seconds > 0 else np.nan,
                "rows_per_second_best": float(len(test_df) / best_seconds) if best_seconds > 0 else np.nan,
                "eco_rate": float(np.mean(actions)) if actions is not None else np.nan,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "note": "Saved-model inference benchmark only; full training runtime is intentionally not part of the headline policy-value claim.",
            }
        )

    benchmark = pd.DataFrame(rows)
    if benchmark.empty:
        return benchmark
    return benchmark.sort_values("mean_seconds", ascending=True).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved-model policy inference runtime.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default=str(RUNTIME_BENCHMARK_PATH))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    benchmark = benchmark_saved_model_inference(repeats=args.repeats)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_path, index=False)
    print(f"Saved runtime benchmark to {output_path} with {len(benchmark)} rows")


if __name__ == "__main__":
    main()
