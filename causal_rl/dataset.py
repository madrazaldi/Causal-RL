from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTION_COLUMN,
    ARTIFACTS_DIR,
    CAUSAL_BACKDOOR_COLUMNS,
    DATE_COLUMN,
    DECISION_LOG_PATH,
    DEPLOYABLE_STATE_COLUMNS,
    LATENT_COLUMNS,
    METADATA_PATH,
    NON_CAUSAL_EXTRA_COLUMNS,
    POST_ACTION_COLUMNS,
    RAW_DATA_PATH,
    REWARD_COLUMNS,
    REWARD_SPECS,
    SEED,
    SORT_COLUMNS,
    SPLIT_RATIOS,
    TRAJECTORY_KEY_COLUMNS,
)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    metadata: dict


def ensure_directories() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def unique_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = np.arange(len(df))
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def compute_reward_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for reward_name, weights in REWARD_SPECS.items():
        reward = np.zeros(len(out), dtype=float)
        for field, weight in weights.items():
            reward -= weight * out[field].to_numpy(dtype=float)
        out[REWARD_COLUMNS[reward_name]] = reward
    return out


def _shifted_expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding().mean().fillna(0.0)


def _shifted_cumulative(series: pd.Series) -> pd.Series:
    return series.shift(1).fillna(0.0).cumsum()


def add_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(SORT_COLUMNS).copy()
    group = out.groupby(TRAJECTORY_KEY_COLUMNS, sort=False)
    out["trajectory_id"] = group.ngroup()
    out["trajectory_length"] = group["row_id"].transform("size")
    out["t"] = group.cumcount()
    out["step_idx"] = out["t"]
    out["remaining_steps"] = out["trajectory_length"] - out["t"] - 1
    out["rolling_mean_traffic"] = group["traffic_index"].transform(_shifted_expanding_mean)
    out["rolling_cumulative_lateness"] = group["lateness_min"].transform(_shifted_cumulative)
    prior_incidents = out["near_miss"].astype(float) + out["crash"].astype(float)
    out["rolling_incident_count"] = prior_incidents.groupby(
        [out[c] for c in TRAJECTORY_KEY_COLUMNS], sort=False
    ).transform(_shifted_cumulative)
    out["prior_reward_primary"] = group["reward_primary"].shift(1).fillna(0.0)
    out["prior_eco_mode"] = group[ACTION_COLUMN].shift(1).fillna(0).astype(int)
    out["done"] = (out["remaining_steps"] == 0).astype(int)
    return out


def assign_temporal_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    unique_dates = np.array(sorted(out[DATE_COLUMN].dt.strftime("%Y-%m-%d").unique()))
    n_dates = len(unique_dates)
    train_end = max(1, int(n_dates * SPLIT_RATIOS[0]))
    val_end = max(train_end + 1, int(n_dates * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])))
    val_end = min(val_end, n_dates - 1)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    split_map = {date: "train" for date in train_dates}
    split_map.update({date: "val" for date in val_dates})
    split_map.update({date: "test" for date in test_dates})

    out["split"] = out[DATE_COLUMN].dt.strftime("%Y-%m-%d").map(split_map)
    metadata = {
        "split_counts": out["split"].value_counts().to_dict(),
        "train_dates": train_dates.tolist(),
        "val_dates": val_dates.tolist(),
        "test_dates": test_dates.tolist(),
    }
    return out, metadata


def create_next_state_columns(df: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby(TRAJECTORY_KEY_COLUMNS, sort=False)
    for column in state_columns:
        next_values = group[column].shift(-1)
        if pd.api.types.is_numeric_dtype(out[column]):
            fill_value = 0.0
            if pd.api.types.is_integer_dtype(out[column]):
                fill_value = 0
            out[f"next_state_{column}"] = next_values.fillna(fill_value)
        else:
            out[f"next_state_{column}"] = next_values.fillna("__terminal__")
        out[f"state_{column}"] = out[column]
    return out


def build_decision_log(raw_df: pd.DataFrame) -> DatasetBundle:
    df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = compute_reward_columns(df)
    df = add_sequential_features(df)
    df, split_metadata = assign_temporal_splits(df)

    state_columns = DEPLOYABLE_STATE_COLUMNS
    df = create_next_state_columns(df, state_columns)
    df["action"] = df[ACTION_COLUMN].astype(int)
    df["reward"] = df[REWARD_COLUMNS["primary"]]

    leaked_state_columns = [c for c in POST_ACTION_COLUMNS + LATENT_COLUMNS if f"state_{c}" in df.columns]
    if leaked_state_columns:
        raise ValueError(f"Leakage detected in decision log: {leaked_state_columns}")

    metadata = {
        "rows": int(len(df)),
        "trajectory_count": int(df["trajectory_id"].nunique()),
        "unique_dates": int(df[DATE_COLUMN].nunique()),
        "state_columns": unique_preserve_order(state_columns),
        "causal_state_columns": unique_preserve_order(CAUSAL_BACKDOOR_COLUMNS),
        "non_causal_state_columns": unique_preserve_order(state_columns + NON_CAUSAL_EXTRA_COLUMNS),
        "reward_columns": REWARD_COLUMNS,
        "action_column": "action",
        "deployable_exclusions": POST_ACTION_COLUMNS + LATENT_COLUMNS,
        "post_action_columns": POST_ACTION_COLUMNS,
        "latent_columns": LATENT_COLUMNS,
        "split_metadata": split_metadata,
        "causal_dag": {
            "confounders": [
                "time_context",
                "vehicle_context",
                "demand_pressure",
                "weather_traffic",
                "route_risk",
                "recent_operational_history",
            ],
            "edges": [
                ["time_context", "eco_mode"],
                ["vehicle_context", "eco_mode"],
                ["demand_pressure", "eco_mode"],
                ["weather_traffic", "eco_mode"],
                ["route_risk", "eco_mode"],
                ["recent_operational_history", "eco_mode"],
                ["time_context", "reward"],
                ["vehicle_context", "reward"],
                ["demand_pressure", "reward"],
                ["weather_traffic", "reward"],
                ["route_risk", "reward"],
                ["recent_operational_history", "reward"],
                ["eco_mode", "reward"],
            ],
        },
        "random_seed": SEED,
    }
    return DatasetBundle(df=df, metadata=metadata)


def persist_bundle(bundle: DatasetBundle) -> None:
    ensure_directories()
    bundle.df.to_csv(DECISION_LOG_PATH, index=False)
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(bundle.metadata, fh, indent=2)


def build_and_save_dataset(raw_path: Path = RAW_DATA_PATH) -> DatasetBundle:
    bundle = build_decision_log(load_raw_data(raw_path))
    persist_bundle(bundle)
    return bundle
