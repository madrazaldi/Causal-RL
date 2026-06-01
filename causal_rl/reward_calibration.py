from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REWARD_CALIBRATION_PATH, REWARD_SPECS


REWARD_COMPONENT_METADATA = {
    "lateness_min": {
        "role": "service",
        "unit": "minutes",
        "interpretation": "one minute of lateness",
    },
    "crash": {
        "role": "safety",
        "unit": "indicator",
        "interpretation": "one crash event",
    },
    "near_miss": {
        "role": "safety",
        "unit": "indicator",
        "interpretation": "one near-miss event",
    },
    "co2_kg": {
        "role": "sustainability",
        "unit": "kg",
        "interpretation": "one kilogram of emitted CO2",
    },
}

CALIBRATION_BASIS_SPLIT = "train"


def _summary_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "overall":
        return df
    if "split" not in df.columns:
        return df.iloc[0:0]
    return df.loc[df["split"] == split]


def build_reward_calibration_table(
    df: pd.DataFrame,
    *,
    basis_split: str = CALIBRATION_BASIS_SPLIT,
) -> pd.DataFrame:
    """Summarize reward weights as auditable utility tradeoffs.

    The table does not estimate new weights. It records the configured
    lateness-minute-equivalent tradeoffs and the observed component scale on
    train/validation/test partitions so the reward design can be inspected
    without tuning on the held-out test set.
    """

    required_columns = set(REWARD_COMPONENT_METADATA)
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing reward component columns: {missing_columns}")

    if "split" in df.columns:
        split_order = [basis_split, "val", "test", "overall"]
    else:
        split_order = ["overall"]

    rows: list[dict[str, float | int | str]] = []
    for split in dict.fromkeys(split_order):
        split_df = _summary_frame(df, split)
        if split_df.empty:
            continue

        for reward_name, weights in REWARD_SPECS.items():
            lateness_weight = float(weights["lateness_min"])
            total_configured_weight = sum(float(weight) for weight in weights.values())
            weighted_means = {
                component: float(weight) * float(split_df[component].mean())
                for component, weight in weights.items()
            }
            total_weighted_mean = sum(weighted_means.values())

            for component, weight in weights.items():
                metadata = REWARD_COMPONENT_METADATA[component]
                raw = split_df[component].astype(float)
                weighted_mean = weighted_means[component]
                rows.append(
                    {
                        "reward_name": reward_name,
                        "split": split,
                        "calibration_basis": int(split == basis_split),
                        "component": component,
                        "component_role": metadata["role"],
                        "component_unit": metadata["unit"],
                        "component_interpretation": metadata["interpretation"],
                        "weight": float(weight),
                        "configured_weight_share_pct": (
                            100.0 * float(weight) / total_configured_weight if total_configured_weight else 0.0
                        ),
                        "lateness_min_equivalent": float(weight) / lateness_weight,
                        "raw_mean": float(raw.mean()),
                        "raw_p50": float(raw.quantile(0.50)),
                        "raw_p95": float(raw.quantile(0.95)),
                        "raw_max": float(raw.max()),
                        "weighted_mean_penalty": weighted_mean,
                        "weighted_penalty_share_pct": (
                            100.0 * weighted_mean / total_weighted_mean if total_weighted_mean else 0.0
                        ),
                        "mean_total_penalty": total_weighted_mean,
                        "n_rows": int(len(split_df)),
                    }
                )

    return pd.DataFrame(rows)


def write_reward_calibration_table(
    df: pd.DataFrame,
    path: Path = REWARD_CALIBRATION_PATH,
    *,
    basis_split: str = CALIBRATION_BASIS_SPLIT,
) -> pd.DataFrame:
    table = build_reward_calibration_table(df, basis_split=basis_split)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
