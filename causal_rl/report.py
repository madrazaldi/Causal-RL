from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, MAIN_RESULTS_TABLE_PATH, METADATA_PATH, METRICS_PATH, REWARD_SENSITIVITY_PATH, ROBUSTNESS_PATH


def plot_policy_values(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = metrics_df.sort_values("policy_value_dr", ascending=False)
    sns.barplot(data=ordered, x="policy", y="policy_value_dr", ax=ax, color="#3b7a57")
    if {"policy_value_dr_ci_low", "policy_value_dr_ci_high"}.issubset(ordered.columns):
        errors = np.vstack(
            [
                ordered["policy_value_dr"] - ordered["policy_value_dr_ci_low"],
                ordered["policy_value_dr_ci_high"] - ordered["policy_value_dr"],
            ]
        )
        ax.errorbar(
            x=range(len(ordered)),
            y=ordered["policy_value_dr"],
            yerr=errors,
            fmt="none",
            ecolor="#1f3b2c",
            elinewidth=1.3,
            capsize=4,
        )
    ax.set_title("Doubly Robust Policy Value by Policy")
    ax.set_ylabel("Estimated Policy Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "policy_values.png", dpi=220)
    plt.close(fig)


def plot_robustness(robustness_df: pd.DataFrame) -> None:
    if robustness_df.empty:
        return
    pivot = robustness_df.pivot_table(
        index="policy",
        columns="segment",
        values="policy_value_dr",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Robustness Across Operational Scenarios")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_heatmap.png", dpi=220)
    plt.close(fig)


def plot_reward_sensitivity(reward_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=reward_df, x="policy", y="policy_value_dr", hue="reward_name", ax=ax)
    ax.set_title("Reward Sensitivity of Policy Value")
    ax.set_ylabel("Doubly Robust Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reward_sensitivity.png", dpi=220)
    plt.close(fig)


def plot_causal_graph(metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    nodes = ["Time", "Vehicle", "Demand", "Weather/Traffic", "Route Risk", "History", "Eco Mode", "Reward"]
    positions = {
        "Time": (0.1, 0.75),
        "Vehicle": (0.1, 0.55),
        "Demand": (0.1, 0.35),
        "Weather/Traffic": (0.4, 0.75),
        "Route Risk": (0.4, 0.45),
        "History": (0.4, 0.15),
        "Eco Mode": (0.7, 0.5),
        "Reward": (0.9, 0.5),
    }
    for node in nodes:
        x, y = positions[node]
        ax.text(x, y, node, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#f3efe0", "ec": "#355070"})
    arrows = [
        ("Time", "Eco Mode"),
        ("Vehicle", "Eco Mode"),
        ("Demand", "Eco Mode"),
        ("Weather/Traffic", "Eco Mode"),
        ("Route Risk", "Eco Mode"),
        ("History", "Eco Mode"),
        ("Time", "Reward"),
        ("Vehicle", "Reward"),
        ("Demand", "Reward"),
        ("Weather/Traffic", "Reward"),
        ("Route Risk", "Reward"),
        ("History", "Reward"),
        ("Eco Mode", "Reward"),
    ]
    for src, dst in arrows:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#355070"})
    ax.set_title("Domain Causal Graph for Eco-Mode Control")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "causal_graph.png", dpi=220)
    plt.close(fig)


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    steps = [
        "Historical logs",
        "Confounder-aware states",
        "Offline policy learning",
        "Off-policy evaluation",
        "Eco-mode recommendation",
    ]
    x_positions = [0.08, 0.30, 0.52, 0.74, 0.92]
    for step, x in zip(steps, x_positions):
        ax.text(x, 0.5, step, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#e9f5db", "ec": "#3a5a40"})
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(x1 - 0.06, 0.5), xytext=(x0 + 0.08, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#3a5a40"})
    ax.set_title("Proposed Eco-Mode Offline RL Workflow")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "workflow.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.read_csv(METRICS_PATH)
    robustness_df = pd.read_csv(ROBUSTNESS_PATH)
    reward_df = pd.read_csv(REWARD_SENSITIVITY_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    plot_policy_values(metrics_df)
    plot_robustness(robustness_df)
    plot_reward_sensitivity(reward_df)
    plot_causal_graph(metadata)
    plot_workflow()

    table_df = pd.read_csv(MAIN_RESULTS_TABLE_PATH)
    print(f"Saved paper-ready table with {len(table_df)} rows")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
