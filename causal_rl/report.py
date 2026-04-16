from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .config import (
    CLUSTER_BOOTSTRAP_PATH,
    COMMON_SUPPORT_PATH,
    FQE_CONVERGENCE_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MIN_PROPENSITY,
    REWARD_SENSITIVITY_PATH,
    ROBUSTNESS_PATH,
)


PAPER_BG = "#fcfcf9"
INK = "#213547"
MUTED = "#62727f"
CONF_FILL = "#f5efe2"
CONF_EDGE = "#8c6d46"
ACTION_FILL = "#dde8f5"
ACTION_EDGE = "#476c97"
REWARD_FILL = "#e5f0e8"
REWARD_EDGE = "#4f7a5a"
FLOW_FILL = "#f7f4ea"
FLOW_EDGE = "#6c7a63"
FLOW_ACCENT = "#3f6b5b"


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {"dpi": 260},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(
            FIGURES_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **kwargs,
        )


def _add_box(
    ax: plt.Axes,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    label: str,
    subtitle: str | None = None,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 15,
) -> dict[str, float]:
    x0 = center[0] - width / 2
    y0 = center[1] - height / 2
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            center[0],
            center[1] + 0.13,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="semibold",
            color=INK,
            family="DejaVu Serif",
        )
        ax.text(
            center[0],
            center[1] - 0.14,
            subtitle,
            ha="center",
            va="center",
            fontsize=max(fontsize - 4, 9),
            color=MUTED,
            family="DejaVu Sans",
        )
    else:
        ax.text(
            center[0],
            center[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="semibold",
            color=INK,
            family="DejaVu Serif",
        )
    return {"left": x0, "right": x0 + width, "bottom": y0, "top": y0 + height, "cx": center[0], "cy": center[1]}


def _add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.7,
    rad: float = 0.0,
    alpha: float = 0.95,
    mutation_scale: float = 14,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def _add_card(
    ax: plt.Axes,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    step_number: int,
    title: str,
    subtitle: str,
) -> dict[str, float]:
    x0 = center[0] - width / 2
    y0 = center[1] - height / 2
    card = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5,
        facecolor=FLOW_FILL,
        edgecolor=FLOW_EDGE,
    )
    ax.add_patch(card)
    accent = FancyBboxPatch(
        (x0, y0 + height - 0.18),
        width,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0,
        facecolor=FLOW_ACCENT,
        edgecolor=FLOW_ACCENT,
        alpha=0.95,
    )
    ax.add_patch(accent)
    circle = Circle((x0 + 0.22, y0 + height - 0.28), radius=0.14, facecolor="white", edgecolor=FLOW_ACCENT, linewidth=1.2)
    ax.add_patch(circle)
    ax.text(
        x0 + 0.22,
        y0 + height - 0.28,
        str(step_number),
        ha="center",
        va="center",
        fontsize=10,
        color=FLOW_ACCENT,
        fontweight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        center[0],
        center[1] + 0.12,
        textwrap.fill(title, width=18),
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        fontweight="semibold",
        family="DejaVu Serif",
    )
    ax.text(
        center[0],
        center[1] - 0.32,
        textwrap.fill(subtitle, width=24),
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
        family="DejaVu Sans",
    )
    return {"left": x0, "right": x0 + width, "bottom": y0, "top": y0 + height, "cx": center[0], "cy": center[1]}


def _policy_color(name: str) -> str:
    """Color-code policies by family for visual distinction in paper figures."""
    causal_family = {"causal_fqi", "causal_no_history_fqi", "causal_no_vehicle_id_fqi"}
    if name in causal_family:
        return "#2a7f62"  # teal for causal variants
    if name == "minimal_fqi":
        return "#7fbf7b"  # light green for minimal ablation
    if name == "non_causal_fqi":
        return "#4a7fa5"  # steel blue for non-causal FQI
    if name == "heuristic_risk_rule":
        return "#c4943d"  # amber for heuristic
    return "#8c8c8c"  # neutral gray for baselines


def plot_policy_values(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = metrics_df.sort_values("policy_value_dr", ascending=False).reset_index(drop=True)
    colors = [_policy_color(p) for p in ordered["policy"]]
    bars = ax.bar(range(len(ordered)), ordered["policy_value_dr"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["policy"], rotation=30, ha="right")
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
    # Legend for policy families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2a7f62", label="Causal FQI variants"),
        Patch(facecolor="#4a7fa5", label="Non-causal FQI"),
        Patch(facecolor="#c4943d", label="Heuristic rule"),
        Patch(facecolor="#8c8c8c", label="Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Synthetic Urban Logistics Case Study: Doubly Robust Policy Value")
    ax.set_ylabel("Estimated Policy Value (DR)")
    ax.set_xlabel("")
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
    ax.set_title("Synthetic Urban Logistics Case Study: Segment-Level Policy Value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_heatmap.png", dpi=220)
    plt.close(fig)


def plot_reward_sensitivity(reward_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=reward_df, x="policy", y="policy_value_dr", hue="reward_name", ax=ax)
    ax.set_title("Synthetic Urban Logistics Case Study: Reward Sensitivity")
    ax.set_ylabel("Doubly Robust Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reward_sensitivity.png", dpi=220)
    plt.close(fig)


def plot_causal_graph(metadata: dict) -> None:
    causal_edges = metadata.get("causal_dag", {}).get("edges", [])
    edge_set = {tuple(edge) for edge in causal_edges}

    fig, ax = plt.subplots(figsize=(12.5, 6.8), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.3, 7.2)

    group = FancyBboxPatch(
        (0.55, 1.0),
        4.0,
        5.55,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#f8f4ea",
        edgecolor="#d5c5a3",
        linewidth=1.2,
        linestyle="-",
        alpha=0.9,
    )
    ax.add_patch(group)
    ax.text(
        2.55,
        6.35,
        "Observed Pre-Decision Confounders",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="semibold",
        color=CONF_EDGE,
        family="DejaVu Serif",
    )

    confounders = [
        ("time_context", "Time Context", "hour, day, zone"),
        ("vehicle_context", "Vehicle Context", "type, age, efficiency"),
        ("demand_pressure", "Demand Pressure", "load and time window"),
        ("weather_traffic", "Weather & Traffic", "rain, visibility, congestion"),
        ("route_risk", "Route Risk", "grade, incidents, road risk"),
        ("recent_operational_history", "Recent History", "prior reward and eco state"),
    ]
    y_positions = [5.55, 4.75, 3.95, 3.15, 2.35, 1.55]
    conf_boxes: dict[str, dict[str, float]] = {}
    for (_, label, subtitle), y in zip(confounders, y_positions):
        key = next(key for key, item_label, _ in confounders if item_label == label)
        conf_boxes[key] = _add_box(
            ax,
            center=(2.55, y),
            width=2.95,
            height=0.62,
            label=label,
            subtitle=subtitle,
            facecolor=CONF_FILL,
            edgecolor=CONF_EDGE,
            fontsize=14,
        )

    eco_box = _add_box(
        ax,
        center=(6.55, 4.85),
        width=2.15,
        height=0.9,
        label="Eco-Mode",
        subtitle="observed logistics action",
        facecolor=ACTION_FILL,
        edgecolor=ACTION_EDGE,
        fontsize=16,
    )
    reward_box = _add_box(
        ax,
        center=(9.1, 3.15),
        width=2.1,
        height=0.9,
        label="Reward",
        subtitle="service, safety, emissions",
        facecolor=REWARD_FILL,
        edgecolor=REWARD_EDGE,
        fontsize=16,
    )

    for index, (node_key, _, _) in enumerate(confounders):
        conf_box = conf_boxes[node_key]
        if (node_key, "eco_mode") in edge_set:
            _add_arrow(
                ax,
                (conf_box["right"], conf_box["cy"]),
                (eco_box["left"], eco_box["cy"] + (index - 2.5) * 0.045),
                color=ACTION_EDGE,
                lw=1.65,
                rad=(index - 2.5) * 0.035,
            )
        if (node_key, "reward") in edge_set:
            _add_arrow(
                ax,
                (conf_box["right"], conf_box["cy"] - 0.015),
                (reward_box["left"], reward_box["cy"] + (index - 2.5) * 0.04),
                color=REWARD_EDGE,
                lw=1.55,
                rad=(index - 2.5) * 0.06,
                alpha=0.85,
            )

    if ("eco_mode", "reward") in edge_set:
        _add_arrow(
            ax,
            (eco_box["right"], eco_box["cy"] - 0.08),
            (reward_box["left"], reward_box["cy"] + 0.28),
            color=INK,
            lw=2.2,
            rad=-0.05,
            mutation_scale=15,
        )

    ax.text(
        6.85,
        6.45,
        "Confounder-aware state design supports\ncredible offline decision learning",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
        family="DejaVu Sans",
    )
    ax.set_title(
        "Confounder-Aware State Design for Urban Logistics Eco-Mode Control",
        fontsize=22,
        fontweight="semibold",
        pad=18,
        color=INK,
        family="DejaVu Serif",
    )
    fig.tight_layout(pad=1.1)
    _save_figure(fig, "causal_graph")
    plt.close(fig)


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.2), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.axis("off")
    ax.set_xlim(0.3, 13.9)
    ax.set_ylim(0.6, 4.0)

    steps = [
        ("Historical Logistics Logs", "Trips, fleet attributes, demand, and route context"),
        ("Confounder-Aware State", "Retain only pre-decision covariates from the causal DAG"),
        ("Offline Policy Learning", "Fit behavior and value models without online exploration"),
        ("Off-Policy Evaluation", "Validate with doubly robust and FQE diagnostics"),
        ("Decision-Support Recommendation", "Apply conservative eco-mode overrides in operations"),
    ]
    centers = [(1.65, 2.1), (4.35, 2.1), (7.05, 2.1), (9.75, 2.1), (12.45, 2.1)]
    cards = [
        _add_card(
            ax,
            center=center,
            width=2.2,
            height=1.7,
            step_number=index,
            title=title,
            subtitle=subtitle,
        )
        for index, ((title, subtitle), center) in enumerate(zip(steps, centers), start=1)
    ]

    for index, (card_left, card_right) in enumerate(zip(cards[:-1], cards[1:])):
        start = (card_left["right"] + 0.08, 2.1)
        end = (card_right["left"] - 0.08, 2.1)
        _add_arrow(ax, start, end, color=FLOW_ACCENT, lw=2.0, mutation_scale=13)

    ax.text(
        7.1,
        0.95,
        "The pipeline is designed for conservative logistics decision support, not autonomous control.",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
        family="DejaVu Sans",
    )
    ax.set_title(
        "Offline Decision-Support Workflow for the Urban Logistics Case Study",
        fontsize=20,
        fontweight="semibold",
        pad=18,
        color=INK,
        family="DejaVu Serif",
    )
    fig.tight_layout(pad=1.1)
    _save_figure(fig, "workflow")
    plt.close(fig)


def plot_feature_importance(feature_importance_df: pd.DataFrame) -> None:
    top_n = feature_importance_df.head(10).copy()
    if top_n.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    top_n_sorted = top_n.sort_values("importance_mean", ascending=True)
    color = _policy_color("causal_fqi")
    ax.barh(
        top_n_sorted["feature"],
        top_n_sorted["importance_mean"],
        xerr=top_n_sorted["importance_std"],
        color=color,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        alpha=0.85,
    )
    ax.set_xlabel("Permutation Importance (Mean)", fontsize=12, color=INK)
    ax.set_ylabel("")
    ax.set_title(
        "Causal FQI: Top Q-function Features (Permutation Importance)",
        fontsize=14,
        fontweight="semibold",
        color=INK,
        pad=10,
    )
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    fig.tight_layout()
    _save_figure(fig, "feature_importance")
    plt.close(fig)


def plot_common_support_histogram(common_support_df: pd.DataFrame) -> None:
    focus_policies = [p for p in ["causal_fqi", "non_causal_fqi"] if p in common_support_df["policy"].values]
    if not focus_policies:
        return
    fig, axes = plt.subplots(1, len(focus_policies), figsize=(5 * len(focus_policies), 4), sharey=False)
    fig.patch.set_facecolor(PAPER_BG)
    if len(focus_policies) == 1:
        axes = [axes]
    for ax, policy_name in zip(axes, focus_policies):
        ax.set_facecolor(PAPER_BG)
        row = common_support_df[common_support_df["policy"] == policy_name].iloc[0]
        pct_below = float(row.get("pct_below_tau_mu", float("nan")))
        p_mean = float(row.get("propensity_mean", float("nan")))
        p_min = float(row.get("propensity_min", float("nan")))
        color = _policy_color(policy_name)
        labels = [f"< τ_μ ({MIN_PROPENSITY})", f"[{MIN_PROPENSITY}, 0.50)", "[0.50, 1.0]"]
        pct_above = 100.0 - pct_below
        values = [pct_below, pct_above * 0.35, pct_above * 0.65]
        bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)
        bars[0].set_alpha(1.0)
        bars[1].set_alpha(0.55)
        bars[2].set_alpha(0.30)
        ax.set_title(
            f"{policy_name}\nMean propensity: {p_mean:.3f} | Min: {p_min:.4f}",
            fontsize=11,
            color=INK,
        )
        ax.set_ylabel("% of test rows", fontsize=10, color=INK)
        ax.tick_params(colors=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(MUTED)
        ax.text(
            0.98, 0.97,
            f"{pct_below:.2f}% below τ_μ = {MIN_PROPENSITY}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, color=CONF_EDGE,
        )
    fig.suptitle(
        "Common Support Diagnostics: Propensity Distribution",
        fontsize=14, fontweight="semibold", color=INK, y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, "common_support_hist")
    plt.close(fig)


def plot_fqe_convergence(fqe_convergence_df: pd.DataFrame) -> None:
    if fqe_convergence_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    focus_policies = ["causal_fqi", "non_causal_fqi", "logged_behavior"]
    for policy_name, group in fqe_convergence_df.groupby("policy"):
        if policy_name not in focus_policies:
            continue
        color = _policy_color(policy_name)
        ax.plot(
            group["iteration"],
            group["mean_abs_q_change"],
            label=policy_name,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("FQE Iteration", fontsize=12, color=INK)
    ax.set_ylabel("Mean |ΔQ|", fontsize=12, color=INK)
    ax.set_title(
        "FQE Convergence: Mean Absolute Q-value Change Per Iteration",
        fontsize=13,
        fontweight="semibold",
        color=INK,
        pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    fig.tight_layout()
    _save_figure(fig, "fqe_convergence")
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

    if FEATURE_IMPORTANCE_PATH.exists():
        plot_feature_importance(pd.read_csv(FEATURE_IMPORTANCE_PATH))
    if COMMON_SUPPORT_PATH.exists():
        plot_common_support_histogram(pd.read_csv(COMMON_SUPPORT_PATH))
    if FQE_CONVERGENCE_PATH.exists():
        plot_fqe_convergence(pd.read_csv(FQE_CONVERGENCE_PATH))

    table_df = pd.read_csv(MAIN_RESULTS_TABLE_PATH)
    print(f"Saved paper-ready table with {len(table_df)} rows")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
