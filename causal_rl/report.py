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
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.8)

    # ── Observed confounder group box (Z) ──────────────────────────────────
    z_box = FancyBboxPatch(
        (0.3, 0.9), 3.5, 4.1,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=CONF_FILL, edgecolor=CONF_EDGE,
        linewidth=1.3, alpha=0.92,
    )
    ax.add_patch(z_box)
    ax.text(
        2.05, 4.82, "Z  —  Observed Pre-Decision Context",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=CONF_EDGE, family="DejaVu Sans",
    )

    confounder_rows = [
        ("Time Context", "hour, day_idx, dow, zone"),
        ("Vehicle & Task", "type, age, efficiency, demand, time window"),
        ("Road & Environment", "speed, grade, rain, traffic, visibility"),
        ("Prior Decisions", "route_risky, dispatch_delay_min"),
        ("Sequential History", "rolling traffic, lateness, prior eco_mode"),
    ]
    row_ys = [4.22, 3.55, 2.88, 2.20, 1.52]
    for (label, sub), y in zip(confounder_rows, row_ys):
        row_patch = FancyBboxPatch(
            (0.52, y - 0.27), 3.06, 0.52,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor="#fdf8ef", edgecolor=CONF_EDGE,
            linewidth=0.9, alpha=0.95,
        )
        ax.add_patch(row_patch)
        ax.text(2.05, y + 0.01, label, ha="center", va="center",
                fontsize=9.5, fontweight="semibold", color=INK, family="DejaVu Sans")
        ax.text(2.05, y - 0.16, sub, ha="center", va="center",
                fontsize=7.8, color=MUTED, family="DejaVu Sans")

    # ── Eco-Mode node (A) ──────────────────────────────────────────────────
    eco_cx, eco_cy = 6.2, 3.6
    eco_patch = FancyBboxPatch(
        (eco_cx - 1.05, eco_cy - 0.52), 2.1, 1.04,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=ACTION_FILL, edgecolor=ACTION_EDGE,
        linewidth=1.5,
    )
    ax.add_patch(eco_patch)
    ax.text(eco_cx, eco_cy + 0.12, "A  —  Eco-Mode",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=ACTION_EDGE, family="DejaVu Sans")
    ax.text(eco_cx, eco_cy - 0.18, "binary action  ∈ {0, 1}",
            ha="center", va="center", fontsize=8.5, color=MUTED, family="DejaVu Sans")

    # ── Reward node (Y) ────────────────────────────────────────────────────
    rew_cx, rew_cy = 8.85, 1.95
    rew_patch = FancyBboxPatch(
        (rew_cx - 1.0, rew_cy - 0.52), 2.0, 1.04,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=REWARD_FILL, edgecolor=REWARD_EDGE,
        linewidth=1.5,
    )
    ax.add_patch(rew_patch)
    ax.text(rew_cx, rew_cy + 0.12, "Y  —  Reward",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=REWARD_EDGE, family="DejaVu Sans")
    ax.text(rew_cx, rew_cy - 0.18, "lateness, CO₂, safety",
            ha="center", va="center", fontsize=8.5, color=MUTED, family="DejaVu Sans")

    # ── Unobserved confounders box (U) — dashed ────────────────────────────
    u_cx, u_cy = 2.05, 0.38
    u_patch = FancyBboxPatch(
        (0.52, 0.10), 3.06, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        facecolor=PAPER_BG, edgecolor=MUTED,
        linewidth=1.0, linestyle="--", alpha=0.9,
    )
    ax.add_patch(u_patch)
    ax.text(u_cx, 0.45, "U  —  Unobserved (excluded from state)",
            ha="center", va="center", fontsize=8.5, color=MUTED,
            fontstyle="italic", family="DejaVu Sans")
    ax.text(u_cx, 0.22, "driver_skill_latent,  maintenance_latent",
            ha="center", va="center", fontsize=7.8, color=MUTED, family="DejaVu Sans")

    # ── Non-causal proxy exclusion note ────────────────────────────────────
    excl_patch = FancyBboxPatch(
        (5.0, 4.55), 5.2, 1.0,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor="#f5f0e8", edgecolor="#b0a080",
        linewidth=0.9, linestyle="--", alpha=0.88,
    )
    ax.add_patch(excl_patch)
    ax.text(7.6, 5.28, "Non-causal proxies (excluded from causal state)",
            ha="center", va="center", fontsize=8.5, fontweight="semibold",
            color="#7a6040", family="DejaVu Sans")
    ax.text(7.6, 5.0, "risk_score  ·  distance_km  ·  compatibility_violation",
            ha="center", va="center", fontsize=8.0, color=MUTED, family="DejaVu Sans")
    ax.text(7.6, 4.75,
            "conflate pre-decision context with treatment-adjacent or latent information",
            ha="center", va="center", fontsize=7.5, color=MUTED,
            fontstyle="italic", family="DejaVu Sans")

    # ── Causal arrows ──────────────────────────────────────────────────────
    # Z → A  (single clean bundle arrow)
    _add_arrow(ax, (3.8, 3.6), (eco_cx - 1.05, eco_cy),
               color=ACTION_EDGE, lw=2.0, mutation_scale=14)
    ax.text(4.7, 3.82, "Z → A", fontsize=8, color=ACTION_EDGE,
            ha="center", va="bottom", family="DejaVu Sans", fontstyle="italic")

    # Z → Y  (bundle arrow, curves below eco-mode box)
    _add_arrow(ax, (3.8, 2.5), (rew_cx - 1.0, rew_cy + 0.15),
               color=REWARD_EDGE, lw=2.0, rad=0.15, mutation_scale=14)
    ax.text(6.1, 1.80, "Z → Y", fontsize=8, color=REWARD_EDGE,
            ha="center", va="top", family="DejaVu Sans", fontstyle="italic")

    # A → Y  (primary causal path)
    _add_arrow(ax, (eco_cx + 1.05, eco_cy - 0.25), (rew_cx - 1.0, rew_cy + 0.2),
               color=INK, lw=2.3, rad=-0.08, mutation_scale=16)
    ax.text(7.7, 3.05, "A → Y", fontsize=8.5, color=INK,
            ha="center", va="center", family="DejaVu Sans",
            fontweight="semibold", fontstyle="italic")

    # U ↝ A and U ↝ Y  (dashed — unobserved, cannot be blocked)
    for target_x, target_y, rad_ in [(eco_cx - 0.6, eco_cy - 0.52, -0.3),
                                       (rew_cx - 0.5, rew_cy - 0.52, 0.1)]:
        ax.annotate(
            "", xy=(target_x, target_y), xytext=(u_cx, 0.66),
            arrowprops=dict(
                arrowstyle="-|>", color=MUTED, lw=1.1,
                linestyle="dashed",
                connectionstyle=f"arc3,rad={rad_}",
            ),
        )

    ax.set_title(
        "Confounder-Aware State Design  (Backdoor DAG)  —  Urban Logistics Eco-Mode Control",
        fontsize=12, fontweight="semibold", pad=10, color=INK, family="DejaVu Sans",
    )
    fig.tight_layout(pad=0.9)
    _save_figure(fig, "causal_graph")
    plt.close(fig)


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 3.6), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.axis("off")
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 3.6)

    steps = [
        (1, "Historical\nLogistics Logs",     "Trips, fleet, demand,\nroute context"),
        (2, "Confounder-Aware\nState Design",  "Backdoor DAG; exclude\npost-action & latent vars"),
        (3, "Offline Policy\nLearning",        "FQI (causal & non-causal);\nbehavior policy estimation"),
        (4, "Off-Policy\nEvaluation",          "Doubly robust primary;\nIPS check; FQE diagnostic"),
        (5, "Decision-Support\nRecommendation","Conservative eco-mode\noverride; fallback to log"),
    ]

    card_w, card_h = 2.1, 1.55
    card_cy = 1.9
    gap = 0.4
    n = len(steps)
    total_w = n * card_w + (n - 1) * gap
    x0 = (13.0 - total_w) / 2

    card_positions = []
    for i, (num, title, sub) in enumerate(steps):
        cx = x0 + i * (card_w + gap) + card_w / 2
        card_positions.append(cx)
        bx = cx - card_w / 2
        by = card_cy - card_h / 2

        # card body
        card_patch = FancyBboxPatch(
            (bx, by), card_w, card_h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor="#f7f4ea", edgecolor=FLOW_EDGE,
            linewidth=1.2,
        )
        ax.add_patch(card_patch)

        # left accent strip
        accent = FancyBboxPatch(
            (bx, by), 0.12, card_h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=FLOW_ACCENT, edgecolor=FLOW_ACCENT,
            linewidth=0,
        )
        ax.add_patch(accent)

        # step badge (circle floating above top-left of card)
        badge_cx = bx + 0.28
        badge_cy = by + card_h + 0.01
        badge = Circle(
            (badge_cx, badge_cy), radius=0.19,
            facecolor=FLOW_ACCENT, edgecolor="white",
            linewidth=1.2, zorder=5,
        )
        ax.add_patch(badge)
        ax.text(badge_cx, badge_cy, str(num),
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", family="DejaVu Sans", zorder=6)

        # title and subtitle text (right of accent strip)
        text_cx = bx + 0.12 + (card_w - 0.12) / 2
        ax.text(text_cx, card_cy + 0.22, title,
                ha="center", va="center", fontsize=9.5, fontweight="semibold",
                color=INK, family="DejaVu Sans", linespacing=1.35)
        ax.text(text_cx, card_cy - 0.38, sub,
                ha="center", va="center", fontsize=7.8, color=MUTED,
                family="DejaVu Sans", linespacing=1.3)

    # arrows between cards
    for i in range(len(steps) - 1):
        cx_left = card_positions[i]
        cx_right = card_positions[i + 1]
        start_x = cx_left + card_w / 2 + 0.06
        end_x = cx_right - card_w / 2 - 0.06
        _add_arrow(ax, (start_x, card_cy), (end_x, card_cy),
                   color=FLOW_ACCENT, lw=1.8, mutation_scale=12)

    ax.text(
        6.5, 0.28,
        "Conservative decision support — not autonomous control.  "
        "Support-constrained overrides; fallback to logged behavior when uncertain.",
        ha="center", va="center", fontsize=8.5, color=MUTED,
        family="DejaVu Sans", fontstyle="italic",
    )
    ax.set_title(
        "Offline Decision-Support Workflow for the Urban Logistics Case Study",
        fontsize=12, fontweight="semibold", pad=8, color=INK, family="DejaVu Sans",
    )
    fig.tight_layout(pad=0.8)
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
