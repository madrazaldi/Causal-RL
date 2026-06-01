from __future__ import annotations

from datetime import datetime, timezone
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .config import (
    CAUSAL_DESIGN_MAPPING_PATH,
    CLUSTER_BOOTSTRAP_PATH,
    COMMON_SUPPORT_PATH,
    DECISION_LOG_PATH,
    FQE_CONVERGENCE_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    REWARD_CALIBRATION_PATH,
    MIN_PROPENSITY,
    REWARD_SENSITIVITY_PATH,
    ROBUSTNESS_PATH,
)
from .reward_calibration import write_reward_calibration_table


PAPER_BG = "#ffffff"
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
FIXED_EXPORT_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

POLICY_LABELS = {
    "non_causal_fqi": "Non-causal FQI",
    "heuristic_risk_rule": "Heuristic",
    "causal_no_vehicle_id_fqi": "Conf.-aware FQI no vehicle",
    "causal_no_history_fqi": "Conf.-aware FQI no history",
    "never_eco": "Never eco",
    "causal_fqi": "Conf.-aware FQI",
    "minimal_fqi": "Minimal FQI",
    "logged_behavior": "Logged replay",
    "always_eco": "Always eco",
}
SEGMENT_LABELS = {
    "high_traffic": "High traffic",
    "late_day": "Late day",
    "rain_or_event": "Rain/event",
    "tight_window": "Tight window",
}
PAPER_POLICY_ORDER = [
    "non_causal_fqi",
    "heuristic_risk_rule",
    "causal_no_vehicle_id_fqi",
    "causal_no_history_fqi",
    "never_eco",
    "causal_fqi",
    "minimal_fqi",
    "logged_behavior",
    "always_eco",
]
ROBUSTNESS_POLICY_ORDER = [
    "non_causal_fqi",
    "causal_fqi",
    "heuristic_risk_rule",
    "never_eco",
    "logged_behavior",
    "always_eco",
]
REWARD_LABELS = {
    "primary": "Primary",
    "service_heavy": "Service-heavy",
    "sustainability_heavy": "Sustainability-heavy",
}


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {
            "dpi": 300,
            "metadata": {
                "Software": "causal_rl.report",
                "Creation Time": FIXED_EXPORT_TIME.isoformat(),
            },
        },
        "pdf": {
            "metadata": {
                "Creator": "causal_rl.report",
                "Producer": "causal_rl.report",
                "CreationDate": FIXED_EXPORT_TIME,
                "ModDate": FIXED_EXPORT_TIME,
            },
        },
        "svg": {
            "metadata": {
                "Creator": "causal_rl.report",
                "Date": FIXED_EXPORT_TIME.isoformat(),
            },
        },
    }.items():
        output_path = FIGURES_DIR / f"{stem}.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **kwargs,
        )
        if suffix == "svg":
            text = output_path.read_text(encoding="utf-8")
            output_path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def _display_policy(name: str) -> str:
    return POLICY_LABELS.get(name, name.replace("_", " "))


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
    confounder_aware_family = {"causal_fqi", "causal_no_history_fqi", "causal_no_vehicle_id_fqi"}
    if name in confounder_aware_family:
        return "#009e73"  # green for confounder-aware variants
    if name == "minimal_fqi":
        return "#78c679"  # light green for minimal ablation
    if name == "non_causal_fqi":
        return "#0072b2"  # blue for non-causal FQI
    if name == "heuristic_risk_rule":
        return "#e69f00"  # orange for heuristic
    if name == "logged_behavior":
        return "#6c757d"  # darker gray for logged replay
    if name in {"always_eco", "never_eco"}:
        return "#8a8a8a"  # neutral gray for simple baselines
    return "#8c8c8c"  # neutral gray for baselines


def plot_policy_values(metrics_df: pd.DataFrame) -> None:
    ordered = (
        metrics_df.set_index("policy")
        .reindex([p for p in PAPER_POLICY_ORDER if p in set(metrics_df["policy"])])
        .dropna(subset=["policy_value_dr"])
        .sort_values("policy_value_dr", ascending=True)
        .reset_index()
    )
    if ordered.empty:
        return

    fig, ax = plt.subplots(figsize=(3.45, 3.15), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    y_pos = np.arange(len(ordered))
    colors = [_policy_color(p) for p in ordered["policy"]]

    if {"policy_value_dr_ci_low", "policy_value_dr_ci_high"}.issubset(ordered.columns):
        xerr = np.vstack(
            [
                ordered["policy_value_dr"] - ordered["policy_value_dr_ci_low"],
                ordered["policy_value_dr_ci_high"] - ordered["policy_value_dr"],
            ]
        )
    else:
        xerr = None

    for i, row in ordered.iterrows():
        ax.errorbar(
            row["policy_value_dr"],
            y_pos[i],
            xerr=None if xerr is None else xerr[:, [i]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=1.0,
            capsize=2.5,
            markersize=4.2,
            zorder=3,
        )

    if "logged_behavior" in set(ordered["policy"]):
        logged_value = float(ordered.loc[ordered["policy"] == "logged_behavior", "policy_value_dr"].iloc[0])
        ax.axvline(logged_value, color="#6c757d", linestyle="--", linewidth=0.9, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_display_policy(p) for p in ordered["policy"]], fontsize=7.2)
    ax.set_xlabel("DR value (higher is better)", fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=INK)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="x", color="#d8dde3", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#9aa4ad")
    fig.tight_layout(pad=0.35)
    _save_figure(fig, "policy_values")
    plt.close(fig)


def plot_robustness(robustness_df: pd.DataFrame) -> None:
    if robustness_df.empty:
        return
    focus = robustness_df[robustness_df["policy"].isin(ROBUSTNESS_POLICY_ORDER)].copy()
    if focus.empty:
        return
    pivot = focus.pivot_table(
        index="policy",
        columns="segment",
        values="policy_value_dr",
        aggfunc="mean",
    )
    segment_order = [s for s in ["high_traffic", "rain_or_event", "tight_window", "late_day"] if s in pivot.columns]
    pivot = pivot.reindex([p for p in ROBUSTNESS_POLICY_ORDER if p in pivot.index])[segment_order]
    pivot.index = [_display_policy(p) for p in pivot.index]
    pivot.columns = [SEGMENT_LABELS.get(c, c.replace("_", " ")) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(3.55, 2.45), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("crest", as_cmap=True),
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "DR value", "shrink": 0.75},
        annot_kws={"fontsize": 6.5},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=6.8, rotation=28, colors=INK)
    ax.tick_params(axis="y", labelsize=6.8, rotation=0, colors=INK)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("DR value", fontsize=7, color=INK)
    fig.tight_layout(pad=0.25)
    _save_figure(fig, "robustness_heatmap")
    plt.close(fig)


def plot_reward_sensitivity(reward_df: pd.DataFrame) -> None:
    plot_df = reward_df.copy()
    plot_df = plot_df[plot_df["policy"].isin(PAPER_POLICY_ORDER)]
    plot_df["policy_label"] = plot_df["policy"].map(_display_policy)
    plot_df["reward_label"] = plot_df["reward_name"].map(REWARD_LABELS).fillna(plot_df["reward_name"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    sns.barplot(
        data=plot_df,
        x="policy_label",
        y="policy_value_dr",
        hue="reward_label",
        palette="colorblind",
        ax=ax,
    )
    ax.set_ylabel("DR value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(title="", fontsize=8, frameon=False, ncol=3, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    _save_figure(fig, "reward_sensitivity")
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
    ax.text(
        u_cx,
        0.22,
        "driver_skill_latent, maintenance_latent\n driver_risk_propensity_latent",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
        family="DejaVu Sans",
    )

    # ── Non-causal proxy exclusion note ────────────────────────────────────
    excl_patch = FancyBboxPatch(
        (5.0, 4.55), 5.2, 1.0,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor="#f5f0e8", edgecolor="#b0a080",
        linewidth=0.9, linestyle="--", alpha=0.88,
    )
    ax.add_patch(excl_patch)
    ax.text(7.6, 5.28, "Proxy discipline",
            ha="center", va="center", fontsize=8.5, fontweight="semibold",
            color="#7a6040", family="DejaVu Sans")
    ax.text(7.6, 5.0, "distance_km, risk_score excluded from learned policy state",
            ha="center", va="center", fontsize=8.0, color=MUTED, family="DejaVu Sans")
    ax.text(7.6, 4.75,
            "compatibility_violation retained only in the non-causal comparator",
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


def write_causal_design_mapping(metadata: dict) -> pd.DataFrame:
    causal_state = metadata.get("causal_state_columns", [])
    latent_columns = metadata.get("latent_columns", [])
    post_action_columns = metadata.get("post_action_columns", [])
    rows = [
        {
            "dag_node": "A",
            "causal_role": "treatment/action",
            "project_term": "eco-mode decision",
            "columns": "eco_mode; action",
            "policy_use": "binary decision learned or replayed by each policy",
            "claim_boundary": "only treatment/action in the main benchmark",
        },
        {
            "dag_node": "Y",
            "causal_role": "primary outcome",
            "project_term": "composite reward",
            "columns": "reward_primary",
            "policy_use": "headline policy-value outcome",
            "claim_boundary": "scalar utility for the synthetic benchmark, not a real deployment cost",
        },
        {
            "dag_node": "Y components",
            "causal_role": "component outcomes",
            "project_term": "service, safety, and emissions",
            "columns": "lateness_min; crash; near_miss; co2_kg",
            "policy_use": "reported as component diagnostics and combined into reward",
            "claim_boundary": "CO2 is one outcome component, not the only estimand",
        },
        {
            "dag_node": "Z",
            "causal_role": "observed adjustment set",
            "project_term": "pre-decision deployable context",
            "columns": "; ".join(causal_state),
            "policy_use": "state for confounder-aware FQI, heuristic, and causal nuisance models",
            "claim_boundary": "domain-guided adjustment heuristic, not proof of full identification",
        },
        {
            "dag_node": "U",
            "causal_role": "unobserved/non-deployable factors",
            "project_term": "latent simulator factors",
            "columns": "; ".join(latent_columns),
            "policy_use": "excluded from deployable policies",
            "claim_boundary": "remaining latent confounding prevents strong backdoor claims",
        },
        {
            "dag_node": "post-action proxies",
            "causal_role": "excluded outcomes/proxies",
            "project_term": "realized trip and safety fields",
            "columns": "; ".join(post_action_columns),
            "policy_use": "excluded from deployable learned state",
            "claim_boundary": "included only as outcomes, diagnostics, or explicit oracle/proxy checks",
        },
        {
            "dag_node": "proxy comparator",
            "causal_role": "non-causal comparator field",
            "project_term": "assignment compatibility flag",
            "columns": "compatibility_violation",
            "policy_use": "included only in non_causal_fqi state and nuisance models",
            "claim_boundary": "not part of the confounder-aware deployable state",
        },
    ]
    mapping = pd.DataFrame(rows)
    CAUSAL_DESIGN_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(CAUSAL_DESIGN_MAPPING_PATH, index=False)
    return mapping


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 4.2), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        (1, "Historical logs", "Trips, demand, fleet, route context"),
        (2, "Confounder-aware state", "Pre-decision covariates; no latent or post-action fields"),
        (3, "Policy learning", "Behavior model plus confounder-aware and non-causal FQI"),
        (4, "OPE audit", "DR headline, IPS check, FQE diagnostic"),
        (5, "Recommendation", "Support gate; fallback to logged replay"),
    ]

    card_x = 0.14
    card_w = 0.78
    card_h = 0.128
    centers_y = np.linspace(0.86, 0.14, len(steps))
    for (num, title, subtitle), cy in zip(steps, centers_y):
        bx = card_x
        by = cy - card_h / 2
        card_patch = FancyBboxPatch(
            (bx, by), card_w, card_h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="#f7f4ea",
            edgecolor=FLOW_EDGE,
            linewidth=0.9,
        )
        ax.add_patch(card_patch)
        ax.plot(
            [bx + 0.018, bx + 0.018],
            [by + 0.02, by + card_h - 0.02],
            color=FLOW_ACCENT,
            lw=3.0,
            solid_capstyle="round",
        )
        badge_cx = bx + 0.085
        badge = Circle(
            (badge_cx, cy),
            radius=0.034,
            facecolor=FLOW_ACCENT,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        ax.add_patch(badge)
        ax.text(
            badge_cx,
            cy,
            str(num),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="white",
            family="DejaVu Sans",
            zorder=6,
        )
        text_x = bx + 0.15
        ax.text(
            text_x,
            cy + 0.026,
            title,
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="semibold",
            color=INK,
            family="DejaVu Sans",
        )
        ax.text(
            text_x,
            cy - 0.026,
            textwrap.fill(subtitle, width=38),
            ha="left",
            va="center",
            fontsize=5.9,
            color=MUTED,
            family="DejaVu Sans",
            linespacing=1.12,
        )

    for top_y, bottom_y in zip(centers_y[:-1], centers_y[1:]):
        _add_arrow(
            ax,
            (0.53, top_y - card_h / 2 - 0.012),
            (0.53, bottom_y + card_h / 2 + 0.012),
            color=FLOW_ACCENT,
            lw=1.15,
            mutation_scale=9,
        )

    ax.text(
        0.53,
        0.975,
        "Offline decision-support pipeline",
        ha="center",
        va="top",
        fontsize=8.6,
        fontweight="semibold",
        color=INK,
        family="DejaVu Sans",
    )
    fig.tight_layout(pad=0.04)
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
        "Confounder-Aware FQI: Top Q-function Features (Permutation Importance)",
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
    focus_policies = [p for p in ["non_causal_fqi", "causal_fqi", "logged_behavior"] if p in common_support_df["policy"].values]
    if not focus_policies:
        return
    plot_df = common_support_df.set_index("policy").reindex(focus_policies).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.6), facecolor=PAPER_BG)
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    labels = [_display_policy(p) for p in plot_df["policy"]]
    colors = [_policy_color(p) for p in plot_df["policy"]]
    bars = ax.bar(labels, plot_df["pct_below_tau_mu"], color=colors, edgecolor="white", linewidth=0.6)
    ax.text(
        0.98,
        0.92,
        f"tau_mu = {MIN_PROPENSITY:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    for bar, pct in zip(bars, plot_df["pct_below_tau_mu"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.set_ylabel("Rows below support threshold (%)", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=7.5, rotation=15)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(0, max(0.08, float(plot_df["pct_below_tau_mu"].max()) * 1.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.45)
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
            label=_display_policy(policy_name),
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
    decision_log_df = pd.read_csv(DECISION_LOG_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    calibration_df = write_reward_calibration_table(decision_log_df)
    causal_design_df = write_causal_design_mapping(metadata)
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
    print(f"Saved reward calibration audit with {len(calibration_df)} rows to {REWARD_CALIBRATION_PATH}")
    print(f"Saved causal design mapping with {len(causal_design_df)} rows to {CAUSAL_DESIGN_MAPPING_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
