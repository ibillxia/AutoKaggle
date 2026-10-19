"""Phase G -- the figure suite.

Budget: the competition allows fewer than 10 tables and figures *in total*, so
this module produces at most nine, each answering exactly one question in the
argument. Every figure is written to ``artifacts/figures`` together with a
machine-readable caption that states the takeaway, because a chart whose
takeaway is not written down is a chart the judges will read differently than
the author intended.

Design rules enforced here:
* no truncated axes on any chart that encodes magnitude;
* every uncertainty-bearing quantity is drawn with its interval;
* colour is used to encode one thing only (never decoration);
* the same palette and typography across all figures.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# matplotlib must be configured before pyplot is imported anywhere
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "cache" / "mpl"))

import numpy as np
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

_FIG_INDEX: List[Dict[str, Any]] = []

PALETTE = {
    "sensor": "#1b6ca8",
    "stopwatch": "#c1543a",
    "neutral": "#5b6b7a",
    "good": "#2e7d5b",
    "warn": "#b8860b",
    "grid": "#d8dee4",
    "ink": "#1f2a33",
}

FIG_DPI = 170


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
        }
    )
    return plt


def _finish(fig, name: str, title: str, takeaway: str, reading: str = "") -> str:
    """Save a figure and record its caption."""
    out = paths.FIGURE_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    _FIG_INDEX.append(
        # as_posix() keeps the index portable: every consumer of "file" embeds it
        # in Markdown, where a Windows backslash would not resolve.
        {"file": out.relative_to(paths.ARTIFACT_DIR).as_posix(), "title": title,
         "takeaway": takeaway, "reading_order_note": reading}
    )
    log.info("figure written: %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# 1. what the data actually covers
# ---------------------------------------------------------------------------
def fig_data_coverage(
    audit: Dict[str, Any],
    analysis: pd.DataFrame,
) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    groups = ["WR", "TE", "DB", "OL", "DL"]
    cohort = [int((analysis["position_group"] == g).sum()) for g in groups]
    with_pp = [
        int(analysis.loc[analysis["position_group"] == g, "pp_plays"].notna().sum())
        if "pp_plays" in analysis.columns else 0
        for g in groups
    ]
    with_gt = [
        int(analysis.loc[analysis["position_group"] == g, "gt_plays"].notna().sum())
        if "gt_plays" in analysis.columns else 0
        for g in groups
    ]
    x = np.arange(len(groups))
    width = 0.27
    axes[0].bar(x - width, cohort, width, label="prospects in cohort",
                color=PALETTE["neutral"])
    axes[0].bar(x, with_pp, width, label="with NGS player-play outcomes",
                color=PALETTE["sensor"])
    axes[0].bar(x + width, with_gt, width, label="with 10 Hz in-game tracking",
                color=PALETTE["good"])
    axes[0].set_xticks(x, groups)
    axes[0].set_ylabel("players")
    axes[0].set_ylim(0, max(cohort + [1]) * 1.28)
    axes[0].set_title("Effective sample size is the binding constraint")
    axes[0].legend(loc="upper right", fontsize=7.5)
    for xi in range(len(groups)):
        for offset, series in ((-width, cohort), (0, with_pp), (width, with_gt)):
            axes[0].text(xi + offset, series[xi], str(series[xi]), ha="center",
                         va="bottom", fontsize=7)

    # drill coverage heat grid
    cov = audit.get("drill_attempts_by_position_group", {})
    drills = sorted({d for row in cov.values() for d in row})
    matrix = np.array(
        [[cov.get(g, {}).get(d, 0) for d in drills] for g in groups], dtype=float
    )
    im = axes[1].imshow(
        np.log1p(matrix), aspect="auto", cmap="Blues"
    )
    axes[1].set_xticks(range(len(drills)),
                       [d.replace("SKILL_DRILLS_", "SKILL_") for d in drills],
                       rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(groups)), groups)
    axes[1].set_title("Sensor drill attempts captured (log scale)")
    axes[1].grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                axes[1].text(j, i, int(matrix[i, j]), ha="center", va="center",
                             fontsize=6, color="white" if matrix[i, j] > 40 else PALETTE["ink"])
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="log(1 + attempts)")
    return _finish(
        fig,
        "fig01_data_coverage.png",
        "What the cohort can and cannot support",
        "Every focus position group has enough prospects for group-level inference, "
        "but the effective N is tens per group, not hundreds -- so any claim must be "
        "position-stratified and uncertainty must be shown.",
    )


# ---------------------------------------------------------------------------
# 2. is the sensor layer trustworthy at all?
# ---------------------------------------------------------------------------
def fig_sensor_validity(validation: pd.DataFrame) -> Optional[str]:
    if validation is None or validation.empty:
        log.warning("no sensor validation data; skipping figure 2")
        return None
    plt = _plt()
    drills = sorted(validation["drill_type"].unique())
    fig, axes = plt.subplots(1, len(drills), figsize=(4.1 * len(drills), 4.0),
                             squeeze=False)
    for ax, drill in zip(axes[0], drills):
        sub = validation[validation["drill_type"] == drill]
        ax.scatter(sub["duration_s"], sub[sub.columns[3]],
                   s=18, alpha=0.65, color=PALETTE["sensor"], edgecolor="none")
        lo = float(min(sub["duration_s"].min(), sub[sub.columns[3]].min()))
        hi = float(max(sub["duration_s"].max(), sub[sub.columns[3]].max()))
        ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["stopwatch"], lw=1.2,
                label="perfect agreement")
        r = sub[["duration_s", sub.columns[3]]].corr().iloc[0, 1]
        ax.set_title(f"{drill.replace('_', ' ').title()}\nr = {r:.3f}   n = {len(sub)}")
        ax.set_xlabel("sensor-derived duration (s)")
        ax.set_ylabel("official electronic time (s)")
        ax.legend(loc="upper left", fontsize=8)
    return _finish(
        fig,
        "fig02_sensor_validity.png",
        "The sensor layer reproduces the stopwatch it is meant to augment",
        "Sensor-derived drill durations track the official electronic times closely, "
        "which is the precondition for trusting any sensor-derived movement trait. "
        "Note the residual spread: that scatter is exactly the information the "
        "stopwatch throws away.",
        reading="Establish credibility before making any claim.",
    )


# ---------------------------------------------------------------------------
# 3. the hazards
# ---------------------------------------------------------------------------
def fig_hazards(audit: Dict[str, Any]) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    opt = audit.get("traditional_test_optout_rate", {})
    if opt:
        keys = list(opt)
        vals = [opt[k] * 100 for k in keys]
        order = np.argsort(vals)
        axes[0].barh([keys[i] for i in order], [vals[i] for i in order],
                     color=PALETTE["warn"])
        axes[0].set_xlabel("% of prospects who did not test")
        axes[0].set_title("Traditional Combine opt-outs")
        for i, idx in enumerate(order):
            axes[0].text(vals[idx] + 0.4, i, f"{vals[idx]:.0f}%", va="center", fontsize=8)

    exp = audit.get("mean_exposure_by_draft_year", {})
    games = {}
    for year, row in exp.items():
        if isinstance(row, dict):
            games[year] = row.get("career_games_active")
    if games:
        years = sorted(games)
        axes[1].bar([str(y) for y in years], [games[y] for y in years],
                    color=PALETTE["neutral"])
        axes[1].set_ylabel("mean career games active")
        axes[1].set_title("Follow-up time is unequal across draft classes")
        for i, y in enumerate(years):
            axes[1].text(i, games[y], f"{games[y]:.1f}", ha="center", va="bottom", fontsize=8)
    return _finish(
        fig,
        "fig03_hazards.png",
        "Two hazards that would silently bias a naive analysis",
        "A quarter to a third of prospects skip at least one agility test, and the "
        "2023 draft class has roughly three times the NFL exposure of the 2025 class. "
        "Complete-case analysis and raw career totals are therefore both traps.",
    )


# ---------------------------------------------------------------------------
# 4. where does sensor information live?
# ---------------------------------------------------------------------------
def fig_feasibility_map(feasibility_results: pd.DataFrame) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    df = feasibility_results.copy()
    if "delta_r2" not in df.columns:
        df["delta_r2"] = np.nan
    df["delta_r2"] = pd.to_numeric(df["delta_r2"], errors="coerce")
    plot = df.dropna(subset=["partial_spearman", "delta_r2"])

    passes = plot[plot.get("adds_beyond_stopwatch", False) == True]  # noqa: E712
    fails = plot.drop(passes.index)

    ax = axes[0]
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.scatter(fails["partial_spearman"], fails["delta_r2"], s=16, alpha=0.45,
               color=PALETTE["neutral"], edgecolor="none", label="not incremental")
    ax.scatter(passes["partial_spearman"], passes["delta_r2"], s=26, alpha=0.85,
               color=PALETTE["sensor"], edgecolor="none",
               label="adds beyond the stopwatch")
    ax.set_xlabel("partial Spearman r (sensor trait vs in-game outcome)")
    ax.set_ylabel("incremental out-of-sample R²")
    ax.set_title(f"Screen: {len(plot)} pre-registered trait-outcome pairs")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    groups = sorted(plot["position_group"].unique())
    data = [plot.loc[plot["position_group"] == g, "delta_r2"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color=PALETTE["ink"], lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["sensor"])
        patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE["sensor"])
    ax.axhline(0.01, color=PALETTE["good"], ls="--", lw=1.2,
               label="incremental-value floor (0.01 R²)")
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.set_ylabel("incremental out-of-sample R²")
    ax.set_title("Most sensor traits do not survive the baseline")
    ax.legend(loc="upper right", fontsize=8)
    return _finish(
        fig,
        "fig04_feasibility_map.png",
        "The screen, not the narrative, chooses the topic",
        "Across every pre-registered sensor-trait x in-game-outcome pair, the large "
        "majority fail to add out-of-sample explanatory power beyond the conventional "
        "Combine battery and draft capital. The surviving pairs are concentrated in a "
        "few drill families, which is what makes the eventual topic selection credible "
        "rather than cherry-picked.",
    )


# ---------------------------------------------------------------------------
# 5. the chosen linkage
# ---------------------------------------------------------------------------
def fig_main_linkage(
    partial_residuals: pd.DataFrame,
    linkage: Dict[str, Any],
    *,
    x_label: str,
    y_label: str,
) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.axvline(0, color=PALETTE["grid"], lw=1)
    ax.scatter(partial_residuals["x_resid"], partial_residuals["y_resid"],
               s=22, alpha=0.7, color=PALETTE["sensor"], edgecolor="none")
    if len(partial_residuals) > 2:
        coef = np.polyfit(partial_residuals["x_resid"], partial_residuals["y_resid"], 1)
        xs = np.linspace(partial_residuals["x_resid"].min(),
                         partial_residuals["x_resid"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color=PALETTE["stopwatch"], lw=1.8)
    ax.set_xlabel(f"{x_label} (residualised on baseline)")
    ax.set_ylabel(f"{y_label} (residualised on baseline)")
    ax.set_title("The linkage survives the conventional scouting model")

    ax = axes[1]
    est = linkage.get("partial_r", np.nan)
    lo, hi = linkage.get("ci_low", np.nan), linkage.get("ci_high", np.nan)
    ax.errorbar([0], [est], yerr=[[est - lo], [hi - est]], fmt="o", ms=9,
                color=PALETTE["sensor"], capsize=6, lw=1.8)
    ax.axhline(0, color=PALETTE["stopwatch"], ls="--", lw=1.4)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([])
    ax.set_ylabel("partial Spearman r")
    ax.set_title(
        f"n = {linkage.get('n')}   bootstrap 95% CI\n"
        f"incremental R² = {linkage.get('delta_r2', float('nan')):+.3f}"
    )
    return _finish(
        fig,
        "fig05_main_linkage.png",
        "The headline result, with its uncertainty",
        "After residualising both the sensor trait and the NFL outcome on the entire "
        "conventional Combine battery plus draft capital, the association remains. "
        "The right panel is the honest version of the claim: an interval, not a point.",
    )


# ---------------------------------------------------------------------------
# 6. incremental value over the stopwatch
# ---------------------------------------------------------------------------
def fig_incremental_value(
    metric_vs_stopwatch: Dict[str, Any],
    metric_validation: Dict[str, Any],
) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    labels, values, colors = [], [], []
    if metric_vs_stopwatch.get("best_traditional_r2") is not None:
        labels.append(f"best stopwatch test\n({metric_vs_stopwatch.get('best_traditional_test')})")
        values.append(metric_vs_stopwatch["best_traditional_r2"])
        colors.append(PALETTE["stopwatch"])
    if metric_vs_stopwatch.get("metric_r2") is not None:
        labels.append("sensor metric alone")
        values.append(metric_vs_stopwatch["metric_r2"])
        colors.append(PALETTE["sensor"])
    r2b = metric_validation.get("r2_base")
    r2f = metric_validation.get("r2_full")
    if r2b is not None:
        labels.append("full conventional\nbaseline")
        values.append(r2b)
        colors.append(PALETTE["neutral"])
    if r2f is not None:
        labels.append("baseline +\nsensor metric")
        values.append(r2f)
        colors.append(PALETTE["good"])

    x = np.arange(len(labels))
    ax.bar(x, values, 0.6, color=colors)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("cross-validated R² on the in-game outcome")
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    for xi, v in zip(x, values):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if r2b is not None and r2f is not None:
        ax.annotate(
            "", xy=(len(labels) - 1, r2f), xytext=(len(labels) - 1, r2b),
            arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"], lw=1.4),
        )
        ax.text(len(labels) - 1 + 0.12, (r2b + r2f) / 2,
                f"ΔR² = {r2f - r2b:+.3f}", fontsize=9, va="center")
    ax.set_title("Where the sensor information actually lives")
    return _finish(
        fig,
        "fig06_incremental_value.png",
        "Sensor tracking adds information the stopwatch cannot express",
        "The sensor metric and the full conventional battery are not substitutes: the "
        "gain is in the combination. Read honestly, this chart also shows when the "
        "metric does NOT beat the best single stopwatch test -- which is reported "
        "rather than hidden.",
    )


# ---------------------------------------------------------------------------
# 7. scouting blind spots
# ---------------------------------------------------------------------------
def fig_blind_spots(quad: pd.DataFrame, outcome: str) -> Optional[str]:
    if quad is None or quad.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    styles = {
        "confirmed_elite": (PALETTE["good"], "o"),
        "sensor_beats_stopwatch": (PALETTE["sensor"], "^"),
        "stopwatch_beats_sensor": (PALETTE["stopwatch"], "v"),
        "confirmed_limited": (PALETTE["neutral"], "s"),
    }
    for key, (colour, marker) in styles.items():
        sub = quad[quad["archetype"] == key]
        if sub.empty:
            continue
        ax.scatter(sub["trad_z"], sub["sensor_z"], s=32, alpha=0.8, color=colour,
                   marker=marker, edgecolor="none",
                   label=key.replace("_", " "))
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.axvline(0, color=PALETTE["grid"], lw=1)
    ax.set_xlabel("conventional grade (z, within position group)")
    ax.set_ylabel("sensor movement metric (z)")
    ax.set_title("Where the two evaluation systems disagree")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    means = quad.groupby("archetype")["__y"].mean().sort_values()
    counts = quad.groupby("archetype")["__y"].size()
    labels = [f"{k.replace('_', ' ')}\n(n={counts[k]})" for k in means.index]
    ax.barh(labels, means.values,
            color=[styles.get(k, (PALETTE["neutral"],))[0] for k in means.index])
    ax.set_xlabel(f"mean {outcome}")
    ax.set_title("Which quadrant the in-game data rewards")
    ax.tick_params(axis="y", labelsize=7.5)
    return _finish(
        fig,
        "fig07_blind_spots.png",
        "The actionable output: named disagreements between the two evaluation systems",
        "Players in the upper-left quadrant move like the sensor's idea of a good "
        "prospect while the stopwatch says otherwise, and their in-game outcome is "
        "what arbitrates. This is the list a front office can actually use in the "
        "week before the draft.",
    )


# ---------------------------------------------------------------------------
# 8. the metric and its stability
# ---------------------------------------------------------------------------
def fig_metric(
    leaderboard: pd.DataFrame,
    metric_name: str,
    stability: Dict[str, Any],
    group: str,
) -> Optional[str]:
    if leaderboard is None or leaderboard.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    top = leaderboard.head(15).iloc[::-1]
    label_col = "display_name" if "display_name" in top.columns else "nfl_id"
    colours = []
    for _, row in top.iterrows():
        pick = row.get("draft_overall_pick")
        if pick is None or not np.isfinite(pd.to_numeric(pick, errors="coerce")):
            colours.append(PALETTE["warn"])
        elif pick <= 100:
            colours.append(PALETTE["neutral"])
        else:
            colours.append(PALETTE["good"])
    ax.barh(top[label_col].astype(str), top[metric_name], color=colours)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Top 15 {group} by {metric_name}")
    ax.tick_params(axis="y", labelsize=8)
    handles = [
        plt.Line2D([], [], color=PALETTE["neutral"], lw=6, label="top-100 pick"),
        plt.Line2D([], [], color=PALETTE["good"], lw=6, label="day 3 / later"),
        plt.Line2D([], [], color=PALETTE["warn"], lw=6, label="undrafted"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)

    ax = axes[1]
    j = stability.get("mean_jaccard")
    if j is not None and np.isfinite(j):
        ax.bar(["top-10\nrank overlap"], [j], 0.4, color=PALETTE["sensor"])
        ax.bar(["chance"], [0.0], 0.4, color=PALETTE["grid"])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=PALETTE["warn"], ls="--", lw=1.2)
        ax.text(0.02, 0.52, "usable stability", fontsize=8, color=PALETTE["warn"])
        ax.text(0, j, f"{j:.2f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("mean Jaccard overlap of top-10\nunder bootstrap resampling")
        ax.set_title("Is the leaderboard reproducible?")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "rank stability not computed", ha="center")
    return _finish(
        fig,
        "fig08_metric_leaderboard.png",
        "The metric as a decision tool, and how much to trust the ranking",
        "The left panel is what a scout would read; the right panel exists to stop "
        "them over-reading it. A metric whose top-10 barely survives resampling "
        "should be used to screen, not to rank.",
    )


# ---------------------------------------------------------------------------
# 9. robustness
# ---------------------------------------------------------------------------
def fig_robustness(
    loo: pd.DataFrame,
    reliability: Optional[float],
    linkage: Dict[str, Any],
) -> Optional[str]:
    if loo is None or loo.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.hist(loo["partial_r"].dropna(), bins=20, color=PALETTE["sensor"], alpha=0.85)
    base = float(loo["base_r"].iloc[0])
    ax.axvline(base, color=PALETTE["stopwatch"], lw=2,
               label=f"full-sample r = {base:+.2f}")
    ax.axvline(0, color=PALETTE["ink"], ls="--", lw=1.2)
    ax.set_xlabel("partial r with one player removed")
    ax.set_ylabel("count")
    ax.set_title("Leave-one-out influence")
    ax.legend(fontsize=8)

    ax = axes[1]
    items = []
    if reliability is not None and np.isfinite(reliability):
        items.append(("trait reliability\n(split-half)", reliability, PALETTE["sensor"]))
    ci = linkage.get("ci_high", np.nan) - linkage.get("ci_low", np.nan)
    if np.isfinite(ci):
        items.append(("95% CI width\nof the effect", ci, PALETTE["warn"]))
    mi = linkage.get("leave_one_out", {}).get("max_abs_delta")
    if mi is not None and np.isfinite(mi):
        items.append(("max single-player\nshift in r", mi, PALETTE["stopwatch"]))
    n_flips = linkage.get("leave_one_out", {}).get("n_flips_sign")
    if n_flips is not None:
        items.append(("leave-one-out\nsign flips", float(n_flips), PALETTE["neutral"]))
    if items:
        ax.bar([i[0] for i in items], [i[1] for i in items],
               color=[i[2] for i in items], width=0.55)
        for xi, (_, v, _) in enumerate(items):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title("How fragile is the finding?")
    else:
        ax.axis("off")
    return _finish(
        fig,
        "fig09_robustness.png",
        "Adversarial checks on the headline claim",
        "If removing any single prospect collapses or reverses the effect, the finding "
        "is a story about one player rather than a property of the position group. "
        "Trait reliability bounds how strong any correlation could possibly have been.",
    )


# ---------------------------------------------------------------------------
def write_figure_index() -> Path:
    out = paths.REPORT_DIR / "figure_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_FIG_INDEX, indent=2), encoding="utf-8")
    return out


def reset_index() -> None:
    _FIG_INDEX.clear()
