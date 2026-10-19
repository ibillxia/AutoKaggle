"""Phases E/F -- confirmatory analysis and metric construction.

The feasibility scan is a *screen*: it deliberately looks at many pairs and can
only tell us where to look. This module is the confirmation step for the one
topic that survives, and it is written to make over-claiming difficult:

* every effect is reported with a bootstrap confidence interval and its n;
* every effect is re-estimated under leave-one-out to expose single-player
  dependence;
* the baseline is always the conventional scouting model, so the headline claim
  is an *incremental* claim;
* the metric that gets published is checked for rank stability, because a metric
  whose leaderboard reshuffles under resampling is not actionable even if its
  aggregate correlation is significant.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import feasibility, io, paths, stats

log = logging.getLogger(__name__)


@dataclass
class LinkageResult:
    """Everything needed to defend one Combine -> NFL linkage claim."""

    position_group: str
    outcome: str
    trait: str
    partial_r: float
    ci_low: float
    ci_high: float
    n: int
    p_boot: float
    q_value: float
    delta_r2: float
    r2_base: float
    r2_full: float
    reliability: Optional[float] = None
    loo: Dict[str, float] = field(default_factory=dict)
    ols: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "position_group": self.position_group,
            "outcome": self.outcome,
            "trait": self.trait,
            "partial_r": _num(self.partial_r),
            "ci_low": _num(self.ci_low),
            "ci_high": _num(self.ci_high),
            "n": int(self.n),
            "p_boot": _num(self.p_boot),
            "q_value": _num(self.q_value),
            "delta_r2": _num(self.delta_r2),
            "r2_base": _num(self.r2_base),
            "r2_full": _num(self.r2_full),
            "reliability": _num(self.reliability),
            "leave_one_out": {k: _num(v) for k, v in self.loo.items()},
            "ols": self.ols,
            **self.extras,
        }
        return payload

    def sentence(self) -> str:
        """A single defensible claim, with uncertainty and incremental value."""
        sign = "positively" if self.partial_r > 0 else "negatively"
        return (
            f"In {self.position_group}, {self.trait} is {sign} associated with "
            f"{self.outcome} after controlling for the conventional Combine battery "
            f"and draft capital (partial r = {self.partial_r:+.2f}, "
            f"95% CI [{self.ci_low:+.2f}, {self.ci_high:+.2f}], n = {self.n}), "
            f"adding {self.delta_r2:+.3f} out-of-sample R^2."
        )


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


# ---------------------------------------------------------------------------
# Confirmation of a single linkage
# ---------------------------------------------------------------------------
def confirm_linkage(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
    trait: str,
    *,
    n_boot: int = 4000,
    seed: int = 0,
    cluster_by: Optional[str] = "draft_year",
) -> LinkageResult:
    """Full confirmatory battery for one (position group, trait, outcome) triple."""
    Z, y, sub, _ = feasibility.build_design(analysis, position_group, outcome)
    Z = Z.apply(feasibility._winsorize)
    x = feasibility._winsorize(pd.to_numeric(sub[trait], errors="coerce"))

    boot = stats.bootstrap_partial_correlation(
        x, y, Z, n_boot=n_boot, seed=seed,
        cluster=sub[cluster_by] if cluster_by and cluster_by in sub.columns else None,
    )
    cv = feasibility.cv_incremental_r2(Z, x, y)

    # leave-one-out influence on the partial correlation
    frame = pd.concat([x.rename("x"), y.rename("y"), sub[["nfl_id"]], Z], axis=1)
    loo_frame = stats.leave_one_out_stability(frame["x"], frame["y"], Z)
    loo = stats.influence_summary(loo_frame)

    # a robust-regression view of the effect size in outcome standard deviations
    ols_text = None
    try:
        design = Z.copy()
        design["__trait_z"] = stats.zscore_within(x)
        fitted = stats.ols_robust(design, stats.zscore_within(y))
        ols_text = fitted.summary()
    except Exception as exc:  # pragma: no cover
        log.warning("OLS failed for %s/%s: %s", trait, outcome, exc)

    # reliability of the trait across the player's own repeated attempts
    reliability = None
    try:
        attempts = io.read_parquet_cache("combine_attempt_features.parquet")
        if attempts is None:
            raise FileNotFoundError("combine attempt features not cached")
        drill, feature, _ = feasibility.parse_trait(trait)
        drill_type = {
            "three_cone_drill": "THREE_CONE_DRILL",
            "short_shuttle": "SHORT_SHUTTLE",
            "forty_yard_dash": "FORTY_YARD_DASH",
        }.get(drill)
        if drill_type and feature in attempts.columns:
            rel = feasibility.split_half_reliability(
                attempts, feature=feature, drill_type=drill_type
            )
            reliability = rel.get("reliability")
        elif drill == "all_drills" or trait.startswith("pooled_"):
            # pooled traits are averages over many attempts and drills, so their
            # reliability is estimated directly with a bootstrap split-half
            reliability = _pooled_reliability(attempts, trait)
    except FileNotFoundError:
        pass

    return LinkageResult(
        position_group=position_group,
        outcome=outcome,
        trait=trait,
        partial_r=boot["estimate"],
        ci_low=boot["ci_low"],
        ci_high=boot["ci_high"],
        n=int(boot["n"]),
        p_boot=boot["p_boot"],
        q_value=float("nan"),
        delta_r2=cv.get("delta_r2", float("nan")),
        r2_base=cv.get("r2_base", float("nan")),
        r2_full=cv.get("r2_full", float("nan")),
        reliability=reliability,
        loo=loo,
        ols=ols_text,
    )


# ---------------------------------------------------------------------------
# From trait to published metric
# ---------------------------------------------------------------------------
DEFAULT_METRIC_NAME = "Break Retention Index"


def build_metric(
    analysis: pd.DataFrame,
    position_group: str,
    *,
    retention_traits: Sequence[str],
    regain_traits: Sequence[str] = (),
    name: str = DEFAULT_METRIC_NAME,
) -> pd.DataFrame:
    """Construct the composite metric and attach it to the analysis rows.

    Components are standardised *within the position group* before averaging so
    that a receiver and a defensive tackle are never compared on a raw quantity
    that means different things for each.

    ``retention_traits`` enter with a positive sign (keep more of your speed
    through the break is good); ``regain_traits`` enter with a negative sign
    (taking longer to get back up to speed is bad).
    """
    components = list(retention_traits) + list(regain_traits)
    direction = [1.0] * len(retention_traits) + [-1.0] * len(regain_traits)
    sub = analysis[analysis["position_group"] == position_group].copy()
    available = [c for c in components if c in sub.columns]

    if not available:
        sub[name] = np.nan
        return sub

    # keep at least one of each family available
    keep_ret = [c for c in retention_traits if c in available]
    keep_reg = [c for c in regain_traits if c in available]
    components = keep_ret + keep_reg
    direction = [1.0] * len(keep_ret) + [-1.0] * len(keep_reg)

    # inverse-variance-free, equal-weight composite; weights are deliberately not
    # fitted to the outcome, which would make the metric circular
    sub[name] = stats.composite_metric(
        sub, components, direction=direction, group="position_group"
    )
    sub["metric_components"] = ", ".join(components)
    return sub


def metric_validation(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
    *,
    metric_name: str = DEFAULT_METRIC_NAME,
    n_boot: int = 3000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Does the composite metric beat the stopwatch on the chosen outcome?"""
    Z, y, sub, _ = feasibility.build_design(analysis, position_group, outcome)
    Z = Z.apply(feasibility._winsorize)
    if metric_name not in sub.columns:
        return {"error": f"metric {metric_name!r} not built"}

    x = feasibility._winsorize(pd.to_numeric(sub[metric_name], errors="coerce"))
    boot = stats.bootstrap_partial_correlation(x, y, Z, n_boot=n_boot, seed=seed)
    cv = feasibility.cv_incremental_r2(Z, x, y)

    result: Dict[str, Any] = {
        "position_group": position_group,
        "outcome": outcome,
        "metric": metric_name,
        **{k: _num(v) for k, v in boot.items()},
        **{k: _num(v) for k, v in cv.items()},
    }

    # Publish the components the metric was actually averaged from, so that any
    # prose describing the formula can be generated from them rather than
    # hard-coded (a hard-coded formula previously drifted away from the
    # components ``build_metric`` selected). Deduplicated because the selected
    # trait is frequently also its own top supporting trait.
    if "metric_components" in sub.columns and len(sub):
        components = str(sub["metric_components"].iloc[0]).strip()
        if components:
            result["metric_components"] = list(
                dict.fromkeys(c for c in components.split(", ") if c)
            )

    # rank stability of the published top-10 under bootstrap resampling
    try:
        stability = stats.rank_stability(sub, lambda s: s[metric_name])
    except Exception as exc:  # pragma: no cover
        log.warning("rank stability failed: %s", exc)
        stability = {}
    result["rank_stability_top10"] = stability
    return result


def metric_gain_over_stopwatch(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
    *,
    metric_name: str = DEFAULT_METRIC_NAME,
) -> Dict[str, float]:
    """Incremental R^2 of the metric on top of the *best* single traditional test.

    This is the fairest possible comparison and the one a sceptical scout would
    demand: the metric must beat the single best stopwatch number, not merely the
    whole battery at once.
    """
    Z, y, sub, _ = feasibility.build_design(analysis, position_group, outcome)
    if metric_name not in sub.columns:
        return {}
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-2, 3, 24)
    cv = RepeatedKFold(n_splits=5, n_repeats=4, random_state=0)

    def _r2(frame: pd.DataFrame) -> float:
        """CV R^2 of a single-feature model on its own complete cases."""
        if len(frame) < feasibility.MIN_N:
            return np.nan
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        scores = cross_validate(
            model, frame[["x"]].to_numpy(), frame["y"].to_numpy(),
            cv=cv, scoring="r2", error_score=np.nan,
        )["test_score"]
        return float(np.nanmean(scores))

    tests = [c for c in feasibility.POSITION_BASELINE_TESTS.get(position_group, [])
             if c in sub.columns]
    best_test, best_r2 = None, -np.inf
    for test in tests:
        t = feasibility._winsorize(pd.to_numeric(sub[test], errors="coerce"))
        frame = pd.concat([t.rename("x"), y.rename("y")], axis=1).dropna()
        r2 = _r2(frame)
        if np.isfinite(r2) and r2 > best_r2:
            best_test, best_r2 = test, r2

    x = feasibility._winsorize(pd.to_numeric(sub[metric_name], errors="coerce"))
    metric_frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    metric_r2 = _r2(metric_frame)
    return {
        "metric_n": int(len(metric_frame)),
        "best_traditional_test": best_test,
        "best_traditional_r2": _num(best_r2),
        "metric_r2": _num(metric_r2),
        "gain_over_best_test": (
            _num(metric_r2 - best_r2)
            if np.isfinite(best_r2) and np.isfinite(metric_r2)
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Scouting blind spots
# ---------------------------------------------------------------------------
def blind_spot_analysis(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
    *,
    metric_name: str = DEFAULT_METRIC_NAME,
    traditional: Optional[str] = None,
    top: int = 10,
) -> Dict[str, Any]:
    """Players the sensor likes and the stopwatch does not (and the reverse).

    This is the most actionable output a front office can use: a named list of
    prospects whose conventional grade and movement profile disagree, plus the
    in-game outcome that arbitrates between them.
    """
    Z, y, sub, _ = feasibility.build_design(analysis, position_group, outcome)
    if metric_name not in sub.columns or not traditional:
        return {"error": "metric or traditional column missing"}

    quad = stats.quadrant_split(
        sub.assign(**{"__y": y}),
        sub[metric_name],
        pd.to_numeric(sub[traditional], errors="coerce")
        if traditional in sub.columns
        else pd.Series(np.nan, index=sub.index),
        sensor_label="sensor",
        traditional_label="stopwatch",
    )
    if quad.empty:
        return {"error": "no overlapping observations"}

    # does the in-game outcome actually favour the sensor's preferred quadrant?
    sensor_beats = quad[quad["archetype"] == "sensor_beats_stopwatch"]["__y"]
    stopwatch_beats = quad[quad["archetype"] == "stopwatch_beats_sensor"]["__y"]
    confirmed = quad[quad["archetype"] == "confirmed_elite"]["__y"]
    limited = quad[quad["archetype"] == "confirmed_limited"]["__y"]

    # lower is better for some outcomes (e.g. get_off, pressure allowed)
    lower_is_better = outcome in {
        "pressure_allowed_rate", "sack_allowed_rate", "ttp_allowed_mean",
        "peak_pressure_prob_allowed", "get_off_mean", "get_off_p25",
        "time_to_pressure_mean", "epa_allowed_mean",
    }

    def _mean(s: pd.Series) -> Optional[float]:
        return _num(s.mean()) if len(s) else None

    verdict = None
    if _mean(sensor_beats) is not None and _mean(stopwatch_beats) is not None:
        better = (
            _mean(sensor_beats) > _mean(stopwatch_beats)
            if not lower_is_better
            else _mean(sensor_beats) < _mean(stopwatch_beats)
        )
        verdict = (
            "the in-game outcome favours the sensor's preferred quadrant"
            if better
            else "the in-game outcome does NOT favour the sensor's preferred quadrant"
        )

    cols = [c for c in ["nfl_id", "display_name", "draft_overall_pick",
                        "draft_round", "archetype", "sensor_z", "trad_z",
                        "disagreement", "__y"] if c in quad.columns]
    return {
        "n": int(len(quad)),
        "archetype_counts": quad["archetype"].value_counts().to_dict(),
        "outcome_mean_by_archetype": {
            k: _mean(v) for k, v in quad.groupby("archetype")["__y"]
        },
        "outcome": outcome,
        "lower_is_better": lower_is_better,
        "verdict": verdict,
        "hidden_talent": quad[quad["archetype"] == "sensor_beats_stopwatch"]
        .nlargest(top, "disagreement")[cols]
        .to_dict(orient="records"),
        "over_graded": quad[quad["archetype"] == "stopwatch_beats_sensor"]
        .nsmallest(top, "disagreement")[cols]
        .to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_deep_dive(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
    trait: str,
    *,
    metric_name: str = DEFAULT_METRIC_NAME,
    retention_traits: Sequence[str] = (),
    regain_traits: Sequence[str] = (),
    traditional: Optional[str] = None,
) -> Dict[str, Any]:
    """Confirm one linkage, build the metric, and validate both."""
    paths.ensure_dirs()

    linkage = confirm_linkage(analysis, position_group, outcome, trait)
    payload: Dict[str, Any] = {"linkage": linkage.to_dict(),
                               "linkage_sentence": linkage.sentence()}

    if retention_traits or regain_traits:
        enriched = build_metric(
            analysis,
            position_group,
            retention_traits=retention_traits or (trait,),
            regain_traits=regain_traits,
            name=metric_name,
        )
        payload["metric_validation"] = metric_validation(
            enriched, position_group, outcome, metric_name=metric_name
        )
        payload["metric_vs_stopwatch"] = metric_gain_over_stopwatch(
            enriched, position_group, outcome, metric_name=metric_name
        )
        payload["blind_spots"] = blind_spot_analysis(
            enriched, position_group, outcome, metric_name=metric_name,
            traditional=traditional,
        )
        out_table = enriched[[
            c for c in ["nfl_id", "display_name", "position_group",
                        "draft_overall_pick", metric_name] if c in enriched.columns
        ]].sort_values(metric_name, ascending=False)
        out_table.to_csv(paths.REPORT_DIR / f"metric_leaderboard_{position_group}.csv",
                         index=False)
        payload["leaderboard_top"] = out_table.head(15).to_dict(orient="records")

    (paths.REPORT_DIR / "deep_dive.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload

def _pooled_reliability(
    attempts: pd.DataFrame, trait: str, *, n_rep: int = 200, seed: int = 0
) -> Optional[float]:
    """Split-half reliability of a pooled trait across a player's own cuts.

    Each player's detected cuts are split at random into two halves; the trait
    is recomputed on each half and the correlation across players is corrected
    with Spearman-Brown. This bounds how strongly the trait *could* correlate
    with anything, which is an essential honesty check for a null result: a
    trait with 0.3 reliability cannot show a large correlation no matter how
    real the underlying signal is.
    """
    from . import kinematics

    if attempts.empty:
        return None
    rng = np.random.default_rng(seed)
    correlations: List[float] = []
    for _ in range(n_rep):
        shuf = attempts.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        shuf = shuf.assign(_half=np.arange(len(shuf)) % 2)
        halves = []
        for half in (0, 1):
            part = shuf[shuf["_half"] == half]
            pooled = kinematics.pooled_cod_player_features(part)
            col = trait if trait in pooled.columns else None
            if col is None:
                return None
            halves.append(pooled.set_index("nfl_id")[col])
        joined = pd.concat(halves, axis=1).dropna()
        if len(joined) < 20:
            continue
        r = joined.corr().iloc[0, 1]
        if np.isfinite(r) and r > -1:
            correlations.append(2 * r / (1 + r))
    return float(np.mean(correlations)) if correlations else None
