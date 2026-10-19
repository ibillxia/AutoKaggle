"""Phase D -- feasibility scan: does a sensor trait add anything *beyond* the stopwatch?

The competition asks for linkages that are "non-obvious". Operationally that
means: a sensor-derived movement trait must explain in-game performance *after*
the traditional Combine numbers and draft capital are already accounted for.
A trait that merely re-expresses the 40-yard dash is not a finding.

The scan is deliberately two-staged so that it is honest *and* fast:

Stage 1 (screen, full sample)
    Partial Spearman correlation between every (sensor trait, in-game outcome)
    pair, residualising both variables on the baseline covariate block
    (traditional Combine tests + draft capital + age + body size).
    p-values are corrected with Benjamini-Hochberg FDR.

Stage 2 (confirmation, cross-validated)
    For the strongest candidates only, fit ridge regressions with repeated
    K-fold cross-validation and report the *incremental* R^2 of adding the
    trait to the baseline block. This is the number quoted in the writeup.

Confounds handled explicitly
----------------------------
* draft capital   -> in the baseline block for every pair
* position        -> by stratifying every analysis by position group
* exposure        -> rate / per-play outcomes preferred over counting stats
* multiple tests  -> BH-FDR in stage 1, and a minimum-N rule
* tracking noise  -> features come from smoothed signals (see kinematics.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import paths, reliability as reliability_mod, schema

log = logging.getLogger(__name__)

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_N = 25                 # minimum complete cases for a pair to be reported
TOP_K_PER_OUTCOME = 12     # stage-2 budget per (position group, outcome)
FDR_ALPHA = 0.10
WINSOR_LIMITS = (0.01, 0.99)
CV_SPLITS = 5
CV_REPEATS = 4
INCREMENTAL_R2_FLOOR = 0.01  # below this, a sensor trait is "just the stopwatch"
# A trait measured on half an athlete's reps must predict the other half at
# least this well. Screening on reliability uses no outcome data at all, so it
# is a free reduction of the search space (see bdb/reliability.py).
RELIABILITY_FLOOR = 0.25

# ---------------------------------------------------------------------------
# The baseline block = "the conventional scouting model without tracking data".
# It must be kept COMPACT: with n in the tens-to-low-hundreds per position
# group, an 11-parameter baseline eats the entire signal budget and makes every
# honest sensor trait look worthless (negative incremental R^2 by construction).
# ---------------------------------------------------------------------------
BASELINE_CORE: List[str] = [
    "log_draft_capital",   # everything teams already knew, in one number
    "age_at_draft",
    "combine_height",
    "combine_weight",
]

# Position-relevant stopwatch tests. A corner's vertical jump is a different
# measurement than a tackle's, so the baseline is not identical across groups.
POSITION_BASELINE_TESTS: Dict[str, List[str]] = {
    "WR": ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"],
    "TE": ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"],
    "DB": ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"],
    "OL": ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "bench_reps"],
    "DL": ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "bench_reps"],
}

# Traditional tests whose opt-out rate is high enough that the *fact of not
# testing* may itself be informative (injured, or protecting a bad number).
MISSINGNESS_FLAG_THRESHOLD = 0.10

# Kept for backwards compatibility / the full-covariate sensitivity analysis.
BASELINE_COVARIATES: List[str] = BASELINE_CORE + sorted(
    {t for tests in POSITION_BASELINE_TESTS.values() for t in tests}
)

# outcome short-name -> source column in the analysis table
OUTCOME_COLUMNS: Dict[str, str] = {
    "sep_mean": "sep_mean",
    "sep_p75": "sep_p75",
    "sep_max": "sep_max",
    "cushion_mean": "cushion_mean",
    "target_rate": "target_rate",
    "yac_mean": "yac_mean",
    "eyac_mean": "eyac_mean",
    "rec_yards_per_target": "rec_yards_per_target",
    "yac_per_reception": "yac_per_reception",
    "get_off_mean": "get_off_mean",
    "get_off_p25": "get_off_p25",
    "pressure_rate": "pressure_allowed_rate",
    "pressure_allowed_rate": "pressure_allowed_rate",
    "sack_allowed_rate": "sack_allowed_rate",
    "sack_rate": "sack_rate",
    "tfl_rate": "tfl_rate",
    "tackle_rate": "tackle_rate",
    "assist_rate": "assist_rate",
    "quick_pressure_rate": "quick_pressure_rate",
    "time_to_pressure_mean": "time_to_pressure_mean",
    "ttp_allowed_mean": "ttp_allowed_mean",
    "peak_pressure_prob_allowed": "peak_pressure_prob_allowed_mean",
    "dropback_duration_mean": "dropback_duration_mean",
    "epa_allowed_mean": "epa_mean",
    "epa_mean": "epa_mean",
    "snaps_per_season": "snaps_per_season",
    "games_started_per_season": "career_games_started_per_season",
    "honors_score": "honors_score",
}

# Hand-curated, football-legible hypotheses: only pair a drill family with the
# outcomes a scout would actually care about. Keeps the scan from becoming a
# 10,000-cell fishing expedition.
_POOLED_PRIORS: Dict[str, List[str]] = {
    "WR": ["sep_mean", "sep_p75", "sep_max", "cushion_mean", "yac_per_reception",
           "target_rate"],
    "TE": ["sep_mean", "sep_p75", "cushion_mean", "yac_per_reception", "target_rate"],
    "DB": ["cushion_mean", "tackle_rate", "assist_rate", "epa_mean"],
    "OL": ["pressure_allowed_rate", "sack_allowed_rate", "ttp_allowed_mean",
           "peak_pressure_prob_allowed"],
    "DL": ["get_off_mean", "get_off_p25", "pressure_rate", "sack_rate",
           "quick_pressure_rate", "time_to_pressure_mean"],
}

POSITION_TRAIT_OUTCOME_PRIORS: Dict[str, Dict[str, List[str]]] = {
    "WR": {
        "pooled": _POOLED_PRIORS["WR"],
        "SKILL_": ["sep_mean", "sep_p75", "cushion_mean", "yac_per_reception", "target_rate"],
        "three_cone": ["sep_mean", "sep_p75", "yac_per_reception"],
        "short_shuttle": ["sep_mean", "sep_p75", "yac_per_reception"],
        "forty": ["sep_mean", "target_rate"],
        "all_drills": ["sep_mean", "sep_p75", "target_rate"],
    },
    "TE": {
        "pooled": _POOLED_PRIORS["TE"],
        "SKILL_": ["sep_mean", "cushion_mean", "yac_per_reception", "target_rate"],
        "three_cone": ["sep_mean", "yac_per_reception"],
        "short_shuttle": ["sep_mean"],
        "all_drills": ["sep_mean", "target_rate"],
    },
    "DB": {
        "pooled": _POOLED_PRIORS["DB"],
        "SKILL_": ["cushion_mean", "tackle_rate", "assist_rate", "epa_mean"],
        "three_cone": ["cushion_mean", "tackle_rate"],
        "short_shuttle": ["cushion_mean", "tackle_rate"],
        "forty": ["cushion_mean"],
        "all_drills": ["cushion_mean", "tackle_rate", "epa_mean"],
    },
    "OL": {
        "pooled": _POOLED_PRIORS["OL"],
        "SKILL_": ["pressure_allowed_rate", "sack_allowed_rate", "ttp_allowed_mean",
                   "peak_pressure_prob_allowed"],
        "three_cone": ["pressure_allowed_rate", "ttp_allowed_mean"],
        "short_shuttle": ["pressure_allowed_rate"],
        "all_drills": ["pressure_allowed_rate", "ttp_allowed_mean"],
    },
    "DL": {
        "pooled": _POOLED_PRIORS["DL"],
        "SKILL_": ["get_off_mean", "get_off_p25", "pressure_rate", "sack_rate",
                   "tfl_rate", "quick_pressure_rate", "time_to_pressure_mean"],
        "forty": ["get_off_mean", "pressure_rate"],
        "three_cone": ["get_off_mean", "pressure_rate"],
        "short_shuttle": ["get_off_mean", "pressure_rate"],
        "all_drills": ["get_off_mean", "pressure_rate", "tfl_rate"],
    },
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[ok] = out
    return q


def _winsorize(s: pd.Series, limits: Tuple[float, float] = WINSOR_LIMITS) -> pd.Series:
    lo, hi = s.quantile(limits[0]), s.quantile(limits[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return s
    return s.clip(lo, hi)


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def _residualize(y: np.ndarray, Z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """OLS residuals of y on [1, Z] with a small ridge for numerical safety."""
    Z = np.column_stack([np.ones(len(Z)), Z])
    gram = Z.T @ Z + ridge * np.eye(Z.shape[1])
    try:
        beta = np.linalg.solve(gram, Z.T @ y)
    except np.linalg.LinAlgError:  # pragma: no cover
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return y - Z @ beta


def partial_spearman(
    x: pd.Series, y: pd.Series, Z: pd.DataFrame
) -> Tuple[float, float, int]:
    """Partial Spearman(x, y | Z) with a two-sided t-test p-value."""
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    n = len(df)
    if n < MIN_N:
        return np.nan, np.nan, n
    k = Z.shape[1]
    if n - k - 2 < 3:
        return np.nan, np.nan, n
    rx = _rank(df["x"]).to_numpy()
    ry = _rank(df["y"]).to_numpy()
    Zv = df[Z.columns].to_numpy(dtype=float)
    ex = _residualize(rx, Zv)
    ey = _residualize(ry, Zv)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(ex, ey)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    dof = n - k - 2
    t = r * np.sqrt(dof / (1 - r ** 2))
    # two-sided p from the t distribution without scipy
    from math import lgamma

    def _betainc(a: float, b: float, x: float, n_iter: int = 200) -> float:
        """Regularised incomplete beta via continued fraction (Lentz)."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(n_iter):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1 - c * d) < 1e-10:
                break
        return front * (f - 1.0)

    x_beta = dof / (dof + t ** 2)
    p = _betainc(dof / 2.0, 0.5, x_beta)
    return r, float(p), n


def cv_incremental_r2(
    X_base: pd.DataFrame,
    x_extra: pd.Series,
    y: pd.Series,
    *,
    splits: int = CV_SPLITS,
    repeats: int = CV_REPEATS,
    seed: int = 0,
) -> Dict[str, float]:
    """Cross-validated R^2 of baseline vs baseline+trait, and the difference."""
    if not _HAVE_SKLEARN:
        return {"r2_base": np.nan, "r2_full": np.nan, "delta_r2": np.nan, "n": 0}

    df = pd.concat([X_base, x_extra.rename("__trait"), y.rename("__y")], axis=1).dropna()
    n = len(df)
    if n < MIN_N:
        return {"r2_base": np.nan, "r2_full": np.nan, "delta_r2": np.nan, "n": n}

    Xb = df[X_base.columns].to_numpy(dtype=float)
    xt = df[["__trait"]].to_numpy(dtype=float)
    yy = df["__y"].to_numpy(dtype=float)

    alphas = np.logspace(-2, 3, 24)
    n = len(yy)

    def _cv_r2(X: np.ndarray) -> float:
        """R^2 on POOLED out-of-fold predictions, averaged over repeats.

        Taking the mean of per-fold R^2 is not safe here. With n in the tens,
        an individual test fold can have near-zero outcome variance, which
        sends that fold's R^2 to a large negative number and makes the
        baseline-vs-full difference explode (observed: an impossible
        incremental R^2 of +3.9 before this fix). Pooling the out-of-fold
        predictions first, then scoring once, is stable and bounded by 1.
        """
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        preds = np.zeros((repeats, n), dtype=float)
        for r in range(repeats):
            cv = KFold(n_splits=splits, shuffle=True, random_state=seed + r)
            try:
                preds[r] = cross_val_predict(model, X, yy, cv=cv)
            except Exception:  # pragma: no cover
                return float("nan")
        pooled = preds.mean(axis=0)
        return float(r2_score(yy, pooled))

    r2_base = _cv_r2(Xb)
    r2_full = _cv_r2(np.column_stack([Xb, xt]))
    r2_trait_only = _cv_r2(xt)

    return {
        "r2_base": r2_base,
        "r2_full": r2_full,
        "delta_r2": r2_full - r2_base,
        "r2_trait_only": r2_trait_only,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Trait discovery
# ---------------------------------------------------------------------------
# ``<drill>__<feature>__<stat>`` -> the drill is the part before the first "__"
_TRAIT_RE = None


_STAT_SUFFIXES = ("mean", "max", "min", "p75", "p25")


def parse_trait(trait: str) -> Tuple[str, str, str]:
    """Split ``<drill_slug>__<feature>_<stat>`` into its three parts.

    Note the single underscore between feature and stat and the double
    underscore between drill and feature -- getting this wrong silently
    disables the change-of-direction guard, which is exactly the bug this
    docstring exists to prevent.
    """
    if trait.startswith("all_drills_"):
        body, drill = trait[len("all_drills_"):], "all_drills"
    elif "__" in trait:
        drill, body = trait.split("__", 1)
    else:
        return "all_drills", trait, ""
    for stat in _STAT_SUFFIXES:
        if body.endswith("_" + stat):
            return drill, body[: -(len(stat) + 1)], stat
    return drill, body, ""


def is_valid_trait(trait: str) -> bool:
    """Reject traits that are undefined for the drill they were computed on.

    A change-of-direction trait extracted from a straight-line sprint is a
    measurement of tracking noise, not of the athlete. Letting those columns
    into the screen produced the single largest spurious effect on the real
    data, so the guard is enforced here rather than left to interpretation.
    """
    drill, feature, _ = parse_trait(trait)
    if feature in schema.NON_TRAIT_FEATURES:
        return False
    if drill == "forty_yard_dash" and feature in schema.COD_ONLY_FEATURES:
        return False
    return True


def list_sensor_traits(
    analysis: pd.DataFrame,
    position_group: str,
) -> List[str]:
    """All sensor-derived columns (``<drill>__<feature>__<stat>``) in the table."""
    prefixes = tuple(
        d.lower().replace("skill_drills_", "skill_") + "__"
        for d in schema.POSITION_TO_DRILLS.get(position_group, [])
    )
    traits = [
        c
        for c in analysis.columns
        if (
            (
                "__" in c
                and (
                    c.lower().startswith(prefixes)
                    or c.startswith("all_drills_")
                )
                and is_valid_trait(c)
            )
            # pooled, player-level change-of-direction traits (100% coverage)
            or c.startswith("pooled_")
        )
        and c != "pooled_retention_asym_low_n"
    ]
    return sorted(set(traits))


def _trait_family(trait: str, position_group: str) -> Optional[str]:
    """Map a trait column back to a canonical drill-family key.

    Canonical keys: ``forty``, ``three_cone``, ``short_shuttle``, ``SKILL_``,
    ``all_drills``. They match the keys of ``POSITION_TRAIT_OUTCOME_PRIORS``.
    """
    low = trait.lower()
    if low.startswith("pooled_"):
        return "pooled"
    if low.startswith("all_drills_"):
        return "all_drills"
    for drill in schema.POSITION_TO_DRILLS.get(position_group, []):
        slug = drill.lower().replace("skill_drills_", "skill_") + "__"
        if low.startswith(slug):
            if drill == "SKILL_DRILLS_" + position_group:
                return "SKILL_"
            if drill == schema.FORTY_YARD_DASH:
                return "forty"
            if drill == schema.THREE_CONE_DRILL:
                return "three_cone"
            if drill == schema.SHORT_SHUTTLE:
                return "short_shuttle"
            return drill.lower()
    return None


def _prior_allows(position_group: str, family: Optional[str], outcome: str) -> Tuple[bool, bool]:
    """Return (allowed, is_confirmatory) for a (family, outcome) pair."""
    priors = POSITION_TRAIT_OUTCOME_PRIORS.get(position_group, {})
    if not priors or family is None or family not in priors:
        return True, False
    return (outcome in set(priors[family])), True


def baseline_columns_for(position_group: str, available: Sequence[str]) -> List[str]:
    """The compact, position-appropriate conventional-scouting baseline."""
    desired = BASELINE_CORE + POSITION_BASELINE_TESTS.get(position_group, [])
    seen, cols = set(), []
    for c in desired:
        if c in available and c not in seen:
            seen.add(c)
            cols.append(c)
    return cols


def impute_baseline(
    Z: pd.DataFrame,
    *,
    flag_threshold: float = MISSINGNESS_FLAG_THRESHOLD,
) -> pd.DataFrame:
    """Median-impute the baseline and flag the heavily-missing tests.

    Complete-case deletion is not an option here: with Combine opt-out rates of
    20-35% on the agility tests, requiring every traditional number would throw
    away more than half of an already small cohort. Imputing the median and
    carrying an explicit "did not test" indicator keeps the sample while making
    the imputation visible to the model (an athlete who skipped the 3-cone is
    not interchangeable with one who ran it).
    """
    Z = Z.apply(pd.to_numeric, errors="coerce")
    out = Z.copy()
    medians = Z.median(numeric_only=True)
    for col in Z.columns:
        miss = Z[col].isna()
        rate = float(miss.mean())
        if rate > flag_threshold:
            out[f"{col}_not_tested"] = miss.astype(float)
        if Z[col].notna().any():
            out[col] = Z[col].fillna(medians[col])
        else:
            out[col] = 0.0
    return out


def build_design(
    analysis: pd.DataFrame,
    position_group: str,
    outcome: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """Return (baseline covariates, outcome, full frame, trait columns)."""
    sub = analysis[analysis["position_group"] == position_group].copy()
    source = OUTCOME_COLUMNS.get(outcome, outcome)
    if source not in sub.columns:
        raise KeyError(f"outcome column {source!r} not present")
    y = pd.to_numeric(sub[source], errors="coerce")
    base_cols = baseline_columns_for(position_group, sub.columns)
    Z = impute_baseline(sub[base_cols])
    return Z, y, sub, list_sensor_traits(sub, position_group)



def trait_base_feature(trait: str) -> Optional[str]:
    """Per-cut feature that a player-level ``trait`` column is aggregated from.

    ``skill_dl__entry_speed_max`` -> ``entry_speed`` and
    ``pooled_retention_mean`` -> ``speed_retention``. Exposed so that reporting
    code resolves reliability exactly the way the screen does, instead of
    failing to find an aggregate column in the per-cut reliability table.
    """
    if trait.startswith("pooled_"):
        base = trait[len("pooled_"):]
        for suffix in ("_mean", "_best", "_worst", "_angle_adj", "_rate"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if base == "retention":
            base = "speed_retention"
        return base
    try:
        _, feature, _ = parse_trait(trait)
    except Exception:  # noqa: BLE001
        return None
    return feature


def trait_reliability(trait: str, lookup: Dict[str, float]) -> Optional[float]:
    """Reliability of the per-cut feature underlying a trait column."""
    if not lookup:
        return None
    base = trait_base_feature(trait)
    if base is None:
        return None
    return lookup.get(base)


def scan_position_group(
    analysis: pd.DataFrame,
    position_group: str,
    *,
    outcomes: Optional[Sequence[str]] = None,
    top_k: int = TOP_K_PER_OUTCOME,
    run_stage2: bool = True,
    reliability_floor: Optional[float] = RELIABILITY_FLOOR,
) -> pd.DataFrame:
    """Stage 1 + stage 2 scan for one position group.

    ``reliability_floor`` removes traits that are not repeatable across an
    athlete's own reps, using no outcome information whatsoever.
    """
    reliability_lookup = reliability_mod.reliability_lookup()
    if reliability_lookup:
        log.info(
            "reliability screening active: %s features known, floor=%.2f",
            len(reliability_lookup), reliability_floor if reliability_floor else 0.0,
        )
    else:
        log.warning("no reliability table found; screening without it")
    outcomes = list(outcomes or schema.POSITION_TO_OUTCOMES.get(position_group, []))
    rows: List[Dict[str, object]] = []

    for outcome in outcomes:
        try:
            Z, y, sub, traits = build_design(analysis, position_group, outcome)
        except KeyError as exc:
            log.warning("skipping outcome %s: %s", outcome, exc)
            continue
        if y.notna().sum() < MIN_N:
            continue

        # winsorise covariates once
        Z = Z.apply(_winsorize)

        for trait in traits:
            family = _trait_family(trait, position_group)
            allowed, confirmatory = _prior_allows(position_group, family, outcome)
            if not allowed:
                continue
            rel = trait_reliability(trait, reliability_lookup)
            if reliability_floor is not None and rel is not None and rel < reliability_floor:
                continue
            x = pd.to_numeric(sub[trait], errors="coerce")
            if x.notna().sum() < MIN_N or x.nunique(dropna=True) < 5:
                continue
            rho, p, n = partial_spearman(_winsorize(x), y, Z)
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "position_group": position_group,
                    "outcome": outcome,
                    "trait": trait,
                    "drill_family": family,
                    "confirmatory": confirmatory,
                    "trait_reliability": rel,
                    "disattenuated_ceiling": (
                        reliability_mod.unreliability_ceiling(rel) if rel is not None else None
                    ),
                    "n": n,
                    "partial_spearman": rho,
                    "abs_rho": abs(rho),
                    "p_value": p,
                }
            )

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    res["q_value"] = np.nan
    for outcome, idx in res.groupby("outcome").groups.items():
        res.loc[idx, "q_value"] = bh_fdr(res.loc[idx, "p_value"].to_numpy())
    res["passes_fdr"] = res["q_value"] < FDR_ALPHA

    if not run_stage2:
        return res.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    # ---------------- stage 2: cross-validated incremental R^2 ------------
    confirm: List[Dict[str, object]] = []
    for outcome, grp in res.groupby("outcome"):
        Z, y, sub, _ = build_design(analysis, position_group, outcome)
        Z = Z.apply(_winsorize)
        candidates = (
            grp.sort_values("abs_rho", ascending=False).head(top_k)["trait"].tolist()
        )
        for trait in candidates:
            x = pd.to_numeric(sub[trait], errors="coerce")
            stats = cv_incremental_r2(Z, _winsorize(x), y)
            row = res[(res["outcome"] == outcome) & (res["trait"] == trait)].iloc[0]
            confirm.append({**row.to_dict(), **stats})
    out = pd.DataFrame(confirm)
    if out.empty:
        return res

    # FDR again, now over the confirmed stage-2 set
    # BH across the *whole* confirmed set for this position group, not within
    # each outcome: controlling per outcome would let a family of weak
    # candidates share a lenient correction.
    out["cv_q_value"] = bh_fdr(np.nan_to_num(out["p_value"].to_numpy(), nan=1.0))
    # A stage-2 claim requires the stage-1 screen to have passed too. Requiring
    # only the (few-candidate, therefore lenient) stage-2 correction produced
    # "survivors" whose stage-1 q was 0.95 -- the two stages disagreeing is a
    # sign of an unstable pair, not of a finding.
    out["adds_beyond_stopwatch"] = (
        (out["delta_r2"] > INCREMENTAL_R2_FLOOR)
        & (out["cv_q_value"] < FDR_ALPHA)
        & (out["q_value"] < FDR_ALPHA)
    )
    return out.sort_values("delta_r2", ascending=False).reset_index(drop=True)


def scan_all(
    analysis: pd.DataFrame,
    *,
    position_groups: Optional[Iterable[str]] = None,
    run_stage2: bool = True,
    reliability_floor: Optional[float] = RELIABILITY_FLOOR,
) -> pd.DataFrame:
    groups = list(position_groups or schema.FOCUS_POSITION_GROUPS)
    frames = []
    for group in groups:
        log.info("feasibility scan: %s", group)
        res = scan_position_group(
            analysis, group, run_stage2=run_stage2, reliability_floor=reliability_floor
        )
        if not res.empty:
            frames.append(res)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Trait redundancy pruning + reliability
# ---------------------------------------------------------------------------
def prune_redundant_traits(
    analysis: pd.DataFrame,
    position_group: str,
    *,
    threshold: float = 0.95,
) -> List[str]:
    """Drop near-duplicate sensor traits (|r| > threshold), keeping the first."""
    traits = list_sensor_traits(analysis, position_group)
    sub = analysis[analysis["position_group"] == position_group]
    if len(sub) < 3:
        return traits
    X = sub[traits].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().sum() >= MIN_N]
    if X.shape[1] < 2:
        return list(X.columns)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return [c for c in X.columns if c not in drop]


def split_half_reliability(
    attempts: pd.DataFrame,
    *,
    feature: str = "speed_retention",
    drill_type: str = "THREE_CONE_DRILL",
) -> Dict[str, float]:
    """Odd-vs-even attempt reliability of a per-attempt feature.

    A trait that is not even reliable within the same player's own Combine
    attempts cannot plausibly predict NFL performance; this bounds how much
    signal any correlation can carry (attenuation).
    """
    sub = attempts[attempts["drill_type"] == drill_type].copy()
    if sub.empty or feature not in sub.columns:
        return {"reliability": np.nan, "n_players": 0}
    sub["parity"] = sub["attempt"].astype("Int64") % 2
    odd = sub[sub["parity"] == 1].groupby("nfl_id")[feature].mean()
    even = sub[sub["parity"] == 0].groupby("nfl_id")[feature].mean()
    joined = pd.concat([odd.rename("odd"), even.rename("even")], axis=1).dropna()
    if len(joined) < 10:
        return {"reliability": np.nan, "n_players": len(joined)}
    r = float(joined.corr().iloc[0, 1])
    # Spearman-Brown correction to full-length reliability
    rho_full = 2 * r / (1 + r) if r > -1 else np.nan
    return {"reliability": rho_full, "split_half_r": r, "n_players": len(joined)}
