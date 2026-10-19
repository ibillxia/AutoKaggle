"""Reusable statistics for small-N, confounded sports analytics.

Everything here is written for the sample sizes this competition actually has:
tens of players per position group, heavy confounding by draft capital, and
outcome variables that are themselves estimated (Next Gen Stats model outputs).
That rules out most of the standard toolbox and forces:

* **bootstrap** rather than asymptotic confidence intervals,
* **robust (HC3) standard errors** rather than classical OLS ones,
* **leave-one-out / influence diagnostics** as a first-class output rather than
  an afterthought, because with n ~ 60 a single player can create or destroy an
  apparent effect,
* **explicit reporting of `n`** on every number that leaves this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank / residual helpers
# ---------------------------------------------------------------------------
def rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def ols_residuals(y: np.ndarray, Z: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Residuals of ``y`` on ``[1, Z]`` using a tiny ridge for stability."""
    if Z.ndim == 1:
        Z = Z[:, None]
    design = np.column_stack([np.ones(len(Z)), Z])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    try:
        beta = np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:  # pragma: no cover
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


@dataclass
class OLSPointEstimate:
    """OLS fit with heteroskedasticity-robust (HC3) inference."""

    coefficients: pd.Series
    robust_se: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    ci_low: pd.Series
    ci_high: pd.Series
    r_squared: float
    adj_r_squared: float
    n: int
    k: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.coefficients,
                "robust_se": self.robust_se,
                "t": self.t_stats,
                "p": self.p_values,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )

    def summary(self) -> str:
        lines = [
            f"OLS with HC3 robust SE   n={self.n}  k={self.k}  "
            f"R2={self.r_squared:.3f}  adjR2={self.adj_r_squared:.3f}",
        ]
        frame = self.to_frame()
        for name, row in frame.iterrows():
            lines.append(
                f"  {name:<28} {row['coef']:+.4f} "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  p={row['p']:.4f}"
            )
        return "\n".join(lines)


def _t_sf(t: np.ndarray | float, dof: int) -> np.ndarray:
    """Two-sided p-value of Student-t, via the incomplete beta function."""
    from math import lgamma

    t = np.atleast_1d(np.asarray(t, dtype=float))
    x = dof / (dof + t ** 2)

    def betainc(a: float, b: float, xx: np.ndarray) -> np.ndarray:
        out = np.empty_like(xx)
        for i, v in enumerate(xx):
            if v <= 0:
                out[i] = 0.0
                continue
            if v >= 1:
                out[i] = 1.0
                continue
            lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
            front = np.exp(np.log(v) * a + np.log(1 - v) * b - lbeta) / a
            f, c, d = 1.0, 1.0, 0.0
            for j in range(300):
                m = j // 2
                if j == 0:
                    num = 1.0
                elif j % 2 == 0:
                    num = (m * (b - m) * v) / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    num = -((a + m) * (a + b + m) * v) / ((a + 2 * m) * (a + 2 * m + 1))
                d = 1.0 + num * d
                d = 1e-30 if abs(d) < 1e-30 else d
                d = 1.0 / d
                c = 1.0 + num / c
                c = 1e-30 if abs(c) < 1e-30 else c
                f *= c * d
                if abs(1 - c * d) < 1e-12:
                    break
            out[i] = front * (f - 1.0)
        return out

    return betainc(dof / 2.0, 0.5, x)


def ols_robust(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    add_intercept: bool = True,
) -> OLSPointEstimate:
    """OLS with HC3 heteroskedasticity-robust standard errors."""
    frame = pd.concat([X, y.rename("__y")], axis=1).dropna()
    n = len(frame)
    if n <= X.shape[1] + 2:
        raise ValueError(f"not enough complete cases (n={n}) for {X.shape[1]} predictors")
    design = frame[X.columns].to_numpy(dtype=float)
    if add_intercept:
        design = np.column_stack([np.ones(n), design])
        names = ["intercept"] + list(X.columns)
    else:  # pragma: no cover
        names = list(X.columns)
    yy = frame["__y"].to_numpy(dtype=float)

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ yy
    resid = yy - design @ beta
    k = design.shape[1]
    dof = max(n - k, 1)

    # HC3: (X'X)^-1 X' diag(e_i^2 / (1-h_ii)^2) X (X'X)^-1
    hat = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    hat = np.clip(hat, 0, 1 - 1e-8)
    weights = (resid ** 2) / (1.0 - hat) ** 2
    meat = design.T @ (design * weights[:, None])
    cov = xtx_inv @ meat @ xtx_inv

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / np.where(se > 0, se, np.nan)
    p = _t_sf(t, dof)
    # 95% CI via the t quantile approximated by the normal for dof > 30, else
    # a conservative 2.1 multiplier; exact quantiles are not worth a scipy dep here
    mult = np.where(dof > 60, 1.96, np.where(dof > 30, 2.04, 2.15))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1) if np.isfinite(r2) else np.nan

    return OLSPointEstimate(
        coefficients=pd.Series(beta, index=names),
        robust_se=pd.Series(se, index=names),
        t_stats=pd.Series(t, index=names),
        p_values=pd.Series(p, index=names),
        ci_low=pd.Series(beta - mult * se, index=names),
        ci_high=pd.Series(beta + mult * se, index=names),
        r_squared=float(r2),
        adj_r_squared=float(adj),
        n=n,
        k=k,
    )


# ---------------------------------------------------------------------------
# Partial correlation + bootstrap
# ---------------------------------------------------------------------------
def partial_correlation(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    *,
    rank: bool = True,
) -> Tuple[float, int]:
    """Partial correlation of x and y given Z (Spearman if ``rank``)."""
    frame = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    n = len(frame)
    if n < 5:
        return np.nan, n
    xv = rank_pct(frame["x"]).to_numpy() if rank else frame["x"].to_numpy()
    yv = rank_pct(frame["y"]).to_numpy() if rank else frame["y"].to_numpy()
    Zv = frame[Z.columns].to_numpy(dtype=float)
    ex = ols_residuals(xv, Zv)
    ey = ols_residuals(yv, Zv)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, n
    return float(np.corrcoef(ex, ey)[0, 1]), n


def bootstrap_partial_correlation(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    rank: bool = True,
    cluster: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the partial correlation.

    If ``cluster`` is supplied (e.g. draft class), whole clusters are resampled,
    which is the right thing to do when players within a season share shocks.
    """
    frame = pd.concat(
        [x.rename("x"), y.rename("y"), Z], axis=1
    ).dropna()
    if cluster is not None:
        frame = frame.join(cluster.rename("__cluster")).dropna(subset=["__cluster"])
    n = len(frame)
    if n < 8:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n": n, "n_boot": 0, "p_boot": np.nan}

    est, _ = partial_correlation(frame["x"], frame["y"], frame[Z.columns], rank=rank)

    rng = np.random.default_rng(seed)
    stats: List[float] = []
    if cluster is not None:
        groups = frame.groupby("__cluster").indices
        keys = list(groups)
        for _ in range(n_boot):
            pick = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in pick])
            sub = frame.iloc[idx]
            r, _ = partial_correlation(sub["x"], sub["y"], sub[Z.columns], rank=rank)
            if np.isfinite(r):
                stats.append(r)
    else:
        pos = np.arange(n)
        for _ in range(n_boot):
            idx = rng.choice(pos, size=n, replace=True)
            sub = frame.iloc[idx]
            r, _ = partial_correlation(sub["x"], sub["y"], sub[Z.columns], rank=rank)
            if np.isfinite(r):
                stats.append(r)

    if not stats:
        return {"estimate": est, "ci_low": np.nan, "ci_high": np.nan,
                "n": n, "n_boot": 0, "p_boot": np.nan}
    arr = np.asarray(stats)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    # bootstrap two-sided p: fraction of resamples on the wrong side of zero
    p_boot = 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())
    return {
        "estimate": float(est),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "n_boot": len(arr),
        "p_boot": float(min(p_boot, 1.0)),
    }


# ---------------------------------------------------------------------------
# Influence / robustness
# ---------------------------------------------------------------------------
def leave_one_out_stability(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    *,
    rank: bool = True,
) -> pd.DataFrame:
    """Recompute the partial correlation with each observation removed."""
    frame = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    base, _ = partial_correlation(frame["x"], frame["y"], Z, rank=rank)
    rows = []
    for i in range(len(frame)):
        sub = frame.drop(frame.index[i])
        r, _ = partial_correlation(sub["x"], sub["y"], Z, rank=rank)
        rows.append({"dropped_index": frame.index[i], "partial_r": r, "base_r": base,
                     "delta": r - base if np.isfinite(r) else np.nan})
    out = pd.DataFrame(rows)
    # attach the player identity when available
    if "nfl_id" in frame.columns:
        out["nfl_id"] = frame["nfl_id"].to_numpy()
    return out


def influence_summary(loo: pd.DataFrame) -> Dict[str, float]:
    if loo.empty:
        return {"max_abs_delta": np.nan, "n_flips_sign": 0, "base_r": np.nan}
    base = float(loo["base_r"].iloc[0])
    flips = int(((loo["partial_r"] > 0) != (base > 0)).sum())
    return {
        "base_r": base,
        "max_abs_delta": float(loo["delta"].abs().max()),
        "min_partial_r": float(loo["partial_r"].min()),
        "max_partial_r": float(loo["partial_r"].max()),
        "n_flips_sign": flips,
        "n": int(len(loo)),
    }


# ---------------------------------------------------------------------------
# Metric construction
# ---------------------------------------------------------------------------
def zscore_within(s: pd.Series, groups: Optional[pd.Series] = None) -> pd.Series:
    """Standardise, optionally within position group, ignoring missing values."""
    if groups is None:
        return (s - s.mean()) / s.std(ddof=1)
    g = s.groupby(groups)
    return (s - g.transform("mean")) / g.transform("std")


def composite_metric(
    frame: pd.DataFrame,
    components: Sequence[str],
    *,
    direction: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    group: Optional[str] = None,
) -> pd.Series:
    """Percentile-average composite, standardised within position group.

    ``direction`` carries ``+1`` for "higher is better" and ``-1`` for the
    reverse, so that a metric such as "Exit Velocity Ratio" (higher better) can
    be combined with "time to regain speed" (lower better).
    """
    if weights is None:
        weights = [1.0] * len(components)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    groups = frame[group] if group and group in frame.columns else None

    total = None
    for comp, sgn, weight in zip(components, direction, w):
        if comp not in frame.columns:
            raise KeyError(f"composite component missing: {comp}")
        z = zscore_within(pd.to_numeric(frame[comp], errors="coerce"), groups) * sgn
        total = z * weight if total is None else total + z * weight
    return total


def metric_leaderboard(
    frame: pd.DataFrame,
    metric: pd.Series,
    *,
    position_group: Optional[str] = None,
    top: int = 15,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rank players by the metric, optionally inside one position group."""
    sub = frame.copy()
    sub["_metric"] = metric
    if position_group and "position_group" in sub.columns:
        sub = sub[sub["position_group"] == position_group]
    cols = ["nfl_id"]
    for c in (columns or ["display_name", "position_group", "draft_overall_pick"]):
        if c in sub.columns:
            cols.append(c)
    cols.append("_metric")
    return (
        sub.dropna(subset=["_metric"])
        .sort_values("_metric", ascending=False)[cols]
        .head(top)
        .reset_index(drop=True)
    )


def quadrant_split(
    frame: pd.DataFrame,
    sensor_metric: pd.Series,
    traditional_metric: pd.Series,
    *,
    sensor_label: str = "sensor",
    traditional_label: str = "stopwatch",
) -> pd.DataFrame:
    """Classify players into the four scouting archetypes.

    The interesting quadrants are the two *disagreements*: players the sensor
    likes but the stopwatch does not (candidate hidden talent) and vice versa
    (candidate over-drafted profile).
    """
    sub = frame.copy()
    sub["_sensor"] = pd.to_numeric(sensor_metric, errors="coerce")
    sub["_trad"] = pd.to_numeric(traditional_metric, errors="coerce")
    sub = sub.dropna(subset=["_sensor", "_trad"])
    if sub.empty:
        return sub
    s_med = sub["_sensor"].median()
    t_med = sub["_trad"].median()

    def _label(row: pd.Series) -> str:
        hi_s, hi_t = row["_sensor"] >= s_med, row["_trad"] >= t_med
        if hi_s and hi_t:
            return "confirmed_elite"
        if hi_s and not hi_t:
            return f"{sensor_label}_beats_{traditional_label}"
        if not hi_s and hi_t:
            return f"{traditional_label}_beats_{sensor_label}"
        return "confirmed_limited"

    sub["archetype"] = sub.apply(_label, axis=1)
    sub["sensor_z"] = (sub["_sensor"] - sub["_sensor"].mean()) / sub["_sensor"].std(ddof=1)
    sub["trad_z"] = (sub["_trad"] - sub["_trad"].mean()) / sub["_trad"].std(ddof=1)
    sub["disagreement"] = sub["sensor_z"] - sub["trad_z"]
    return sub


def rank_stability(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    n_boot: int = 500,
    seed: int = 0,
    top: int = 10,
) -> Dict[str, float]:
    """How stable is the top-k of a metric under bootstrap resampling?

    A metric whose leaderboard reshuffles completely under resampling is not
    something a front office can act on, however significant its aggregate
    correlation looks.
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(len(frame))
    tops: List[set] = []
    for _ in range(n_boot):
        idx = rng.choice(pos, size=len(frame), replace=True)
        sample = frame.iloc[idx].copy()
        try:
            m = metric_fn(sample)
        except Exception:  # pragma: no cover
            continue
        ranked = m.dropna().sort_values(ascending=False)
        tops.append(set(ranked.head(top).index))
    if not tops:
        return {"mean_jaccard": np.nan, "n_boot": 0}
    jaccards = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            a, b = tops[i], tops[j]
            if not a and not b:
                continue
            jaccards.append(len(a & b) / max(len(a | b), 1))
    return {
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
        "median_jaccard": float(np.median(jaccards)) if jaccards else np.nan,
        "n_boot": len(tops),
    }
