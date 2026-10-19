"""Split-half reliability screening for sensor-derived traits.

Why this module exists
----------------------
A correlation between a sensor trait and an NFL outcome can only be as large as
the trait's *reliability* allows. If a trait measured on half of an athlete's
reps does not predict the same trait measured on the other half, then the trait
is mostly measurement noise, and any correlation with an outcome is attenuated
towards zero by a known factor:

    r_observed  =  r_true * sqrt(reliability_trait * reliability_outcome)

At 10 Hz optical sampling, a change-of-direction "shape" metric such as
``speed_retention = exit_speed / entry_speed`` compounds two noisy speed
estimates inside a 0.2 s window. The competition asks for a metric that a team
can act on, so reliability is not a technicality here -- it is the difference
between a scouting grade and a random number.

Screening traits on reliability *before* looking at any outcome is also the
cheapest available defence against false discovery: it removes the noisiest
candidates from the search space without ever consulting the labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

CACHE_NAME = "trait_reliability.parquet"
REPORT_NAME = "reliability_report.csv"

# traits whose value is a level or a timing, versus those formed as a ratio or a
# difference of two noisy quantities. Used only for interpretation/reporting.
LEVEL_FEATURES = {
    "peak_speed", "mean_speed", "peak_accel", "max_abs_accel", "entry_speed",
    "exit_speed", "cut_min_speed", "turn_angle_deg", "turn_duration_s",
    "turn_rate_deg_s", "max_turn_rate_deg_s", "total_abs_turn_deg",
    "time_to_peak_speed", "duration_s", "path_length", "net_displacement",
}
RATIO_FEATURES = {
    "speed_retention", "speed_loss", "brake_rate", "exit_accel_rate",
    "regain_time_90", "regain_speed_slope", "path_efficiency",
    "cut_radius_proxy", "turn_start_frac",
}

# The pool the reliability screen is computed over: every detected cut from
# every drill that actually contains a change of direction.
POOL_EXCLUDED_DRILLS = {"FORTY_YARD_DASH"}


def trait_kind(feature: str) -> str:
    if feature in RATIO_FEATURES:
        return "ratio/difference"
    if feature in LEVEL_FEATURES:
        return "level/timing"
    return "other"


def split_half_reliability(
    attempts: pd.DataFrame,
    feature: str,
    *,
    n_rep: int = 120,
    min_players: int = 40,
    seed: int = 0,
) -> Dict[str, float]:
    """Reliability of a per-cut feature, by randomly splitting each player's cuts.

    Important detail: the halves are formed from the **raw** feature values. A
    previous version centred the feature within player before splitting, which
    forces the two half-means to be negatively correlated by construction and
    produced reliabilities of about -5.8. The estimate here is the plain
    Spearman-Brown corrected correlation between half-means across players.
    """
    if feature not in attempts.columns:
        return {"reliability": np.nan, "n_cuts": 0, "n_players": 0}

    pool = attempts[
        (pd.to_numeric(attempts.get("n_turns"), errors="coerce").fillna(0) > 0)
        & (~attempts["drill_type"].isin(POOL_EXCLUDED_DRILLS))
    ]
    pool = pool[pd.to_numeric(pool[feature], errors="coerce").notna()]
    if len(pool) < 200:
        return {"reliability": np.nan, "n_cuts": int(len(pool)), "n_players": 0}

    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(n_rep):
        shuffled = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        shuffled = shuffled.assign(_half=np.arange(len(shuffled)) % 2)
        a = shuffled[shuffled["_half"] == 0].groupby("nfl_id")[feature].mean()
        b = shuffled[shuffled["_half"] == 1].groupby("nfl_id")[feature].mean()
        joined = pd.concat([a, b], axis=1).dropna()
        if len(joined) < min_players:
            continue
        r = joined.corr().iloc[0, 1]
        if np.isfinite(r) and r > -0.95:
            estimates.append(2 * r / (1 + r))
    return {
        "reliability": float(np.mean(estimates)) if estimates else np.nan,
        "reliability_sd": float(np.std(estimates)) if estimates else np.nan,
        "n_cuts": int(len(pool)),
        "n_players": int(pool["nfl_id"].nunique()),
        "n_rep": len(estimates),
    }


def build_reliability_table(
    attempts: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
    n_rep: int = 120,
) -> pd.DataFrame:
    """Reliability for every per-cut feature, cached as parquet + CSV."""
    cache = paths.CACHE_DIR / CACHE_NAME
    if cache.exists() and not force:
        cached = _read_parquet(cache)
        if cached is not None:
            return cached

    if features is None:
        features = [
            c
            for c in attempts.select_dtypes(include=[np.number]).columns
            if c not in {"draft_year", "nfl_id", "attempt", "n_frames", "mean_dt"}
        ]

    rows: List[Dict[str, object]] = []
    for feature in sorted(features):
        stats = split_half_reliability(attempts, feature, n_rep=n_rep)
        if not np.isfinite(stats.get("reliability", np.nan)):
            continue
        rows.append({"feature": feature, "kind": trait_kind(feature), **stats})
        log.info("reliability %-24s %.3f", feature, stats["reliability"])

    out = pd.DataFrame(rows).sort_values("reliability", ascending=False)
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(cache, index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet unavailable (%s); reliability cache written as CSV only", exc)
    # the CSV cache twin is what makes this module work without a parquet engine
    out.to_csv(cache.with_suffix(".csv"), index=False)
    out.to_csv(paths.REPORT_DIR / REPORT_NAME, index=False)
    return out


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a parquet cache, or return None when no engine is installed.

    pyarrow is an optional dependency here: the CSV twin written beside every
    parquet cache is authoritative enough for reporting, so a missing engine
    must degrade rather than raise.
    """
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log.info("parquet cache %s unreadable (%s); falling back to CSV", path.name, exc)
        return None


def load_reliability_table() -> Optional[pd.DataFrame]:
    cache = paths.CACHE_DIR / CACHE_NAME
    if cache.exists():
        cached = _read_parquet(cache)
        if cached is not None:
            return cached
    twin = cache.with_suffix(".csv")
    if twin.exists():
        return pd.read_csv(twin)
    csv = paths.REPORT_DIR / REPORT_NAME
    if csv.exists():
        return pd.read_csv(csv)
    return None


def reliability_lookup() -> Dict[str, float]:
    """``feature -> reliability`` mapping, empty when not yet computed."""
    table = load_reliability_table()
    if table is None or table.empty:
        return {}
    return dict(zip(table["feature"], table["reliability"]))


def unreliability_ceiling(reliability: float) -> float:
    """Largest correlation the trait could show with a perfectly measured outcome.

    This is the number that turns an apparent null result into an informative
    one: an observed r of 0.13 from a trait with reliability 0.11 implies a
    disattenuated r of about 0.39, so the honest statement is "underpowered",
    not "no relationship".
    """
    if not np.isfinite(reliability) or reliability <= 0:
        return 0.0
    return float(np.sqrt(reliability))
