"""Combine 10 Hz tracking -> interpretable movement kinematics.

This module is the analytical heart of the solution. It converts the raw
per-frame optical tracking of a Combine drill attempt into a small set of
*coach-readable* movement traits, with the explicit goal of separating the
"stopwatch" component of a drill (total time, which is already in
``combine_results.csv``) from the *shape* of the movement (how the athlete
braked, how much speed survived the cut, and how quickly they re-accelerated).

Feature families
----------------
``speed``       straight-line capability: peak / mean speed, acceleration.
``path``        geometry: path length, net displacement, path efficiency.
``turn``        change-of-direction: entry/exit speed, retention, angles.
``brake``       deceleration into a break.
``reaccel``     acceleration out of a break, and time to regain entry speed.
``asym``        left-vs-right asymmetry of the above.

Two design rules are enforced throughout:

1. 10 Hz optical speed is noisy. Every differentiation happens on a
   Savitzky-Golay smoothed signal, never on the raw ``a`` column.
2. Every feature is defined per *attempt* (``event_id``). Player-level
   aggregation is a separate, explicit step so that "best attempt" and "mean
   attempt" can both be analysed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - scipy is a declared dependency but stay defensive
    from scipy.signal import savgol_filter

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# Tunable parameters (single source of truth; surfaced in the writeup appendix)
# ---------------------------------------------------------------------------
SMOOTH_WINDOW_S = 0.30          # Savitzky-Golay window in seconds (3 frames @10Hz)
SMOOTH_POLYORDER = 2
TURN_RATE_THRESHOLD_DEG_S = 60.0  # angular velocity that counts as "a cut"
TURN_MIN_ANGLE_DEG = 25.0         # ignore micro-corrections below this total angle
TURN_MERGE_GAP_S = 0.20           # merge turn segments separated by < 0.2 s
SPEED_WINDOW_S = 0.20             # window used for entry / exit speed means
BRAKE_LOOKBACK_S = 0.50           # how far before the cut to look for braking
REACCEL_LOOKAHEAD_S = 0.60        # how far after the cut to look for re-acceleration
REGAIN_TARGET = 0.90              # "regained" threshold relative to entry speed
REGAIN_HORIZON_S = 2.00           # give up after this long


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _smooth(values: np.ndarray, dt: float, window_s: float = SMOOTH_WINDOW_S) -> np.ndarray:
    """Savitzky-Golay smoothing with a hard fallback to a centred moving mean."""
    n = len(values)
    if n < 3:
        return values.astype(float)
    if _HAVE_SCIPY:
        fps = 1.0 / max(dt, 1e-3)
        win = int(round(window_s * fps))
        win = max(3, win)
        if win % 2 == 0:
            win += 1
        win = min(win, n if n % 2 == 1 else n - 1)
        if win >= 3:
            try:
                return savgol_filter(values.astype(float), win, SMOOTH_POLYORDER)
            except Exception:  # pragma: no cover
                pass
    k = max(1, int(round(window_s * 10 / 2)))
    return (
        pd.Series(values, dtype="float64")
        .rolling(2 * k + 1, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def angular_velocity_deg_s(dir_deg: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    """Signed turn rate in deg/s, robust to the 0/360 wrap-around.

    Returns an array of length ``len(dir_deg)`` aligned to the frame index
    (first element is back-filled from the second).
    """
    if len(dir_deg) < 2:
        return np.zeros(len(dir_deg))
    rad = np.deg2rad(np.asarray(dir_deg, dtype=float))
    delta = np.diff(rad)
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi)
    dt = np.diff(t_sec)
    dt = np.where(dt <= 0, np.nan, dt)
    omega = np.degrees(delta) / dt
    omega = np.concatenate([[omega[0]], omega])
    return np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)


def _segments_from_mask(mask: np.ndarray, t_sec: np.ndarray, gap_s: float) -> List[Tuple[int, int]]:
    """Contiguous True runs in ``mask``, merging runs closer than ``gap_s``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, splits)
    merged: List[Tuple[int, int]] = []
    for run in runs:
        start, end = int(run[0]), int(run[-1])
        if merged and (t_sec[start] - t_sec[merged[-1][1]]) <= gap_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _window_mean(values: np.ndarray, t: np.ndarray, lo: float, hi: float) -> float:
    sel = (t >= lo) & (t <= hi)
    if not np.any(sel):
        return float("nan")
    return float(np.nanmean(values[sel]))


def _regain_time(
    speed: np.ndarray,
    t: np.ndarray,
    start_idx: int,
    entry_speed: float,
    horizon_s: float = REGAIN_HORIZON_S,
) -> float:
    """Seconds after ``start_idx`` until speed >= REGAIN_TARGET * entry_speed.

    ``nan`` means the athlete never got back up to speed inside the horizon
    (i.e. a censored observation, not a missing one).
    """
    if not np.isfinite(entry_speed) or entry_speed <= 0:
        return float("nan")
    target = REGAIN_TARGET * entry_speed
    t0 = t[start_idx]
    sel = (t >= t0) & (t <= t0 + horizon_s)
    if not np.any(sel):
        return float("nan")
    hits = np.flatnonzero(sel & (speed >= target))
    if hits.size == 0:
        return float("nan")
    return float(t[hits[0]] - t0)


# ---------------------------------------------------------------------------
# Per-attempt feature extraction
# ---------------------------------------------------------------------------
ATTEMPT_FEATURE_COLUMNS: List[str] = [
    # identity
    "draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt",
    # coverage
    "n_frames", "duration_s", "mean_dt",
    # speed
    "peak_speed", "mean_speed", "peak_accel", "peak_decel", "max_abs_accel",
    "time_to_peak_speed", "distance",
    # path geometry
    "path_length", "net_displacement", "path_efficiency",
    # turning
    "total_abs_turn_deg", "max_turn_rate_deg_s", "n_turns",
    # main cut
    "turn_angle_deg", "turn_duration_s", "turn_rate_deg_s",
    "entry_speed", "exit_speed", "cut_min_speed",
    "speed_retention", "speed_loss", "brake_rate", "exit_accel_rate",
    "regain_time_90", "regain_speed_slope", "cut_radius_proxy",
    # signed direction of the main cut (for asymmetry analysis)
    "turn_sign", "turn_start_frac",
]


@dataclass
class AttemptKinematics:
    """Kinematic summary of a single drill attempt."""

    meta: Dict[str, object] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    # smoothed series kept for figure generation / diagnostics
    series: Optional[Dict[str, np.ndarray]] = None


def extract_attempt_kinematics(
    frames: pd.DataFrame,
    *,
    keep_series: bool = False,
) -> AttemptKinematics:
    """Compute kinematics for one attempt (one ``event_id`` / one player)."""
    frames = frames.sort_values("time")
    t_raw = frames["time"].to_numpy(dtype="datetime64[ns]")
    t = (t_raw - t_raw[0]).astype("float64") / 1e9

    # de-duplicate timestamps (optical systems occasionally emit ties)
    if len(t) > 1:
        keep = np.concatenate([[True], np.diff(t) > 0])
        t = t[keep]
        frames = frames.iloc[keep]

    x = frames["x"].to_numpy(dtype=float)
    y = frames["y"].to_numpy(dtype=float)
    s_raw = frames["s"].to_numpy(dtype=float)
    dir_deg = frames["dir"].to_numpy(dtype=float)
    dis_raw = frames["dis"].to_numpy(dtype=float)

    meta = {
        "draft_year": frames["draft_year"].iloc[0],
        "event_id": frames["event_id"].iloc[0],
        "nfl_id": frames["nfl_id"].iloc[0],
        "drill_type": frames["drill_type"].iloc[0],
        "drill_name": frames["drill_name"].iloc[0],
        "attempt": frames["attempt"].iloc[0],
    }

    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.1

    feats: Dict[str, float] = {"n_frames": n, "mean_dt": dt}
    feats["duration_s"] = float(t[-1] - t[0]) if n > 1 else 0.0

    if n < 5:
        for col in ATTEMPT_FEATURE_COLUMNS:
            feats.setdefault(col, np.nan)
        return AttemptKinematics(meta=meta, features=feats)

    # ---- smoothing -------------------------------------------------------
    s = _smooth(s_raw, dt)
    seconds = np.clip(s, 0, None)
    # acceleration from the smoothed speed signal (never from the raw `a`)
    accel = np.gradient(seconds, t, edge_order=1)

    feats["peak_speed"] = float(np.nanmax(seconds))
    feats["mean_speed"] = float(np.nanmean(seconds))
    feats["peak_accel"] = float(np.nanmax(accel))
    feats["peak_decel"] = float(np.nanmin(accel))
    feats["max_abs_accel"] = float(np.nanmax(np.abs(accel)))
    feats["time_to_peak_speed"] = float(t[int(np.nanargmax(seconds))])

    # ---- path geometry ---------------------------------------------------
    step = np.hypot(np.diff(x), np.diff(y))
    path_length = float(np.nansum(step))
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    feats["path_length"] = path_length
    feats["net_displacement"] = net
    # prefer the tracked `dis` sum when it is plausible, else the chord sum
    dis_sum = float(np.nansum(dis_raw))
    feats["distance"] = dis_sum if dis_sum > 0 else path_length
    feats["path_efficiency"] = net / path_length if path_length > 0 else np.nan

    # ---- turning ---------------------------------------------------------
    omega = angular_velocity_deg_s(dir_deg, t)
    omega_s = _smooth(omega, dt, window_s=0.2)
    feats["total_abs_turn_deg"] = float(np.nansum(np.abs(omega_s) * dt))
    feats["max_turn_rate_deg_s"] = float(np.nanmax(np.abs(omega_s)))

    mask = np.abs(omega_s) > TURN_RATE_THRESHOLD_DEG_S
    runs = _segments_from_mask(mask, t, TURN_MERGE_GAP_S)

    turns: List[Dict[str, float]] = []
    for start, end in runs:
        # uniform 10 Hz sampling: total signed rotation = mean rate * elapsed
        signed_angle = float(np.nansum(omega_s[start : end + 1]) * dt)
        angle = abs(signed_angle)
        if angle < TURN_MIN_ANGLE_DEG:
            continue
        t_start, t_end = t[start], t[end]
        entry = _window_mean(seconds, t, t_start - SPEED_WINDOW_S, t_start)
        exit_ = _window_mean(seconds, t, t_end, t_end + SPEED_WINDOW_S)
        seg = slice(start, end + 1)
        cut_min = float(np.nanmin(seconds[seg]))
        brake = float(np.nanmin(accel[(t >= t_start - BRAKE_LOOKBACK_S) & (t <= t_start)]))
        reacc = float(np.nanmax(accel[(t >= t_end) & (t <= t_end + REACCEL_LOOKAHEAD_S)]))
        retention = exit_ / entry if (np.isfinite(entry) and entry > 0) else np.nan
        turns.append(
            {
                "signed_angle": signed_angle,
                "angle": angle,
                "start_idx": start,
                "t_start": t_start,
                "t_end": t_end,
                "duration": max(t_end - t_start, dt),
                "entry_speed": entry,
                "exit_speed": exit_,
                "cut_min_speed": cut_min,
                "speed_retention": retention,
                "speed_loss": entry - exit_ if np.isfinite(entry) and np.isfinite(exit_) else np.nan,
                "brake_rate": brake,
                "exit_accel_rate": reacc,
                "regain_time_90": _regain_time(seconds, t, end, entry),
                "turn_rate": angle / max(t_end - t_start, dt),
            }
        )

    feats["n_turns"] = len(turns)
    if turns:
        main = max(turns, key=lambda d: d["angle"])
        feats["turn_angle_deg"] = main["angle"]
        feats["turn_sign"] = float(np.sign(main["signed_angle"]))
        feats["turn_duration_s"] = main["duration"]
        feats["turn_rate_deg_s"] = main["turn_rate"]
        feats["entry_speed"] = main["entry_speed"]
        feats["exit_speed"] = main["exit_speed"]
        feats["cut_min_speed"] = main["cut_min_speed"]
        feats["speed_retention"] = main["speed_retention"]
        feats["speed_loss"] = main["speed_loss"]
        feats["brake_rate"] = main["brake_rate"]
        feats["exit_accel_rate"] = main["exit_accel_rate"]
        feats["regain_time_90"] = main["regain_time_90"]
        feats["turn_start_frac"] = main["t_start"] / feats["duration_s"] if feats["duration_s"] else np.nan
        # how fast they climb back out of the cut's speed minimum
        post = (t >= main["t_end"]) & (t <= main["t_end"] + REACCEL_LOOKAHEAD_S)
        if np.count_nonzero(post) >= 2:
            tt = t[post]
            ss = seconds[post]
            slope = np.polyfit(tt - tt[0], ss, 1)[0] if len(tt) > 1 else np.nan
        else:
            slope = np.nan
        feats["regain_speed_slope"] = float(slope) if slope is not None else np.nan
        feats["cut_radius_proxy"] = (
            main["entry_speed"] / np.deg2rad(main["turn_rate"])
            if main["turn_rate"] > 0
            else np.nan
        )
    else:
        for col in (
            "turn_angle_deg", "turn_sign", "turn_duration_s", "turn_rate_deg_s",
            "entry_speed", "exit_speed", "cut_min_speed", "speed_retention",
            "speed_loss", "brake_rate", "exit_accel_rate", "regain_time_90",
            "turn_start_frac", "regain_speed_slope", "cut_radius_proxy",
        ):
            feats[col] = np.nan

    series = None
    if keep_series:
        series = {"t": t, "speed": seconds, "accel": accel, "omega": omega_s, "x": x, "y": y}
    return AttemptKinematics(meta=meta, features=feats, series=series)


# ---------------------------------------------------------------------------
# Table level driver
# ---------------------------------------------------------------------------
def extract_all_attempt_features(
    tracking: pd.DataFrame,
    *,
    min_frames: int = 10,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Apply :func:`extract_attempt_kinematics` to every attempt in the table."""
    tracking = tracking.dropna(subset=["nfl_id"])
    keys = ["draft_year", "event_id", "nfl_id"]
    grouped = tracking.groupby(keys, sort=False, observed=True)

    rows: List[Dict[str, object]] = []
    for i, (_, frames) in enumerate(grouped):
        if len(frames) < min_frames:
            continue
        kin = extract_attempt_kinematics(frames)
        rows.append({**kin.meta, **kin.features})
        if progress_every and (i + 1) % progress_every == 0:
            log.info("kinematics: %s attempts processed", i + 1)

    out = pd.DataFrame(rows)
    log.info("extracted kinematics for %s attempts", len(out))
    return out


# ---------------------------------------------------------------------------
# Player-level aggregation
# ---------------------------------------------------------------------------
def aggregate_attempts_to_player(
    attempts: pd.DataFrame,
    *,
    agg: Tuple[str, ...] = ("mean", "max", "min"),
) -> pd.DataFrame:
    """Collapse attempt-level kinematics to one row per (nfl_id, drill_type).

    Deliberately keeps ``best_*`` (max) and ``worst_*`` (min) alongside the mean
    because Combine athletes get several attempts and the coaching question is
    usually "what is this player capable of at his best?".
    """
    numeric = attempts.select_dtypes(include=[np.number]).columns.tolist()
    # identity / bookkeeping columns are NOT movement traits. Including them
    # would let "which attempt number this was" masquerade as a kinematic signal.
    drop = {"draft_year", "nfl_id", "attempt"}
    numeric = [c for c in numeric if c not in drop]

    grouped = attempts.groupby(["nfl_id", "drill_type"], observed=True)[numeric]
    wide = grouped.agg(list(agg))

    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [f"{drill}_{stat}" for drill, stat in wide.columns]
    else:  # pragma: no cover - defensive
        wide.columns = [f"{c}_{agg[0]}" for c in wide.columns]

    wide = wide.reset_index()
    wide["n_attempts"] = (
        attempts.groupby(["nfl_id", "drill_type"], observed=True)
        .size()
        .reset_index(drop=True)
        .to_numpy()
    )
    wide["drill_type"] = wide["drill_type"].astype(str)
    wide["drill_slug"] = (
        wide["drill_type"].str.replace("SKILL_DRILLS_", "SKILL_", regex=False).str.lower()
    )
    return wide


def pivot_player_features(player_drill: pd.DataFrame) -> pd.DataFrame:
    """Wide table, one row per player, columns ``<drill>__<feature>__<stat>``."""
    value_cols = [
        c
        for c in player_drill.columns
        if c not in {"nfl_id", "drill_type", "drill_slug", "n_attempts"}
    ]
    melted = player_drill.melt(
        id_vars=["nfl_id", "drill_slug"],
        value_vars=value_cols,
        var_name="feature",
        value_name="value",
    )
    melted["column"] = (
        melted["drill_slug"].astype(str) + "__" + melted["feature"].astype(str)
    )
    wide = melted.pivot_table(
        index="nfl_id", columns="column", values="value", aggfunc="first"
    )
    wide.columns.name = None
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Pooled, player-level change-of-direction traits
# ---------------------------------------------------------------------------
# Rationale (this is the analytical pivot of the whole solution):
# drill-specific COD traits are only computable for the players who ran *that*
# drill, and on this dataset 47-80% of prospects opt out of the 3-cone and
# short shuttle. That fragments a 510-player cohort into cells of n = 26-47 and
# destroys statistical power for exactly the traits the competition is about.
#
# A change-of-direction trait is a property of the *athlete*, not of the drill.
# Pooling every detected cut across every drill that actually contains a turn
# restores coverage to 510/510 (median 9 cuts per player) while remaining
# conceptually cleaner: the question becomes "how does this player move through
# cuts", which is what a position coach asks.
#
# The 40-yard dash is excluded from the pool: it is a straight line, and cuts
# "detected" in it are tracking noise (see schema.COD_ONLY_FEATURES).

COD_POOL_EXCLUDED_DRILLS = {"FORTY_YARD_DASH"}

POOLED_TRAIT_COLUMNS: List[str] = [
    "pooled_n_cuts",
    "pooled_n_drills",
    "pooled_turn_angle_mean",
    "pooled_entry_speed_mean",
    "pooled_exit_speed_mean",
    "pooled_retention_mean",
    "pooled_retention_best",
    "pooled_retention_worst",
    "pooled_retention_angle_adj",
    "pooled_speed_loss_mean",
    "pooled_brake_mean",
    "pooled_exit_accel_mean",
    "pooled_regain_time_mean",
    "pooled_regain_censored_rate",
    "pooled_cut_radius_mean",
    "pooled_retention_asym",
]


def pooled_cod_player_features(
    attempts: pd.DataFrame,
    *,
    horizon_s: float = REGAIN_HORIZON_S,
) -> pd.DataFrame:
    """One row per player: change-of-direction traits pooled across all drills.

    Two corrections are applied that matter for validity:

    * **Angle adjustment.** Speed retention falls as the cut gets sharper, so a
      raw pooled mean partly measures which drills a player happened to run.
      ``pooled_retention_angle_adj`` is the residual of retention on turn angle,
      fitted across the pool, so that a player is compared to a player who cut
      by the same amount.
    * **Censoring.** ``regain_time_90`` is ``NaN`` when the athlete never got
      back to 90% of entry speed inside the horizon. That is a censored
      observation, not a missing one, so it is imputed at the horizon and the
      censoring rate is carried as its own feature.
    """
    if attempts.empty:
        return pd.DataFrame(columns=["nfl_id"] + POOLED_TRAIT_COLUMNS)

    cod = attempts[
        (pd.to_numeric(attempts.get("n_turns"), errors="coerce").fillna(0) > 0)
        & (~attempts["drill_type"].isin(COD_POOL_EXCLUDED_DRILLS))
        & attempts["speed_retention"].notna()
    ].copy()

    if cod.empty:
        return pd.DataFrame(columns=["nfl_id"] + POOLED_TRAIT_COLUMNS)

    # angle-adjusted retention: remove the mechanical angle dependence
    ang = pd.to_numeric(cod["turn_angle_deg"], errors="coerce")
    ret = pd.to_numeric(cod["speed_retention"], errors="coerce")
    ok = ang.notna() & ret.notna()
    if ok.sum() > 30 and ang[ok].std() > 0:
        slope, intercept = np.polyfit(ang[ok], ret[ok], 1)
        cod["retention_angle_adj"] = ret - (intercept + slope * ang)
    else:
        cod["retention_angle_adj"] = ret - ret.mean()

    # censored regain time -> impute at the horizon
    regain = pd.to_numeric(cod["regain_time_90"], errors="coerce")
    cod["regain_censored"] = regain.isna().astype(float)
    cod["regain_filled"] = regain.fillna(horizon_s)

    # signed asymmetry: does the athlete retain more speed cutting one way?
    if "turn_sign" in cod.columns:
        left = (
            cod[cod["turn_sign"] > 0].groupby("nfl_id")["speed_retention"].mean()
        )
        right = (
            cod[cod["turn_sign"] < 0].groupby("nfl_id")["speed_retention"].mean()
        )
        asym = (left - right).rename("pooled_retention_asym")
    else:  # pragma: no cover
        asym = pd.Series(dtype=float, name="pooled_retention_asym")

    g = cod.groupby("nfl_id", observed=True)
    out = pd.DataFrame(
        {
            "pooled_n_cuts": g.size(),
            "pooled_n_drills": g["drill_type"].nunique(),
            "pooled_turn_angle_mean": g["turn_angle_deg"].mean(),
            "pooled_entry_speed_mean": g["entry_speed"].mean(),
            "pooled_exit_speed_mean": g["exit_speed"].mean(),
            "pooled_retention_mean": g["speed_retention"].mean(),
            "pooled_retention_best": g["speed_retention"].max(),
            "pooled_retention_worst": g["speed_retention"].min(),
            "pooled_retention_angle_adj": g["retention_angle_adj"].mean(),
            "pooled_speed_loss_mean": g["speed_loss"].mean(),
            "pooled_brake_mean": g["brake_rate"].mean(),
            "pooled_exit_accel_mean": g["exit_accel_rate"].mean(),
            "pooled_regain_time_mean": g["regain_filled"].mean(),
            "pooled_regain_censored_rate": g["regain_censored"].mean(),
            "pooled_cut_radius_mean": g["cut_radius_proxy"].mean(),
        }
    ).reset_index()
    out = out.merge(asym.reset_index(), on="nfl_id", how="left")
    return out


def player_level_combine_features(attempts: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: attempt table -> per-player wide feature table."""
    per_drill = aggregate_attempts_to_player(attempts)
    wide = pivot_player_features(per_drill)
    # position-agnostic "best across all drills" summaries
    overall = (
        attempts.groupby("nfl_id", observed=True)
        .agg(
            all_drills_peak_speed=("peak_speed", "max"),
            all_drills_mean_speed=("mean_speed", "mean"),
            all_drills_peak_accel=("peak_accel", "max"),
            all_drills_best_retention=("speed_retention", "max"),
            all_drills_mean_retention=("speed_retention", "mean"),
            all_drills_min_regain=("regain_time_90", "min"),
            all_drills_turns=("n_turns", "sum"),
            all_drills_path_eff=("path_efficiency", "mean"),
        )
        .reset_index()
    )
    merged = wide.merge(overall, on="nfl_id", how="outer")
    pooled = pooled_cod_player_features(attempts)
    return merged.merge(pooled, on="nfl_id", how="outer")


# ---------------------------------------------------------------------------
# Validation against the official electronic timings
# ---------------------------------------------------------------------------
def validate_against_official_timings(
    attempts: pd.DataFrame,
    combine_results: pd.DataFrame,
) -> pd.DataFrame:
    """Sanity-check sensor durations against the stopwatch numbers.

    This is the credibility figure for the whole analysis: if sensor-derived
    drill duration does not track the official electronic time, no downstream
    sensor feature can be trusted.
    """
    res = combine_results[["nfl_id", "forty", "three_cone", "short_shuttle", "ten_yd_split"]]
    checks = [
        ("THREE_CONE_DRILL", "three_cone"),
        ("SHORT_SHUTTLE", "short_shuttle"),
        ("FORTY_YARD_DASH", "forty"),
    ]
    out: List[pd.DataFrame] = []
    for drill, official in checks:
        sub = attempts[attempts["drill_type"] == drill][
            ["nfl_id", "attempt", "duration_s", "peak_speed"]
        ].merge(res, on="nfl_id", how="inner")
        sub = sub.dropna(subset=["duration_s", official])
        if sub.empty:
            continue
        # attempts can include partial captures; keep the attempt whose duration
        # is closest to the official time (that is the one that was timed)
        sub["abs_err"] = (sub["duration_s"] - sub[official]).abs()
        best = sub.sort_values("abs_err").groupby("nfl_id", as_index=False).first()
        best["drill_type"] = drill
        best["official_metric"] = official
        out.append(best)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
