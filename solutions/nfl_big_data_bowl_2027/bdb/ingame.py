"""In-game outcomes: what the prospect actually did on an NFL field.

Two complementary sources are used, and the fact that they are *independent*
is what makes the linkage analysis credible:

``player_play.csv``
    Vendor/Next Gen Stats derived per-play metrics (separation, cushion,
    pass-rush get-off, time to pressure, EPA, ...). Rich but model-derived.

``game_tracking_YYYY.csv``
    The raw 10 Hz optical tracking. We re-derive movement quantities ourselves
    (peak speed, distance, burst immediately after the snap) so the headline
    claim can be checked without trusting the vendor columns.

Outcome naming convention
-------------------------
``<metric>_mean``, ``_p75``, ``_p25``, ``_rate`` etc. Every aggregate is
computed *within position group* only at analysis time; here we simply produce
one row per ``nfl_id``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import io, paths, schema

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# player_play.csv -> per-player outcome table
# ---------------------------------------------------------------------------
# metric -> (source column, aggregation recipe)
_MEAN_METRICS = {
    "sep": "separation_at_pass_forward",
    "cushion": "cushion",
    "get_off": "player_get_off",
    "time_to_pressure": "time_to_pressure",
    "time_to_qb_hurry": "time_to_qb_hurry",
    "peak_pressure_prob_allowed": "peak_pressure_probability_allowed",
    "ttp_allowed": "time_to_pressure_allowed",
    "dropback_duration": "dropback_duration",
    "rush_encountered": "pass_rushers_encountered",
    "yac": "yards_after_catch",
    "eyac": "expected_yards_after_catch",
    "pass_length": "pass_length",
    "epa": "expected_points_added",
    "rec_yards": "rec_yards",
}

_RATE_METRICS = {
    "target": "target",
    "sack": "sack",
    "tackle": "tackle",
    "assist": "assist",
    "tfl": "tackle_for_loss",
    "ff": "caused_forced_fumble",
    "fumble_rec": "recovered_fumble",
    "pressure_allowed": "pressure_allowed",
    "sack_allowed": "sack_allowed",
    "blitz": "blitzing",
    "quick_pressure": "quick_pressure",
}

PLAYER_PLAY_USECOLS: List[str] = (
    ["game_id", "play_id", "nfl_id", "lined_up_position"]
    + sorted(set(_MEAN_METRICS.values()) | set(_RATE_METRICS.values()))
)


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce")


def build_player_play_outcomes(
    *,
    min_plays: int = 20,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate the 316k-row player-play table to one row per prospect.

    ``min_plays`` is deliberately *not* used to drop rows -- players with very
    few snaps carry real information (they are the draft busts) -- but the play
    count is returned so the analysis can weight or filter explicitly.
    """
    cols = list(usecols) if usecols else PLAYER_PLAY_USECOLS
    # keep only columns that actually exist in this release of the file, so the
    # pipeline degrades gracefully rather than crashing on a schema tweak
    available = set(io.csv_columns(paths.PLAYER_PLAY))
    cols = [c for c in cols if c in available]
    missing_requested = sorted(set(PLAYER_PLAY_USECOLS) - available)
    if missing_requested:
        log.warning("player_play is missing expected metric columns: %s", missing_requested)
    pp = io.load_player_play(usecols=cols)
    games = io.load_games()[["game_id", "season", "season_type", "week"]]
    pp = pp.merge(games, on="game_id", how="left")

    for col in _MEAN_METRICS.values():
        if col in pp.columns:
            pp[col] = _to_numeric(pp[col])
    for col in _RATE_METRICS.values():
        if col in pp.columns:
            pp[col] = _to_numeric(pp[col])

    g = pp.groupby("nfl_id", observed=True)

    out = pd.DataFrame(index=g.size().index)
    out["pp_plays"] = g.size()
    out["pp_seasons"] = g["season"].nunique()
    out["pp_games"] = g["game_id"].nunique()

    # ---- central tendency + upper/lower tail for each continuous metric ----
    for short, col in _MEAN_METRICS.items():
        if col not in pp.columns:
            continue
        series = pp[col]
        out[f"{short}_mean"] = g[col].mean()
        out[f"{short}_n"] = g[col].count()
        out[f"{short}_p75"] = g[col].quantile(0.75)
        out[f"{short}_p25"] = g[col].quantile(0.25)
        out[f"{short}_max"] = g[col].max()
        # opportunities: fraction of the player's snaps on which the metric exists
        out[f"{short}_avail_rate"] = g[col].apply(lambda s: float(s.notna().mean()))

    # ---- rate metrics ------------------------------------------------------
    for short, col in _RATE_METRICS.items():
        if col not in pp.columns:
            continue
        s = pp[col]
        out[f"{short}_rate"] = g[col].mean()
        out[f"{short}_per100"] = g[col].mean() * 100.0

    # ---- composite outcomes that matter to NFL evaluators -----------------
    if {"expected_points_added"}.issubset(pp.columns):
        out["epa_per_play"] = out["epa_mean"]
    if {"separation_at_pass_forward", "target"}.issubset(pp.columns):
        # receiving yards per target = the canonical efficiency outcome
        ry = pp["rec_yards"] if "rec_yards" in pp.columns else None
        if ry is not None:
            tg = pp[pp["target"] == 1].groupby("nfl_id", observed=True)
            out["rec_yards_per_target"] = (
                tg["rec_yards"].sum() / tg.size().replace(0, np.nan)
            )
            out["yac_per_reception"] = (
                tg["yards_after_catch"].sum() / tg.size().replace(0, np.nan)
            )

    out = out.reset_index()
    out.attrs["min_plays_hint"] = min_plays
    return out


# ---------------------------------------------------------------------------
# game_tracking_YYYY.csv -> raw-motion per-player features
# ---------------------------------------------------------------------------
_GAME_TRACKING_USECOLS = [
    "game_id", "play_id", "nfl_id", "time", "s", "a", "dis", "event",
]
# window (seconds) after the snap used for the "first-step burst" proxy
SNAP_BURST_WINDOW_S = 1.5
PLAY_FEATURE_COLUMNS = [
    "peak_speed", "mean_speed", "peak_accel", "peak_decel",
    "distance", "high_speed_distance", "n_frames",
    "snap_first_frame_s", "snap_speed_0p5", "snap_burst_peak_accel",
    "snap_burst_peak_speed", "snap_burst_distance", "snap_burst_n",
]


def _snap_times_for(chunk: pd.DataFrame) -> pd.DataFrame:
    """Earliest ``ball_snap`` timestamp per (game_id, play_id)."""
    snaps = chunk[chunk["event"] == "ball_snap"][["game_id", "play_id", "time"]]
    if snaps.empty:
        return pd.DataFrame(columns=["game_id", "play_id", "snap_time"])
    return (
        snaps.groupby(["game_id", "play_id"], observed=True)["time"]
        .min()
        .reset_index()
        .rename(columns={"time": "snap_time"})
    )


def _play_features_for_chunk(
    chunk: pd.DataFrame, snap_times: pd.DataFrame
) -> pd.DataFrame:
    keys = ["game_id", "play_id", "nfl_id"]
    chunk = chunk.copy()
    chunk["dis"] = pd.to_numeric(chunk["dis"], errors="coerce").fillna(0.0)
    chunk["s"] = pd.to_numeric(chunk["s"], errors="coerce")
    chunk["a"] = pd.to_numeric(chunk["a"], errors="coerce")
    chunk["high_speed_dis"] = chunk["dis"].where(chunk["s"] >= 5.0, 0.0)

    g = chunk.groupby(keys, sort=False, observed=True)
    feats = pd.DataFrame(
        {
            "peak_speed": g["s"].max(),
            "mean_speed": g["s"].mean(),
            "peak_accel": g["a"].max(),
            "peak_decel": g["a"].min(),
            "distance": g["dis"].sum(),
            "high_speed_distance": g["high_speed_dis"].sum(),
            "n_frames": g.size(),
        }
    )

    # ---- post-snap burst -------------------------------------------------
    merged = chunk.merge(snap_times, on=["game_id", "play_id"], how="left")
    merged["rel"] = (
        merged["time"] - merged["snap_time"]
    ).dt.total_seconds()
    post = merged[(merged["rel"] >= 0) & (merged["rel"] <= SNAP_BURST_WINDOW_S)]
    if not post.empty:
        pg = post.groupby(keys, sort=False, observed=True)
        burst = pd.DataFrame(
            {
                "snap_burst_peak_accel": pg["a"].max(),
                "snap_burst_peak_speed": pg["s"].max(),
                "snap_burst_distance": pg["dis"].sum(),
                "snap_burst_n": pg.size(),
            }
        )
        # speed ~0.5 s after the snap: frame closest to rel == 0.5
        near = post.assign(_d=(post["rel"] - 0.5).abs())
        idx = near.groupby(keys, sort=False, observed=True)["_d"].idxmin()
        s05 = (
            near.loc[idx, keys + ["s"]]
            .rename(columns={"s": "snap_speed_0p5"})
            .set_index(keys)
        )
        # first tracked frame relative to the snap
        first = post.groupby(keys, sort=False, observed=True)["rel"].min().rename(
            "snap_first_frame_s"
        )
        feats = feats.join([burst, s05, first], how="left")

    return feats.reset_index()


def build_game_tracking_features(
    *,
    seasons: Sequence[int] = (2023, 2024, 2025),
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Per-player raw in-game motion features, streamed season by season.

    Returns one row per ``nfl_id`` with peak/typical values across all of that
    player's tracked plays, plus a play-level table cached separately.
    """
    play_frames: List[pd.DataFrame] = []
    for season in seasons:
        log.info("game tracking: season %s", season)
        total = 0
        for chunk in io.iter_game_tracking(
            season, chunksize=chunksize, usecols=_GAME_TRACKING_USECOLS
        ):
            chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
            chunk = chunk.dropna(subset=["time", "nfl_id"])
            snaps = _snap_times_for(chunk)
            feats = _play_features_for_chunk(chunk, snaps)
            feats["season"] = season
            play_frames.append(feats)
            total += len(chunk)
        log.info("  %s frames -> %s player-plays", total, sum(len(f) for f in play_frames))

    if not play_frames:
        return pd.DataFrame(columns=["nfl_id"])

    play = pd.concat(play_frames, ignore_index=True)
    io.write_parquet_cache(play, paths.CACHE_INGAME_PLAY_FEATURES)

    g = play.groupby("nfl_id", observed=True)
    out = pd.DataFrame(
        {
            "gt_plays": g.size(),
            "gt_games": g["game_id"].nunique(),
            "gt_peak_speed": g["peak_speed"].max(),
            "gt_p90_peak_speed": g["peak_speed"].quantile(0.90),
            "gt_mean_speed": g["mean_speed"].mean(),
            "gt_peak_accel": g["peak_accel"].max(),
            "gt_peak_decel": g["peak_decel"].min(),
            "gt_distance_per_play": g["distance"].mean(),
            "gt_distance_total": g["distance"].sum(),
            "gt_high_speed_distance": g["high_speed_distance"].sum(),
            "gt_n_frames": g["n_frames"].sum(),
            "gt_snap_burst_peak_accel": g["snap_burst_peak_accel"].quantile(0.90),
            "gt_snap_burst_peak_speed": g["snap_burst_peak_speed"].quantile(0.90),
            "gt_snap_speed_0p5": g["snap_speed_0p5"].median(),
            "gt_snap_first_frame_s": g["snap_first_frame_s"].median(),
        }
    )
    out["gt_high_speed_share"] = out["gt_high_speed_distance"] / out[
        "gt_distance_total"
    ].replace(0, np.nan)
    out["gt_high_speed_distance_per_play"] = out["gt_high_speed_distance"] / out[
        "gt_plays"
    ].replace(0, np.nan)
    return out.reset_index()


def build_ingame_player_features(*, force: bool = False) -> pd.DataFrame:
    """Combine vendor outcomes (player_play) with our own tracking features."""

    def _build() -> pd.DataFrame:
        pp = build_player_play_outcomes()
        gt = build_game_tracking_features()
        return pp.merge(gt, on="nfl_id", how="outer")

    return io.cached(paths.CACHE_INGAME_PLAYER_FEATURES, _build, force=force)
