"""Schema-faithful synthetic competition data.

Purpose
-------
The real competition data is 2.2 GiB behind Kaggle authentication. This module
lets the entire pipeline be validated end-to-end *before* the real data lands,
and -- more importantly -- it embeds a **known ground-truth signal** so that the
feasibility scan can be tested for both sensitivity and specificity:

    * ``forty`` and ``three_cone`` (traditional stopwatch numbers) depend on a
      latent straight-line speed trait ``v_max`` and a latent agility trait.
    * A *sensor-only* trait -- the fraction of entry speed retained through the
      sharpest cut (``speed_retention``) -- depends on a latent
      change-of-direction skill ``cod_skill`` that is **not** observable in the
      traditional numbers alone.
    * The in-game outcome (receiving separation / pass-rush get-off) depends on
      BOTH the traditional numbers AND on ``cod_skill``.

A correct pipeline must therefore recover ``speed_retention`` as a trait that
"adds beyond the stopwatch", while also correctly reporting that most other
sensor traits do not. Generating data where the answer is known is the cheapest
possible defence against a pipeline that finds spurious structure.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import paths, schema

log = logging.getLogger(__name__)

RNG_SEED = 7

POSITION_WEIGHTS = {
    "WR": 0.20, "DB": 0.22, "OL": 0.20, "DL": 0.20, "TE": 0.08, "LB": 0.10,
}
POSITION_SPECS = {
    #            height(in)  weight(lb)  v_max(yd/s)  cod_skill  forty(s)
    "WR": (72.0, 2.0, 195, 12, 9.3, 0.70, 4.50),
    "DB": (71.5, 2.0, 195, 12, 9.2, 0.72, 4.52),
    "TE": (76.0, 2.0, 250, 14, 8.4, 0.60, 4.72),
    "OL": (77.0, 2.0, 315, 18, 7.2, 0.50, 5.25),
    "DL": (75.5, 2.0, 290, 18, 7.9, 0.55, 4.95),
    "LB": (73.5, 2.0, 235, 12, 8.6, 0.63, 4.68),
}
SKILL_DRILL_FOR_POSITION = {
    "WR": "SKILL_DRILLS_WR",
    "TE": "SKILL_DRILLS_TE",
    "DB": "SKILL_DRILLS_DB",
    "OL": "SKILL_DRILLS_OL",
    "DL": "SKILL_DRILLS_DL",
}
# waypoint paths (yards). Every drill ends near its start for the shuttle/cone.
DRILL_PATHS: Dict[str, List[Tuple[float, float]]] = {
    "FORTY_YARD_DASH": [(0, 0), (40, 0)],
    "THREE_CONE_DRILL": [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)],
    "SHORT_SHUTTLE": [(0, 0), (5, 0), (10, 0), (5, 0), (0, 0)],
    "SKILL_DRILLS_WR": [(0, 0), (8, 0), (8, 6), (2, 6)],
    "SKILL_DRILLS_TE": [(0, 0), (7, 0), (7, 5), (2, 5)],
    "SKILL_DRILLS_DB": [(0, 0), (-6, 0), (-6, -6), (0, -6)],
    "SKILL_DRILLS_OL": [(0, 0), (3, 0), (3, 3), (0, 3)],
    "SKILL_DRILLS_DL": [(0, 0), (-3, 0), (-3, 3), (0, 3)],
}
DT = 0.10
A_MAX = 6.0  # yd/s^2
MAX_TURN_RATE = 320.0  # deg/s


# ---------------------------------------------------------------------------
# Drill simulation
# ---------------------------------------------------------------------------
def _simulate_path(
    waypoints: List[Tuple[float, float]],
    v_max: float,
    cod_skill: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Integrate a first-order speed/heading model through the waypoints."""
    pos = np.array(waypoints[0], dtype=float)
    wp_idx = 1
    heading = 0.0
    speed = 0.0
    t = 0.0
    frames = []

    # better change-of-direction skill => less braking into and quicker out of cuts
    brake_factor = 1.25 - 0.5 * cod_skill
    turn_rate_max = MAX_TURN_RATE * (0.75 + 0.5 * cod_skill)

    max_steps = int(30.0 / DT)
    for _ in range(max_steps):
        target = np.array(waypoints[wp_idx], dtype=float)
        delta = target - pos
        dist_to_wp = float(np.hypot(*delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)

        # required turn angle -> how much speed must be shed
        turn_deg = abs((bearing - heading + 180.0) % 360.0 - 180.0)
        is_last = wp_idx == len(waypoints) - 1
        if turn_deg > 20.0:
            v_target = v_max * max(0.35, 1.0 - brake_factor * turn_deg / 180.0)
        elif is_last:
            v_target = max(0.0, min(v_max, dist_to_wp / 0.6))
        else:
            v_target = v_max

        dv = np.clip(v_target - speed, -A_MAX * DT, A_MAX * DT)
        speed = float(max(0.0, speed + dv))

        # rotate heading toward the bearing with a bounded turn rate
        diff = (bearing - heading + 180.0) % 360.0 - 180.0
        step = float(np.clip(diff, -turn_rate_max * DT, turn_rate_max * DT))
        heading = float((heading + step) % 360.0)

        rad = np.radians(heading)
        new_pos = pos + np.array([np.cos(rad), np.sin(rad)]) * speed * DT
        step_len = float(np.hypot(*(new_pos - pos)))

        frames.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "s": speed,
                "dir": heading,
                "dis": step_len,
            }
        )
        pos = new_pos
        t += DT

        if dist_to_wp < 0.35 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
        elif dist_to_wp < 0.35 and wp_idx == len(waypoints) - 1:
            break

    df = pd.DataFrame(frames)
    if df.empty:
        return df
    # acceleration is the derivative of the smoothed speed, mirroring the raw feed
    df["a"] = np.gradient(df["s"].to_numpy(), DT)
    df["time"] = pd.to_datetime("2025-03-02T18:00:00") + pd.to_timedelta(
        np.arange(len(df)) * DT, unit="s"
    )
    return df


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_synthetic_competition(
    out_dir: Path | None = None,
    *,
    n_players: int = 510,
    seed: int = RNG_SEED,
    small: bool = False,
    signal_strength: float = 0.75,
) -> Path:
    """Write all nine competition CSVs (with a known embedded signal)."""
    out_dir = Path(out_dir) if out_dir else paths.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if small:
        n_players = min(n_players, 60)

    # ---------------------------------------------------------- players ---
    groups = list(POSITION_WEIGHTS)
    weights = np.array([POSITION_WEIGHTS[g] for g in groups], dtype=float)
    weights = weights / weights.sum()
    pos_group = rng.choice(groups, size=n_players, p=weights)

    # latent traits -------------------------------------------------------
    v_max_z = rng.normal(size=n_players)
    cod_z = rng.normal(size=n_players)
    prod_z = rng.normal(size=n_players)

    rows = []
    draft_year = rng.choice([2023, 2024, 2025], size=n_players,
                            p=[0.34, 0.33, 0.33])
    # draft capital correlates with the latent traits (that is the confound!)
    talent = 0.62 * v_max_z + 0.45 * cod_z + 0.35 * prod_z + 0.5 * rng.normal(size=n_players)
    order = np.argsort(-talent)
    overall_pick = np.full(n_players, np.nan)
    n_drafted = int(n_players * 0.72)
    overall_pick[order[:n_drafted]] = np.arange(1, n_drafted + 1)

    for i in range(n_players):
        g = pos_group[i]
        h_mu, h_sd, w_mu, w_sd, _, _, _ = POSITION_SPECS[g]
        rows.append(
            {
                "nfl_id": 50_000 + i,
                "display_name": f"Prospect {i:03d}",
                "draft_year": int(draft_year[i]),
                "nfl_position": {
                    "WR": "WR", "TE": "TE", "DB": "CB", "OL": "T",
                    "DL": "DE", "LB": "OLB",
                }[g],
                "birth_date": (
                    pd.Timestamp("2000-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 1200)))
                ).strftime("%Y-%m-%d"),
                "college_name": f"University {int(rng.integers(0, 130))}",
                "college_conference": rng.choice(
                    ["SEC", "Big Ten", "ACC", "Big 12", "Pac-12"]
                ),
                "draft_round": np.nan,
                "draft_pick_within_round": np.nan,
                "draft_overall_pick": float(overall_pick[i])
                if np.isfinite(overall_pick[i])
                else np.nan,
                "_v_max_z": v_max_z[i],
                "_cod_z": cod_z[i],
                "_prod_z": prod_z[i],
                "_height": rng.normal(h_mu, h_sd),
                "_weight": rng.normal(w_mu, w_sd),
            }
        )
    players = pd.DataFrame(rows)
    for i in players.index:
        pick = players.at[i, "draft_overall_pick"]
        if np.isfinite(pick):
            players.at[i, "draft_round"] = float(min(7, int((pick - 1) // 32) + 1))
            players.at[i, "draft_pick_within_round"] = float(int((pick - 1) % 32) + 1)

    # ------------------------------------------------- combine_results ---
    res_rows = []
    for _, p in players.iterrows():
        g = schema.POSITION_GROUP.get(p["nfl_position"], "OTHER")
        _, _, _, _, v_mu, _, forty_mu = POSITION_SPECS[g]
        v_max = v_mu + 0.55 * p["_v_max_z"]
        forty = forty_mu - 0.22 * p["_v_max_z"] + rng.normal(0, 0.06)
        ten_split = forty * 0.335 + rng.normal(0, 0.012)
        # the 3-cone blends straight-line speed with change-of-direction skill
        three_cone = (
            7.45 - 0.28 * p["_cod_z"] - 0.12 * p["_v_max_z"] + rng.normal(0, 0.10)
        )
        shuttle = 4.35 - 0.14 * p["_cod_z"] - 0.05 * p["_v_max_z"] + rng.normal(0, 0.07)
        res_rows.append(
            {
                "draft_year": p["draft_year"],
                "nfl_id": p["nfl_id"],
                "combine_position": g,
                "combine_height": round(p["_height"], 3),
                "combine_weight": int(round(p["_weight"])),
                "hand_size": round(float(rng.normal(9.4, 0.6)), 2),
                "arm_length": round(float(rng.normal(32.5, 1.4)), 2),
                "wing_span": round(p["_height"] + float(rng.normal(4.4, 1.0)), 2),
                "ten_yd_split": round(float(ten_split), 2) if rng.random() > 0.05 else np.nan,
                "forty": round(float(forty), 2) if rng.random() > 0.04 else np.nan,
                "vertical": round(float(34.0 - 1.6 * p["_weight"] / 50 + rng.normal(0, 2.4)), 1)
                if rng.random() > 0.12 else np.nan,
                "broad_jump": round(float(118.0 - 3.0 * (p["_weight"] - 200) / 50
                                          + rng.normal(0, 7.0)), 1)
                if rng.random() > 0.18 else np.nan,
                "three_cone": round(float(three_cone), 2) if rng.random() > 0.25 else np.nan,
                "short_shuttle": round(float(shuttle), 2) if rng.random() > 0.30 else np.nan,
                "bench_reps": float(int(rng.integers(8, 32))) if rng.random() > 0.35 else np.nan,
                "ngs_athleticism_score": int(np.clip(50 + 20 * p["_v_max_z"]
                                                     + 18 * p["_cod_z"], 1, 99)),
                "ngs_college_production_score": round(float(60 + 18 * p["_prod_z"]), 1),
                "ngs_final_score": round(float(58 + 14 * p["_v_max_z"]
                                               + 10 * p["_cod_z"] + 12 * p["_prod_z"]), 1),
            }
        )
    combine_results = pd.DataFrame(res_rows)

    # ------------------------------ combine_tracking (10 Hz sensor feed) --
    tracking_frames = []
    drills = ["FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"]
    if small:
        drills = ["FORTY_YARD_DASH", "THREE_CONE_DRILL"]
    for _, p in players.iterrows():
        g = schema.POSITION_GROUP.get(p["nfl_position"], "OTHER")
        _, _, _, _, v_mu, _, _ = POSITION_SPECS.get(g, POSITION_SPECS["WR"])
        v_max = float(np.clip(v_mu + 0.55 * p["_v_max_z"], 5.0, 11.5))
        cod = float(np.clip(0.55 + 0.16 * p["_cod_z"], 0.15, 0.98))
        player_drills = list(drills)
        if g in SKILL_DRILL_FOR_POSITION and not small:
            player_drills.append(SKILL_DRILL_FOR_POSITION[g])
        for drill in player_drills:
            n_attempts = 2 if drill in ("FORTY_YARD_DASH",) else int(rng.integers(2, 4))
            for attempt in range(1, n_attempts + 1):
                cod_j = float(np.clip(cod + rng.normal(0, 0.05), 0.1, 0.99))
                v_j = float(np.clip(v_max + rng.normal(0, 0.15), 5.0, 11.5))
                path = DRILL_PATHS[drill]
                frames = _simulate_path(path, v_j, cod_j, rng)
                if frames.empty:
                    continue
                event_id = f"{p['draft_year']}{int(p['nfl_id'])}-{drill[:3]}-{attempt}"
                frames = frames.assign(
                    draft_year=int(p["draft_year"]),
                    event_id=event_id,
                    nfl_id=int(p["nfl_id"]),
                    entity_type="PLAYER",
                    drill_type=drill,
                    drill_name=f"{drill}_SYNTH",
                    attempt=attempt,
                )
                tracking_frames.append(
                    frames[
                        ["draft_year", "event_id", "nfl_id", "entity_type", "time",
                         "drill_type", "drill_name", "attempt",
                         "x", "y", "s", "a", "dis", "dir"]
                    ]
                )
    combine_tracking = pd.concat(tracking_frames, ignore_index=True) if tracking_frames else pd.DataFrame()
    if not combine_tracking.empty:
        combine_tracking = combine_tracking.sample(
            frac=1.0 if not small else 0.5, random_state=seed
        ).reset_index(drop=True)

    # ------------------------------------------------------ player_play ---
    # The ground-truth linkage:
    #   outcome ~ traditional numbers (in the baseline) + cod_skill (sensor only)
    games = pd.DataFrame(
        {
            "game_id": [2023090700 + i for i in range(200 if small else 1002)],
            "game_key": np.arange(59000, 59000 + (200 if small else 1002)),
            "season": np.resize([2023, 2024, 2025], 200 if small else 1002),
            "season_type": "REG",
            "week": np.resize(np.arange(1, 19), 200 if small else 1002),
            "home_team_abbr": np.resize(["KC", "DET", "BAL", "SF"], 200 if small else 1002),
            "visitor_team_abbr": np.resize(["BUF", "DAL", "PHI", "MIA"], 200 if small else 1002),
        }
    )

    pp_rows = []
    for _, p in players.iterrows():
        g = schema.POSITION_GROUP.get(p["nfl_position"], "OTHER")
        # opportunity is driven by draft capital
        pick = p["draft_overall_pick"]
        opportunity = 1.0 if not np.isfinite(pick) else float(np.clip(1.6 - pick / 260, 0.05, 1.0))
        n_plays = int(rng.integers(30, 380) * opportunity) + 5
        if small:
            n_plays = min(n_plays, 60)
        year = int(p["draft_year"])
        for _ in range(n_plays):
            game_id = int(rng.choice(games["game_id"].to_numpy()))
            row = {
                "game_id": game_id,
                "play_id": int(rng.integers(1, 200)),
                "team_abbr": str(rng.choice(["KC", "DET", "BAL", "SF"])),
                "nfl_id": int(p["nfl_id"]),
                "lined_up_position": p["nfl_position"],
            }
            noise = rng.normal(0, 1.0)
            if g in ("WR", "TE"):
                sep = (
                    2.00
                    + 0.30 * p["_v_max_z"]
                    + signal_strength * 0.42 * p["_cod_z"]
                    + 0.35 * noise
                )
                row["separation_at_pass_forward"] = float(np.clip(sep, 0.1, 8.0))
                row["cushion"] = float(np.clip(6.0 - 0.6 * p["_v_max_z"] + rng.normal(0, 1.0), 1, 12))
                row["target"] = bool(rng.random() < 0.18)
                row["rec_yards"] = float(max(0.0, rng.normal(7.0 + 3 * p["_prod_z"], 8))) if row["target"] else np.nan
                row["yards_after_catch"] = float(max(0.0, rng.normal(4.0 + 2.5 * p["_cod_z"], 4))) if row["target"] else np.nan
            elif g == "DB":
                row["cushion"] = float(np.clip(6.2 - 0.7 * p["_v_max_z"] - 0.3 * p["_cod_z"]
                                               + rng.normal(0, 1.0), 1, 12))
                row["separation_at_pass_forward"] = float(np.clip(2.3 + 0.4 * p["_cod_z"] + 0.4 * noise, 0.1, 8))
                row["tackle"] = float(rng.random() < 0.10)
                row["assist"] = float(rng.random() < 0.06)
                row["expected_points_added"] = float(rng.normal(-0.02 * (1 + p["_cod_z"]), 1.2))
            elif g == "DL":
                get_off = (
                    0.62
                    - 0.045 * p["_v_max_z"]
                    - signal_strength * 0.055 * p["_cod_z"]
                    + 0.03 * noise
                )
                row["player_get_off"] = float(np.clip(get_off, 0.15, 1.5))
                row["time_to_pressure"] = float(np.clip(3.2 - 0.35 * p["_cod_z"] + 0.4 * noise, 0.4, 8))
                row["sack"] = float(rng.random() < 0.05)
                row["tackle_for_loss"] = float(rng.random() < 0.04)
                row["quick_pressure"] = float(rng.random() < 0.08)
            elif g == "OL":
                row["pressure_allowed"] = bool(rng.random() < 0.08 - 0.02 * p["_cod_z"])
                row["sack_allowed"] = float(rng.random() < 0.02)
                row["time_to_pressure_allowed"] = float(np.clip(3.4 + 0.4 * p["_cod_z"] + 0.4 * noise, 0.4, 9))
                row["peak_pressure_probability_allowed"] = float(np.clip(0.25 + 0.05 * noise, 0, 1))
                row["dropback_duration"] = float(np.clip(2.5 + 0.2 * noise, 0.5, 8))
            pp_rows.append(row)
    player_play = pd.DataFrame(pp_rows)

    # ------------------------------------------- player_career_successes --
    career_rows = []
    for _, p in players.iterrows():
        pick = p["draft_overall_pick"]
        opportunity = 1.0 if not np.isfinite(pick) else float(np.clip(1.6 - pick / 260, 0.05, 1.0))
        seasons = 2025 - int(p["draft_year"]) + 1
        quality = 0.55 * p["_v_max_z"] + 0.45 * p["_cod_z"] + 0.4 * p["_prod_z"]
        snaps = int(max(0, rng.normal(420 * opportunity * seasons * (1 + 0.35 * quality), 180)))
        career_rows.append(
            {
                "nfl_id": int(p["nfl_id"]),
                "career_offensive_snaps": snaps,
                "career_defensive_snaps": snaps,
                "career_special_teams_snaps": int(max(0, rng.normal(80 * opportunity, 60))),
                "career_games_active": int(max(0, rng.normal(14 * opportunity * seasons, 6))),
                "career_games_started": int(max(0, rng.normal(8 * opportunity * seasons * (1 + 0.3 * quality), 6))),
                "ap_all_pro_1st_team": int(rng.random() < 0.01 + 0.02 * max(0, quality)),
                "ap_all_pro_2nd_team": int(rng.random() < 0.02 + 0.03 * max(0, quality)),
                "pro_bowl_original_ballot": int(rng.random() < 0.03 + 0.05 * max(0, quality)),
            }
        )
    career = pd.DataFrame(career_rows)

    # ------------------------------------------------ game tracking (10Hz) --
    gt_rows = []
    n_gt_plays = 40 if small else 400
    for _, p in players.iterrows():
        g = schema.POSITION_GROUP.get(p["nfl_position"], "OTHER")
        _, _, _, _, v_mu, _, _ = POSITION_SPECS.get(g, POSITION_SPECS["WR"])
        v_max = float(np.clip(v_mu + 0.55 * p["_v_max_z"], 5.0, 11.5))
        cod = float(np.clip(0.55 + 0.16 * p["_cod_z"], 0.15, 0.98))
        if rng.random() < (0.25 if small else 0.0):
            continue
        for _ in range(int(rng.integers(1, 4))):
            frames = _simulate_path([(0, 0), (12, 0), (12, 5)], v_max, cod, rng)
            if frames.empty:
                continue
            frames = frames.assign(
                game_id=int(rng.choice(games["game_id"].to_numpy())),
                play_id=int(rng.integers(1, 200)),
                nfl_id=int(p["nfl_id"]),
            )
            frames["o"] = frames["dir"]
            frames["event"] = None
            if len(frames) > 3:
                frames.loc[frames.index[2], "event"] = "ball_snap"
                frames.loc[frames.index[-1], "event"] = "tackle"
            gt_rows.append(
                frames[["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a",
                        "dis", "o", "dir", "event"]]
            )
    game_tracking = pd.concat(gt_rows, ignore_index=True) if gt_rows else pd.DataFrame(
        columns=schema.GAME_TRACKING_COLUMNS
    )

    # -------------------------------------------------------------- write --
    players_out = players.drop(columns=[c for c in players.columns if c.startswith("_")])
    players_out.to_csv(out_dir / paths.PLAYERS, index=False)
    combine_results.to_csv(out_dir / paths.COMBINE_RESULTS, index=False)
    combine_tracking.to_csv(out_dir / paths.COMBINE_TRACKING, index=False)
    career.to_csv(out_dir / paths.CAREER_SUCCESSES, index=False)
    player_play.to_csv(out_dir / paths.PLAYER_PLAY, index=False)
    games.to_csv(out_dir / paths.GAMES, index=False)

    by_season = game_tracking.copy()
    if not by_season.empty:
        by_season["season"] = by_season["game_id"].map(
            games.set_index("game_id")["season"]
        )
    for season, fname in paths.GAME_TRACKING.items():
        sub = by_season[by_season.get("season") == season] if not by_season.empty else by_season
        sub = sub.drop(columns=["season"], errors="ignore")
        sub.to_csv(out_dir / fname, index=False)

    truth = pd.DataFrame(
        {
            "nfl_id": players["nfl_id"],
            "v_max_z": players["_v_max_z"],
            "cod_z": players["_cod_z"],
            "prod_z": players["_prod_z"],
        }
    )
    truth.to_csv(out_dir / "_ground_truth.csv", index=False)

    log.info("synthetic competition written to %s", out_dir)
    return out_dir
