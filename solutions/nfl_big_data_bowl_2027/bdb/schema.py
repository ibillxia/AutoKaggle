"""Column contracts for the nine NFL Big Data Bowl 2027 source tables.

The schema is declared explicitly (rather than inferred) so that the pipeline
fails loudly -- instead of silently -- if Kaggle ever ships a different header.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# players.csv
# ---------------------------------------------------------------------------
PLAYERS_COLUMNS: List[str] = [
    "nfl_id",
    "display_name",
    "draft_year",
    "nfl_position",
    "birth_date",
    "college_name",
    "college_conference",
    "draft_round",
    "draft_pick_within_round",
    "draft_overall_pick",
]

PLAYERS_DTYPES: Dict[str, str] = {
    "nfl_id": "int64",
    "display_name": "string",
    "draft_year": "int64",
    "nfl_position": "string",
    "birth_date": "string",
    "college_name": "string",
    "college_conference": "string",
    "draft_round": "Float64",
    "draft_pick_within_round": "Float64",
    "draft_overall_pick": "Float64",
}

# ---------------------------------------------------------------------------
# combine_results.csv
# ---------------------------------------------------------------------------
COMBINE_RESULTS_COLUMNS: List[str] = [
    "draft_year",
    "nfl_id",
    "combine_position",
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps",
    "ngs_athleticism_score",
    "ngs_college_production_score",
    "ngs_final_score",
]

# Traditional (stopwatch / tape) testing numbers. These form the *baseline*
# that every sensor-derived feature must beat to be interesting.
TRADITIONAL_COMBINE_METRICS: List[str] = [
    "forty",
    "ten_yd_split",
    "three_cone",
    "short_shuttle",
    "vertical",
    "broad_jump",
    "bench_reps",
]

ANTHROPOMETRIC_METRICS: List[str] = [
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
]

# ---------------------------------------------------------------------------
# combine_tracking.csv  (10 Hz optical tracking during Combine drills)
# ---------------------------------------------------------------------------
COMBINE_TRACKING_COLUMNS: List[str] = [
    "draft_year",
    "event_id",
    "nfl_id",
    "entity_type",
    "time",
    "drill_type",
    "drill_name",
    "attempt",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "dir",
]

COMBINE_TRACKING_DTYPES: Dict[str, str] = {
    "draft_year": "int64",
    "event_id": "string",
    "nfl_id": "Int64",
    "entity_type": "string",
    "drill_type": "string",
    "drill_name": "string",
    "attempt": "Int64",
    "x": "float64",
    "y": "float64",
    "s": "float64",
    "a": "float64",
    "dis": "float64",
    "dir": "float64",
}

FORTY_YARD_DASH = "FORTY_YARD_DASH"
THREE_CONE_DRILL = "THREE_CONE_DRILL"
SHORT_SHUTTLE = "SHORT_SHUTTLE"
FORTY_DRILL_TYPES = [FORTY_YARD_DASH]

# Traits that only carry meaning when the drill actually contains a change of
# direction. A 40-yard dash is a straight line: any "cut" the detector finds in
# it is tracking noise, and letting those features into the screen produces
# large spurious effects (verified: a noise "regain slope" reached the top of
# the real-data screen before this guard was added).
COD_ONLY_FEATURES = {
    "speed_retention", "speed_loss", "entry_speed", "exit_speed",
    "cut_min_speed", "brake_rate", "exit_accel_rate", "regain_time_90",
    "regain_speed_slope", "turn_angle_deg", "turn_duration_s",
    "turn_rate_deg_s", "cut_radius_proxy", "turn_start_frac", "turn_sign",
    "n_turns", "max_turn_rate_deg_s", "total_abs_turn_deg",
}

# Bookkeeping columns that must never be treated as sensor traits.
NON_TRAIT_FEATURES = {"mean_dt", "n_frames"}
COD_DRILL_TYPES = [THREE_CONE_DRILL, SHORT_SHUTTLE]
# Prefixed "SKILL_DRILLS_<POSGROUP>"
SKILL_DRILL_PREFIX = "SKILL_DRILLS_"

# ---------------------------------------------------------------------------
# player_career_successes.csv
# ---------------------------------------------------------------------------
CAREER_SUCCESS_COLUMNS: List[str] = [
    "nfl_id",
    "career_offensive_snaps",
    "career_defensive_snaps",
    "career_special_teams_snaps",
    "career_games_active",
    "career_games_started",
    "ap_all_pro_1st_team",
    "ap_all_pro_2nd_team",
    "pro_bowl_original_ballot",
]

# ---------------------------------------------------------------------------
# player_play.csv -- in-game Next Gen Stats outcomes
# ---------------------------------------------------------------------------
PLAYER_PLAY_ID_COLUMNS: List[str] = [
    "game_id",
    "play_id",
    "team_abbr",
    "nfl_id",
    "lined_up_position",
]

# Rate / efficiency outcomes that do NOT mechanically scale with opportunity.
PLAYER_PLAY_RATE_METRICS: List[str] = [
    "separation_at_pass_forward",
    "cushion",
    "player_get_off",
    "time_to_pressure",
    "time_to_qb_hurry",
    "peak_pressure_probability_allowed",
    "time_to_pressure_allowed",
    "dropback_duration",
    "pass_rushers_encountered",
    "expected_yards_after_catch",
    "yards_after_catch",
    "pass_length",
    "expected_points_added",
]

# Binary / share outcomes (aggregated as means -> rates)
PLAYER_PLAY_BINARY_METRICS: List[str] = [
    "target",
    "sack",
    "tackle",
    "assist",
    "tackle_for_loss",
    "caused_forced_fumble",
    "recovered_fumble",
    "fumble",
    "fumble_lost",
    "pressure_allowed",
    "blitzing",
    "quick_pressure",
]

# ---------------------------------------------------------------------------
# games.csv
# ---------------------------------------------------------------------------
GAMES_COLUMNS: List[str] = [
    "game_id",
    "game_key",
    "season",
    "season_type",
    "week",
    "home_team_abbr",
    "visitor_team_abbr",
]

# ---------------------------------------------------------------------------
# game_tracking_YYYY.csv  (10 Hz in-game optical tracking)
# ---------------------------------------------------------------------------
GAME_TRACKING_COLUMNS: List[str] = [
    "game_id",
    "play_id",
    "nfl_id",
    "time",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir",
    "event",
]

GAME_TRACKING_DTYPES: Dict[str, str] = {
    "game_id": "int64",
    "play_id": "int64",
    "nfl_id": "int64",
    "x": "float64",
    "y": "float64",
    "s": "float64",
    "a": "float64",
    "dis": "float64",
    "o": "float64",
    "dir": "float64",
    "event": "string",
}

# ---------------------------------------------------------------------------
# Position grouping
# ---------------------------------------------------------------------------
# The competition focuses on five position groups.
POSITION_GROUP: Dict[str, str] = {
    # wide receivers
    "WR": "WR",
    # tight ends
    "TE": "TE",
    # defensive backs
    "CB": "DB",
    "DB": "DB",
    "SS": "DB",
    "FS": "DB",
    "S": "DB",
    # offensive line
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "OT": "OL",
    "OG": "OL",
    "OL": "OL",
    # defensive line
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    # edge / off-ball linebackers are out of scope but kept for completeness
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "RB": "RB",
    "FB": "RB",
}

FOCUS_POSITION_GROUPS: List[str] = ["WR", "TE", "DB", "OL", "DL"]

# Which in-game outcome families are meaningful for which position group.
# Used to bound the feasibility scan to hypotheses that a football analyst
# would consider legitimate (this is the analyst-in-the-loop prior).
POSITION_TO_OUTCOMES: Dict[str, List[str]] = {
    "WR": [
        "sep_mean", "sep_p75", "cushion_mean", "target_rate",
        "yac_mean", "eyac_mean", "rec_yards_per_target",
    ],
    "TE": [
        "sep_mean", "sep_p75", "cushion_mean", "target_rate", "yac_mean",
    ],
    "DB": [
        "cushion_mean", "sep_allowed_mean", "tackle_rate", "assist_rate",
        "epa_allowed_mean",
    ],
    "OL": [
        "pressure_allowed_rate", "sack_allowed_rate", "ttp_allowed_mean",
        "peak_pressure_prob_allowed", "dropback_duration_mean",
    ],
    "DL": [
        "get_off_mean", "get_off_p25", "pressure_rate", "sack_rate",
        "tfl_rate", "quick_pressure_rate", "time_to_pressure_mean",
    ],
}

# Which combine drill families are meaningful for which position group.
POSITION_TO_DRILLS: Dict[str, List[str]] = {
    "WR": [FORTY_YARD_DASH, SHORT_SHUTTLE, THREE_CONE_DRILL, "SKILL_DRILLS_WR"],
    "TE": [FORTY_YARD_DASH, SHORT_SHUTTLE, THREE_CONE_DRILL, "SKILL_DRILLS_TE"],
    "DB": [FORTY_YARD_DASH, SHORT_SHUTTLE, THREE_CONE_DRILL, "SKILL_DRILLS_DB"],
    "OL": [FORTY_YARD_DASH, SHORT_SHUTTLE, THREE_CONE_DRILL, "SKILL_DRILLS_OL"],
    "DL": [FORTY_YARD_DASH, SHORT_SHUTTLE, THREE_CONE_DRILL, "SKILL_DRILLS_DL"],
}
