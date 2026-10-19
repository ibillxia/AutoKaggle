"""Phase A -- data audit.

Before any modelling, prove to the reader (and to ourselves) that the nine
tables actually join the way the data dictionary claims, and quantify the
opportunities and hazards that shape the rest of the analysis:

* row/column counts vs the published catalog
* key uniqueness and referential integrity between tables
* the effective sample size per position group (the real limit on inference)
* drill inventory per position group (which sensors exist for whom)
* missingness of the traditional Combine tests (drill opt-outs)
* exposure imbalance across draft years (survivorship / follow-up time)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import io, paths, schema

log = logging.getLogger(__name__)

EXPECTED_ROW_COUNTS = {
    "players": 510,
    "combine_results": 510,
    "combine_tracking": 463_189,
    "player_career_successes": 510,
    "player_play": 316_338,
    "games": 1_002,
    "game_tracking_2023": 1_421_805,
    "game_tracking_2024": 3_042_118,
    "game_tracking_2025": 4_339_021,
}


def _key_integrity(child: pd.Series, parent: pd.Series, label: str) -> Dict[str, Any]:
    child_set = set(child.dropna().unique().tolist())
    parent_set = set(parent.dropna().unique().tolist())
    orphans = child_set - parent_set
    return {
        "relationship": label,
        "child_unique_keys": len(child_set),
        "parent_unique_keys": len(parent_set),
        "orphan_keys": len(orphans),
        "coverage": round(1.0 - len(orphans) / max(len(child_set), 1), 4),
    }


def run_audit(*, write_report: bool = True) -> Dict[str, Any]:
    paths.ensure_dirs()
    report: Dict[str, Any] = {}

    # ---------------------------------------------------------------- tables
    players = io.load_players()
    results = io.load_combine_results()
    career = io.load_career_successes()
    games = io.load_games()
    tracking_head = io.load_combine_tracking(players_only=False)

    report["row_counts"] = {
        "players": len(players),
        "combine_results": len(results),
        "combine_tracking": len(tracking_head),
        "player_career_successes": len(career),
        "games": len(games),
    }
    report["row_count_matches_catalog"] = {
        table: report["row_counts"].get(table) == expected
        for table, expected in EXPECTED_ROW_COUNTS.items()
        if table in report["row_counts"]
    }

    # ------------------------------------------------------- key integrity
    integrity = [
        _key_integrity(results["nfl_id"], players["nfl_id"], "combine_results -> players"),
        _key_integrity(career["nfl_id"], players["nfl_id"], "career_successes -> players"),
        _key_integrity(tracking_head["nfl_id"], players["nfl_id"], "combine_tracking -> players"),
    ]
    report["key_integrity"] = integrity
    report["players_pk_unique"] = bool(players["nfl_id"].is_unique)
    report["results_pk_unique"] = bool(results["nfl_id"].is_unique)

    # ------------------------------------------------- position composition
    pos = io.position_group_table()
    report["position_group_counts"] = (
        pos["position_group"].value_counts().to_dict()
    )
    report["focus_group_counts"] = {
        g: int((pos["position_group"] == g).sum())
        for g in schema.FOCUS_POSITION_GROUPS
    }
    report["draft_year_counts"] = players["draft_year"].value_counts().sort_index().to_dict()

    # ------------------------------------------------ combine tracking shape
    trk = tracking_head.dropna(subset=["nfl_id"]).copy()
    report["combine_tracking_entity_types"] = (
        tracking_head["entity_type"].value_counts().to_dict()
    )
    report["combine_tracking_drill_types"] = (
        trk["drill_type"].value_counts().to_dict()
    )
    report["combine_tracking_n_attempts"] = int(
        trk.groupby(["event_id", "nfl_id"], observed=True).ngroups
    )
    frames_per_attempt = trk.groupby(["event_id", "nfl_id"], observed=True).size()
    report["combine_frames_per_attempt"] = {
        "mean": round(float(frames_per_attempt.mean()), 1),
        "median": float(frames_per_attempt.median()),
        "p05": float(frames_per_attempt.quantile(0.05)),
        "p95": float(frames_per_attempt.quantile(0.95)),
    }

    drill_by_pos = (
        trk.merge(pos[["nfl_id", "position_group"]], on="nfl_id", how="left")
        .pivot_table(
            index="position_group",
            columns="drill_type",
            values="event_id",
            aggfunc="nunique",
            fill_value=0,
        )
    )
    report["drill_attempts_by_position_group"] = drill_by_pos.T.to_dict()
    report["combine_tracking_players"] = int(trk["nfl_id"].nunique())
    report["combine_tracking_players_missing"] = int(
        len(set(players["nfl_id"]) - set(trk["nfl_id"]))
    )

    # --------------------------------------------------- traditional drills
    report["traditional_test_optout_rate"] = {
        col: round(float(results[col].isna().mean()), 4)
        for col in schema.TRADITIONAL_COMBINE_METRICS
        if col in results.columns
    }
    report["combine_results_missing_by_column"] = {
        col: int(results[col].isna().sum())
        for col in schema.COMBINE_RESULTS_COLUMNS
        if results[col].isna().any()
    }

    # ------------------------------------------------------ exposure / risk
    joined = players.merge(career, on="nfl_id", how="left")
    exposure = (
        joined.groupby("draft_year")[
            ["career_games_active", "career_offensive_snaps", "career_defensive_snaps"]
        ]
        .mean()
        .round(2)
    )
    # orient as {draft_year: {metric: value}} so downstream consumers can index
    # by season directly
    report["mean_exposure_by_draft_year"] = {
        str(year): {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        for year, row in exposure.iterrows()
    }
    report["udfa_count"] = int(players["draft_overall_pick"].isna().sum())
    report["draft_capital_null_rate"] = round(
        float(players["draft_overall_pick"].isna().mean()), 4
    )

    # --------------------------------------------------------- player_play
    pp = io.load_player_play()
    report["player_play_rows"] = len(pp)
    report["player_play_players"] = int(pp["nfl_id"].nunique())
    report["player_play_players_missing_vs_players"] = int(
        len(set(players["nfl_id"]) - set(pp["nfl_id"]))
    )
    report["player_play_key_unique"] = bool(
        not pp.duplicated(subset=["game_id", "play_id", "nfl_id"]).any()
    )
    plays_per_player = pp.groupby("nfl_id", observed=True).size()
    report["player_play_snaps_per_player"] = {
        "mean": round(float(plays_per_player.mean()), 1),
        "median": float(plays_per_player.median()),
        "p10": float(plays_per_player.quantile(0.10)),
        "max": int(plays_per_player.max()),
        "n_players_with_lt_20_snaps": int((plays_per_player < 20).sum()),
    }
    outcome_cols = [
        c
        for c in schema.PLAYER_PLAY_RATE_METRICS + schema.PLAYER_PLAY_BINARY_METRICS
        if c in pp.columns
    ]
    report["player_play_outcome_availability"] = {
        c: round(float(pd.to_numeric(pp[c], errors="coerce").notna().mean()), 4)
        for c in outcome_cols
    }

    if write_report:
        out = paths.REPORT_DIR / "phase_a_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        log.info("audit report written to %s", out)

    return report


def audit_summary_text(report: Dict[str, Any]) -> str:
    """Human-readable rendering used in the writeup appendix."""
    lines: List[str] = []
    lines.append("DATA AUDIT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Players: {report['row_counts']['players']} "
                 f"(unique pk: {report['players_pk_unique']})")
    lines.append(f"Combine tracking attempts: {report['combine_tracking_n_attempts']}")
    lines.append(f"Combine tracking players covered: {report['combine_tracking_players']}"
                 f" / missing {report['combine_tracking_players_missing']}")
    lines.append("")
    lines.append("Effective sample size per position group (the real N):")
    for group, n in sorted(report["focus_group_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {group:>3}: n = {n}")
    lines.append("")
    lines.append("Traditional Combine opt-out rates (missing fraction):")
    for col, rate in report["traditional_test_optout_rate"].items():
        lines.append(f"  {col:<16} {rate:.1%}")
    lines.append("")
    lines.append("Mean career exposure by draft year (survivorship hazard):")
    for year, row in report["mean_exposure_by_draft_year"].items():
        lines.append(f"  {year}: {row}")
    lines.append("")
    lines.append("Player-play snaps per prospect: " + str(report["player_play_snaps_per_player"]))
    return "\n".join(lines)
