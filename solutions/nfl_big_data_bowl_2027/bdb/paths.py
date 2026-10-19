"""Filesystem layout for the NFL Big Data Bowl 2027 solution package.

All heavy CSV inputs are read from ``<DATA_DIR>`` and every derived artefact is
written under ``<ARTIFACT_DIR>`` so that the pipeline is fully re-runnable and
nothing large is ever committed to git.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
# Order of preference:
#   1. $BDB_DATA_DIR
#   2. solutions/nfl_big_data_bowl_2027/data
#   3. multi_agents/competition/nfl_big_data_bowl_2027/data
#   4. a Kaggle input mount (so the public notebook runs unchanged on Kaggle)
_CANDIDATE_DATA_DIRS = [
    PACKAGE_ROOT / "data",
    REPO_ROOT / "multi_agents" / "competition" / "nfl_big_data_bowl_2027" / "data",
]

# Kaggle mounts competition data read-only under /kaggle/input/<slug>. Scanning
# for it means the published notebook needs no path edit when it is re-run there.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_SLUG = "nfl-big-data-bowl-2027"


def _kaggle_data_dirs() -> list:
    """Directories under /kaggle/input that actually hold the competition CSVs."""
    if not KAGGLE_INPUT_ROOT.is_dir():
        return []
    # Literals, not the module constants: this runs during module import, before
    # the file-name section below has been executed.
    probe = ("players.csv", "combine_tracking.csv")
    found = []
    preferred = KAGGLE_INPUT_ROOT / KAGGLE_SLUG
    # depth 2 covers both /kaggle/input/<slug>/ and /kaggle/input/<dataset>/<slug>/
    for candidate in [preferred, *sorted(KAGGLE_INPUT_ROOT.glob("*")),
                      *sorted(KAGGLE_INPUT_ROOT.glob("*/*"))]:
        if candidate.is_dir() and all((candidate / name).exists() for name in probe):
            if candidate not in found:
                found.append(candidate)
    return found


def data_dir() -> Path:
    override = os.environ.get("BDB_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in _CANDIDATE_DATA_DIRS:
        if candidate.is_dir():
            return candidate
    kaggle = _kaggle_data_dirs()
    if kaggle:
        return kaggle[0]
    return _CANDIDATE_DATA_DIRS[0]


DATA_DIR = data_dir()

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(os.environ.get("BDB_ARTIFACT_DIR", PACKAGE_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("BDB_CACHE_DIR", PACKAGE_ROOT / "cache"))
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ARTIFACT_DIR / "reports"

# ---------------------------------------------------------------------------
# Input file names
# ---------------------------------------------------------------------------
PLAYERS = "players.csv"
COMBINE_RESULTS = "combine_results.csv"
COMBINE_TRACKING = "combine_tracking.csv"
CAREER_SUCCESSES = "player_career_successes.csv"
PLAYER_PLAY = "player_play.csv"
GAMES = "games.csv"
GAME_TRACKING = {
    2023: "game_tracking_2023.csv",
    2024: "game_tracking_2024.csv",
    2025: "game_tracking_2025.csv",
}

REQUIRED_FILES = [
    PLAYERS,
    COMBINE_RESULTS,
    COMBINE_TRACKING,
    CAREER_SUCCESSES,
    PLAYER_PLAY,
    GAMES,
    *GAME_TRACKING.values(),
]

# Derived caches
CACHE_COMBINE_ATTEMPT_FEATURES = "combine_attempt_features.parquet"
CACHE_COMBINE_PLAYER_FEATURES = "combine_player_features.parquet"
CACHE_INGAME_PLAY_FEATURES = "ingame_play_features.parquet"
CACHE_INGAME_PLAYER_FEATURES = "ingame_player_features.parquet"
CACHE_PLAYER_PLAY_AGG = "player_play_agg.parquet"
CACHE_ANALYSIS_TABLE = "analysis_table.parquet"

KAGGLE_COMPETITION = "nfl-big-data-bowl-2027"


def ensure_dirs() -> None:
    for path in (DATA_DIR, ARTIFACT_DIR, CACHE_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def missing_files() -> list[str]:
    return [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]


def describe() -> str:
    lines = [
        f"PACKAGE_ROOT   : {PACKAGE_ROOT}",
        f"DATA_DIR       : {DATA_DIR}",
        f"ARTIFACT_DIR   : {ARTIFACT_DIR}",
        f"CACHE_DIR      : {CACHE_DIR}",
    ]
    missing = missing_files()
    if missing:
        lines.append(f"MISSING ({len(missing)}): " + ", ".join(missing))
    else:
        lines.append("All required input files present.")
    return "\n".join(lines)
