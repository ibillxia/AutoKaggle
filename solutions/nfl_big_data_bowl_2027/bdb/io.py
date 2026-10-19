"""Cached, validated loaders for the NFL Big Data Bowl 2027 source tables.

Design goals
------------
* Fail loudly on schema drift (header check against :mod:`bdb.schema`).
* Never hold more than one game-tracking season in memory at a time.
* Cache everything expensive as parquet so re-runs are cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import paths, schema

log = logging.getLogger(__name__)


class SchemaError(RuntimeError):
    """Raised when a source file's header does not match the declared contract."""


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------
def _validate_header(
    path: Path,
    expected: Iterable[str],
    *,
    allow_extra: bool = False,
    sample_rows: int = 5,
) -> None:
    """Read only the header (+ a few rows) and verify the declared contract."""
    expected_set = set(expected)
    actual = pd.read_csv(path, nrows=sample_rows)
    actual_set = set(actual.columns)
    missing = expected_set - actual_set
    if missing:
        raise SchemaError(
            f"{path.name}: missing expected columns {sorted(missing)}; "
            f"got {sorted(actual_set)}"
        )
    if not allow_extra:
        extra = actual_set - expected_set
        if extra:
            log.warning("%s: unexpected extra columns %s", path.name, sorted(extra))


def read_csv(
    name: str,
    dtypes: Optional[Dict[str, str]] = None,
    *,
    usecols: Optional[List[str]] = None,
    validate_against: Optional[Iterable[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    path = paths.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing competition file: {path}\n"
            "Run `python -m bdb.cli download` (requires ~/.kaggle/kaggle.json and "
            "accepted competition rules) or place the file manually."
        )
    kwargs.setdefault("low_memory", False)
    if validate_against is not None:
        _validate_header(path, validate_against)
    return pd.read_csv(path, dtype=dtypes, usecols=usecols, **kwargs)


def _csv_twin(cache_name: str) -> Path:
    """CSV twin of a ``*.parquet`` cache name."""
    return paths.CACHE_DIR / (Path(cache_name).stem + ".csv")


def read_parquet_cache(cache_name: str) -> Optional[pd.DataFrame]:
    """Read a cache, preferring parquet and falling back to the CSV twin.

    ``pyarrow`` is optional here so the pipeline still runs where it cannot be
    installed (for example a Kaggle notebook without that wheel).
    """
    path = paths.CACHE_DIR / cache_name
    if path.exists():
        try:
            frame = pd.read_parquet(path)
            log.info("cache hit: %s", path)
            return frame
        except Exception as exc:  # noqa: BLE001
            log.info("parquet cache %s unreadable (%s); trying CSV twin", path.name, exc)
    twin = _csv_twin(cache_name)
    if twin.exists():
        log.info("cache hit: %s", twin)
        return pd.read_csv(twin)
    return None


def write_parquet_cache(df: pd.DataFrame, cache_name: str) -> None:
    """Write a cache as parquet when possible, always leaving a CSV twin."""
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.CACHE_DIR / cache_name
    try:
        df.to_parquet(path, index=False)
        log.info("wrote cache: %s (%s rows)", path, len(df))
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet unavailable (%s); writing CSV twin only", exc)
    twin = _csv_twin(cache_name)
    df.to_csv(twin, index=False)
    log.info("wrote cache: %s (%s rows)", twin, len(df))


def cached(cache_name: str, builder: Callable[[], pd.DataFrame], *, force: bool = False) -> pd.DataFrame:
    """Memoise an expensive dataframe builder into the on-disk cache."""
    if not force:
        hit = read_parquet_cache(cache_name)
        if hit is not None:
            return hit
    df = builder()
    write_parquet_cache(df, cache_name)
    return df


# ---------------------------------------------------------------------------
# Individual tables
# ---------------------------------------------------------------------------
def load_players() -> pd.DataFrame:
    return read_csv(
        paths.PLAYERS,
        schema.PLAYERS_DTYPES,
        validate_against=schema.PLAYERS_COLUMNS,
    )


def load_combine_results() -> pd.DataFrame:
    return read_csv(
        paths.COMBINE_RESULTS,
        validate_against=schema.COMBINE_RESULTS_COLUMNS,
    )


def load_career_successes() -> pd.DataFrame:
    return read_csv(
        paths.CAREER_SUCCESSES,
        validate_against=schema.CAREER_SUCCESS_COLUMNS,
    )


def load_games() -> pd.DataFrame:
    return read_csv(paths.GAMES, validate_against=schema.GAMES_COLUMNS)


def load_combine_tracking(
    *,
    players_only: bool = True,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the 463k-row Combine tracking table (~80 MiB, fits in memory)."""
    usecols = columns or schema.COMBINE_TRACKING_COLUMNS
    df = read_csv(
        paths.COMBINE_TRACKING,
        schema.COMBINE_TRACKING_DTYPES,
        usecols=usecols,
        validate_against=schema.COMBINE_TRACKING_COLUMNS,
    )
    if players_only and "entity_type" in df.columns:
        before = len(df)
        df = df[df["entity_type"] == "PLAYER"].copy()
        log.info("combine_tracking: kept %s/%s PLAYER frames", len(df), before)
    return df


def csv_columns(name: str) -> List[str]:
    """Header of a source CSV without reading the body."""
    path = paths.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing competition file: {path}")
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_player_play(usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load in-game Next Gen Stats player-play table (316k rows, 64 cols)."""
    path = paths.DATA_DIR / paths.PLAYER_PLAY
    if not path.exists():
        raise FileNotFoundError(f"Missing competition file: {path}")
    # only the identity columns are contractual; the metric columns are many
    # and we select them by name, so extra columns must not raise a warning
    _validate_header(path, schema.PLAYER_PLAY_ID_COLUMNS, allow_extra=True)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def iter_game_tracking(
    season: int,
    *,
    chunksize: int = 1_500_000,
    usecols: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    """Stream one season of 10 Hz in-game tracking (up to ~4.3M rows)."""
    name = paths.GAME_TRACKING[season]
    path = paths.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing competition file: {path}")
    _validate_header(path, schema.GAME_TRACKING_COLUMNS)
    return pd.read_csv(
        path,
        dtype=schema.GAME_TRACKING_DTYPES,
        usecols=usecols or schema.GAME_TRACKING_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )


def load_player_play_with_play_context(usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """player_play enriched with games.csv season/week information."""
    pp = load_player_play(usecols=usecols)
    games = load_games()[["game_id", "season", "season_type", "week"]]
    merged = pp.merge(games, on="game_id", how="left", validate="many_to_one")
    return merged


# ---------------------------------------------------------------------------
# Reference / derived tables
# ---------------------------------------------------------------------------
def position_group_table() -> pd.DataFrame:
    """players.csv plus a ``position_group`` column used to stratify analyses."""
    players = load_players().copy()
    players["position_group"] = (
        players["nfl_position"].map(schema.POSITION_GROUP).fillna("OTHER")
    )
    return players


def build_analysis_table(
    combine_player_features: pd.DataFrame,
    ingame_player_features: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """One row per prospect: identity + traditional combine + sensors + outcomes."""

    def _build() -> pd.DataFrame:
        players = position_group_table()
        results = load_combine_results().drop(columns=["draft_year"], errors="ignore")
        career = load_career_successes()

        players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
        players["age_at_draft"] = (
            players["draft_year"] - players["birth_date"].dt.year
        ).astype("Float64")

        table = (
            players.merge(results, on="nfl_id", how="left", validate="one_to_one")
            .merge(career, on="nfl_id", how="left", validate="one_to_one")
            .merge(combine_player_features, on="nfl_id", how="left")
            .merge(ingame_player_features, on="nfl_id", how="left")
        )

        # --- derived exposure / opportunity controls ---------------------
        table["draft_capital"] = table["draft_overall_pick"].fillna(300.0)
        table["log_draft_capital"] = np.log1p(table["draft_capital"])
        table["is_udfa"] = table["draft_overall_pick"].isna().astype(int)

        table["seasons_observed"] = (2025 - table["draft_year"] + 1).astype("Int64")
        for col in (
            "career_offensive_snaps",
            "career_defensive_snaps",
            "career_special_teams_snaps",
        ):
            per_season = f"{col}_per_season"
            table[per_season] = (
                table[col] / table["seasons_observed"].astype("float64")
            )
        table["career_total_snaps"] = table[
            [
                "career_offensive_snaps",
                "career_defensive_snaps",
                "career_special_teams_snaps",
            ]
        ].sum(axis=1, min_count=1)
        table["snaps_per_season"] = (
            table["career_total_snaps"] / table["seasons_observed"].astype("float64")
        )
        table["honors_score"] = (
            table["ap_all_pro_1st_team"].fillna(0)
            + table["ap_all_pro_2nd_team"].fillna(0)
            + table["pro_bowl_original_ballot"].fillna(0)
        )
        return table

    return cached(paths.CACHE_ANALYSIS_TABLE, _build, force=force)
