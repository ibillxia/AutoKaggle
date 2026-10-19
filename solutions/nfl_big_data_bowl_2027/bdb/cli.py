"""Command line entry point for the BDB2027 pipeline.

    python -m bdb.cli info                       # show paths / missing files
    python -m bdb.cli download                   # fetch data via the Kaggle API
    python -m bdb.cli synthetic --out DIR        # schema-faithful fake data
    python -m bdb.cli audit                      # phase A data audit
    python -m bdb.cli features [--force]         # phases B+C feature build
    python -m bdb.cli analyze [--stage1-only]    # phase D feasibility scan
    python -m bdb.cli all                        # audit + features + analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# matplotlib must be told where it may write before it is imported anywhere
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "cache" / "mpl")
)

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    from . import paths

    print(paths.describe())
    missing = paths.missing_files()
    if missing:
        print("\nTo fetch the data:")
        print("  1. Download kaggle.json from https://www.kaggle.com/settings")
        print("  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("     chmod 600 ~/.kaggle/kaggle.json")
        print("  3. Accept the rules at "
              "https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules")
        print(f"  4. python -m bdb.cli download --dest {paths.DATA_DIR}")
        return 1
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from . import paths

    dest = Path(args.dest).expanduser().resolve() if args.dest else paths.DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover
        print(f"kaggle package unavailable: {exc}", file=sys.stderr)
        return 2

    api = KaggleApi()
    api.authenticate()
    print(f"downloading {paths.KAGGLE_COMPETITION} -> {dest}")
    api.competition_download_files(
        paths.KAGGLE_COMPETITION, path=str(dest), quiet=False
    )
    for zf in dest.glob("*.zip"):
        print(f"unzipping {zf.name}")
        import zipfile

        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        if not args.keep_zip:
            zf.unlink()
    print("done. available files:")
    for f in sorted(dest.glob("*.csv")):
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:8.1f} MB")
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    from .synthetic import generate_synthetic_competition

    out = Path(args.out).expanduser().resolve() if args.out else None
    generate_synthetic_competition(
        out, n_players=args.n_players, seed=args.seed, small=args.small
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from . import audit

    report = audit.run_audit()
    print()
    print(audit.audit_summary_text(report))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import io, kinematics, paths

    paths.ensure_dirs()

    # ---- Combine kinematics ---------------------------------------------
    def _build_combine() -> pd.DataFrame:
        tracking = io.load_combine_tracking(players_only=True)
        attempts = kinematics.extract_all_attempt_features(tracking)
        io.write_parquet_cache(attempts, "combine_attempt_features.parquet")
        player = kinematics.player_level_combine_features(attempts)
        io.write_parquet_cache(player, "combine_player_features.parquet")
        return player

    combine_player = io.cached(
        paths.CACHE_COMBINE_PLAYER_FEATURES, _build_combine, force=args.force
    )
    print(f"combine player features: {combine_player.shape}")

    # ---- Split-half reliability of the sensor traits ----------------------
    # Must be built here, before the feasibility screen runs: the screen drops
    # traits whose reliability falls below the floor, and it also changes the
    # p-value pool that BH-FDR is computed over. Skipping it silently disables
    # the screen (``feasibility.py`` logs a warning) and lets the topic selection
    # land on whichever unreliable trait happened to score highest.
    from . import reliability

    attempts = io.read_parquet_cache(paths.CACHE_COMBINE_ATTEMPT_FEATURES)
    if attempts is None:
        _build_combine()
        attempts = io.read_parquet_cache(paths.CACHE_COMBINE_ATTEMPT_FEATURES)
    rel = reliability.build_reliability_table(attempts, force=args.force)
    print(f"trait reliability: {rel.shape[0]} traits from {len(attempts)} cut attempts")

    # ---- In-game outcomes ------------------------------------------------
    from . import ingame

    ingame_player = ingame.build_ingame_player_features(force=args.force)
    print(f"in-game player features: {ingame_player.shape}")

    # ---- Analysis table --------------------------------------------------
    table = io.build_analysis_table(combine_player, ingame_player, force=args.force)
    print(f"analysis table: {table.shape}")
    if args.show_columns:
        print(sorted(table.columns.tolist()))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import feasibility, io, paths

    paths.ensure_dirs()
    table = io.read_parquet_cache(paths.CACHE_ANALYSIS_TABLE)
    if table is None:
        print("analysis table missing -- run `python -m bdb.cli features` first",
              file=sys.stderr)
        return 2

    res = feasibility.scan_all(table, run_stage2=not args.stage1_only)
    if res.empty:
        print("no results (check that features were built)")
        return 1

    out = paths.REPORT_DIR / "phase_d_feasibility.csv"
    res.to_csv(out, index=False)
    print(f"wrote {out} ({len(res)} rows)")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    cols = [c for c in [
        "position_group", "outcome", "trait", "n", "partial_spearman",
        "q_value", "delta_r2", "adds_beyond_stopwatch",
    ] if c in res.columns]
    print(res[cols].head(40).to_string(index=False))
    return 0


def cmd_deepdive(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import deepdive, io, paths

    paths.ensure_dirs()
    table = io.read_parquet_cache(paths.CACHE_ANALYSIS_TABLE)
    if table is None:
        print("analysis table missing -- run `python -m bdb.cli features` first",
              file=sys.stderr)
        return 2

    feas_path = paths.REPORT_DIR / "phase_d_feasibility.csv"
    if args.trait and args.outcome and args.position_group:
        position_group, outcome, trait = args.position_group, args.outcome, args.trait
    else:
        if not feas_path.exists():
            print("feasibility results missing -- run `python -m bdb.cli analyze` first",
                  file=sys.stderr)
            return 2
        feas = pd.read_csv(feas_path)
        cand = feas[feas.get("adds_beyond_stopwatch", False) == True]  # noqa: E712
        cand = cand if not cand.empty else feas
        cand = cand.sort_values("delta_r2", ascending=False)
        if args.position_group:
            cand = cand[cand["position_group"] == args.position_group]
        if cand.empty:
            print("no candidate found", file=sys.stderr)
            return 1
        best = cand.iloc[0]
        position_group, outcome, trait = (
            best["position_group"], best["outcome"], best["trait"]
        )

    print(f"deep dive: position_group={position_group} outcome={outcome} trait={trait}")
    payload = deepdive.run_deep_dive(
        table,
        position_group,
        outcome,
        trait,
        metric_name=args.metric_name,
        retention_traits=[trait],
        traditional=args.traditional,
    )
    print(payload["linkage_sentence"])
    for key in ("metric_validation", "metric_vs_stopwatch"):
        if key in payload:
            print(f"\n{key}:")
            print(json.dumps(payload[key], indent=2, default=str)[:2500])
    if "blind_spots" in payload and "verdict" in payload["blind_spots"]:
        print(f"\nblind-spot verdict: {payload['blind_spots']['verdict']}")
    print(f"\nwrote {paths.REPORT_DIR / 'deep_dive.json'}")
    return 0


def cmd_submission(args: argparse.Namespace) -> int:
    import json

    from . import paths, report

    payload = report.build_submission(position_group=args.position_group)
    print(f"selected topic : {payload['topic']['position_group']} / "
          f"{payload['topic']['outcome']} / {payload['topic']['trait']}")
    print(f"figures        : {len(payload['figures'])}")
    print(f"writeup        : {payload['writeup_path']}")
    print(f"notebook       : {payload['notebook_path']}")
    review = payload["self_review"]
    print(f"words          : {review['word_count']} / 2000")
    print(f"self-review    : {'PASS' if review['passed'] else 'FAIL'}")
    for check in review["checks"]:
        print(f"  [{'x' if check['passed'] else ' '}] {check['check']} -- {check['detail']}")
    return 0 if review["passed"] else 1


def cmd_all(args: argparse.Namespace) -> int:
    ns = argparse.Namespace(**vars(args))
    cmd_audit(ns)
    cmd_features(ns)
    return cmd_analyze(ns)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bdb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show resolved paths and missing files").set_defaults(
        func=cmd_info
    )

    d = sub.add_parser("download", help="download competition data via Kaggle API")
    d.add_argument("--dest", default=None)
    d.add_argument("--keep-zip", action="store_true")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("synthetic", help="generate schema-faithful synthetic data")
    s.add_argument("--out", default=None)
    s.add_argument("--n-players", type=int, default=510)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--small", action="store_true",
                   help="tiny dataset for fast smoke tests")
    s.set_defaults(func=cmd_synthetic)

    sub.add_parser("audit", help="phase A data audit").set_defaults(func=cmd_audit)

    f = sub.add_parser("features", help="build combine + in-game features")
    f.add_argument("--force", action="store_true", help="ignore parquet caches")
    f.add_argument("--show-columns", action="store_true")
    f.set_defaults(func=cmd_features)

    a = sub.add_parser("analyze", help="phase D feasibility scan")
    a.add_argument("--stage1-only", action="store_true")
    a.set_defaults(func=cmd_analyze)

    al = sub.add_parser("all", help="audit + features + analyze")
    al.add_argument("--force", action="store_true")
    al.add_argument("--show-columns", action="store_true")
    al.add_argument("--stage1-only", action="store_true")
    al.set_defaults(func=cmd_all)

    dd = sub.add_parser("deepdive", help="phase E/F confirmatory analysis + metric")
    dd.add_argument("--position-group", default=None,
                    help="WR / TE / DB / OL / DL (default: best candidate)")
    dd.add_argument("--outcome", default=None)
    dd.add_argument("--trait", default=None)
    dd.add_argument("--metric-name", default="Break Retention Index")
    dd.add_argument("--traditional", default=None,
                    help="stopwatch column used for the blind-spot quadrant")
    dd.set_defaults(func=cmd_deepdive)

    sb = sub.add_parser("submission", help="phase G/H figures + writeup + notebook")
    sb.add_argument("--position-group", default=None)
    sb.set_defaults(func=cmd_submission)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
