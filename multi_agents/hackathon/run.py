"""CLI for the hackathon-mode SOP.

    python -m multi_agents.hackathon.run --competition nfl_big_data_bowl_2027
    python -m multi_agents.hackathon.run --check          # config self-check
    python -m multi_agents.hackathon.run --phase "Data Audit and Sensor Inventory"

Note that this driver requires an LLM endpoint configured in ``api_key.txt``
(see the repository README). The analysis pipeline itself
(``solutions/nfl_big_data_bowl_2027``) runs without any LLM and is what the
Developer agent invokes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE.parent
_ROOT = _MULTI.parent
for _p in (str(_ROOT), str(_MULTI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hackathon.sop import HackathonSOP  # noqa: E402
from hackathon.state import HackathonState, load_hackathon_config, read_overview  # noqa: E402
from hackathon.unit_tests import TEST_REGISTRY  # noqa: E402

log = logging.getLogger("hackathon")


def cmd_check(args: argparse.Namespace) -> int:
    config = load_hackathon_config()
    print(f"mode       : {config['mode']}")
    print(f"competition: {config['competition']}")
    print(f"phases     : {len(config['phases'])}")
    problems = []

    for phase in config["phases"]:
        for key in ("phase_to_agents", "phase_to_directory",
                    "phase_to_required_artifacts", "phase_to_unit_tests"):
            if phase not in config[key]:
                problems.append(f"{key} missing entry for phase {phase!r}")
        for test in config["phase_to_unit_tests"].get(phase, []):
            if test not in TEST_REGISTRY:
                problems.append(f"phase {phase!r} references unknown test {test!r}")
    weights = config["rubric"]["weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        problems.append(f"rubric weights do not sum to 1: {weights}")

    try:
        overview = read_overview(config["competition"])
        print(f"overview   : {len(overview)} chars")
    except FileNotFoundError as exc:
        problems.append(str(exc))

    print(f"unit tests : {len(TEST_REGISTRY)} registered")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nconfig OK")
    return 0


def cmd_phase(args: argparse.Namespace) -> int:
    config = load_hackathon_config()
    if args.phase not in config["phases"]:
        print(f"unknown phase {args.phase!r}; choose from:", file=sys.stderr)
        for p in config["phases"]:
            print(f"  - {p}", file=sys.stderr)
        return 2
    competition = args.competition or config["competition"]
    state = HackathonState(phase=args.phase, competition=competition, config=config)
    state.make_dir()
    state.set_background_info(read_overview(competition)[:20000])
    sop = HackathonSOP(competition, model=args.model, config=config)
    status, _ = sop.run_phase(state)
    print(f"{args.phase}: {status}")
    return 0 if status == "Success" else 1


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Run the phase unit tests against the artefacts already on disk.

    This closes the loop between the analysis pipeline
    (``solutions/nfl_big_data_bowl_2027``) and the hackathon gates: the pipeline
    writes into the canonical phase directories, and these tests re-derive
    whether the deliverable actually satisfies the competition's limits.
    """
    from hackathon.unit_tests import format_unit_test_results, run_unit_tests

    config = load_hackathon_config()
    competition = args.competition or config["competition"]
    phases = args.phase or [
        p for p in config["phases"]
        if config["phase_to_unit_tests"].get(p)
    ]
    failures = 0
    for phase in phases:
        state = HackathonState(phase=phase, competition=competition, config=config)
        state.make_dir()
        results = run_unit_tests(state)
        if not results:
            continue
        print(f"\n=== {phase} ===")
        print(format_unit_test_results(results))
        failures += sum(1 for r in results.values() if not r["passed"])
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    config = load_hackathon_config()
    competition = args.competition or config["competition"]
    sop = HackathonSOP(competition, model=args.model, config=config)
    result = sop.run()
    print(f"SOP finished with status: {result['status']}")
    for entry in result["history"]:
        print(
            f"  {entry['phase']:<48} attempt={entry['attempt']} "
            f"score={entry['weighted_score']:.2f} passed={entry['passed']}"
        )
    return 0 if result["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hackathon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--competition", default=None)
    p.add_argument("--model", default="gpt-4o")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("check", help="validate the hackathon config").set_defaults(
        func=cmd_check
    )
    ph = sub.add_parser("phase", help="run a single phase")
    ph.add_argument("phase")
    ph.set_defaults(func=cmd_phase)
    sub.add_parser("run", help="run the full SOP").set_defaults(func=cmd_run)
    sc = sub.add_parser(
        "selfcheck",
        help="run the phase unit tests against artefacts already on disk",
    )
    sc.add_argument("--phase", action="append", default=None)
    sc.set_defaults(func=cmd_selfcheck)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    if not getattr(args, "command", None):
        args = build_parser().parse_args(["check"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
