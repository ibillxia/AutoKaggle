"""CLI for the DevAgent-mode SOP.

    python -m multi_agents.devagent.run --check
    python -m multi_agents.devagent.run --phase "Gate Validation"
    python -m multi_agents.devagent.run --selfcheck
    python -m multi_agents.devagent.run

No LLM endpoint is required: the phase work is deterministic and delegates to
``solutions/gemma_4_developer_agent``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE.parent
_ROOT = _MULTI.parent
for _p in (str(_MULTI), str(_ROOT), str(_ROOT / "solutions" / "gemma_4_developer_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from devagent.sop import DevAgentSOP  # noqa: E402
from devagent.state import (  # noqa: E402
    DevAgentState,
    load_devagent_config,
    read_overview,
)
from devagent.unit_tests import TEST_REGISTRY, format_unit_test_results, run_unit_tests  # noqa: E402

log = logging.getLogger("devagent")


def cmd_check(args) -> int:
    cfg = load_devagent_config()
    print(f"mode       : {cfg['mode']}")
    print(f"competition: {cfg['competition']}")
    print(f"phases     : {len(cfg['phases'])}")
    problems = []
    for phase in cfg["phases"]:
        for key in ("phase_to_agents", "phase_to_directory",
                    "phase_to_required_artifacts", "phase_to_unit_tests"):
            if phase not in cfg[key]:
                problems.append(f"{key} missing entry for {phase!r}")
        for test in cfg["phase_to_unit_tests"].get(phase, []):
            if test not in TEST_REGISTRY:
                problems.append(f"phase {phase!r} references unknown test {test!r}")
    weights = cfg["rubric"]["weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        problems.append(f"rubric weights do not sum to 1: {weights}")
    try:
        overview = read_overview(cfg["competition"])
        print(f"overview   : {len(overview)} chars")
    except FileNotFoundError as exc:
        problems.append(str(exc))

    import importlib
    for mod in ("phases", "sop", "state", "unit_tests"):
        try:
            importlib.import_module(f"devagent.{mod}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"devagent.{mod} failed to import: {exc}")

    print(f"unit tests : {len(TEST_REGISTRY)} registered")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nconfig OK")
    return 0


def cmd_phase(args) -> int:
    cfg = load_devagent_config()
    if args.phase not in cfg["phases"]:
        print(f"unknown phase {args.phase!r}; choose from:", file=sys.stderr)
        for p in cfg["phases"]:
            print(f"  - {p}", file=sys.stderr)
        return 2
    competition = args.competition or cfg["competition"]
    state = DevAgentState(phase=args.phase, competition=competition, config=cfg)
    state.make_dir()
    try:
        state.set_background_info(read_overview(competition)[:20000])
    except FileNotFoundError:
        pass
    sop = DevAgentSOP(competition, model=args.model, config=cfg)
    status, state = sop.run_phase(state)
    print(f"{args.phase}: {status} (score {state.weighted_score:.2f})")
    print(format_unit_test_results(state.unit_test_results))
    return 0 if status == "Success" else 1


def cmd_selfcheck(args) -> int:
    cfg = load_devagent_config()
    competition = args.competition or cfg["competition"]
    phases = args.phase or [p for p in cfg["phases"]
                            if cfg["phase_to_unit_tests"].get(p)]
    failures = 0
    for phase in phases:
        state = DevAgentState(phase=phase, competition=competition, config=cfg)
        state.make_dir()
        results = run_unit_tests(state)
        if not results:
            continue
        print(f"\n=== {phase} ===")
        print(format_unit_test_results(results))
        failures += sum(1 for r in results.values() if not r["passed"])
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


def cmd_run(args) -> int:
    cfg = load_devagent_config()
    competition = args.competition or cfg["competition"]
    sop = DevAgentSOP(competition, model=args.model, config=cfg)
    result = sop.run()
    print(f"SOP finished with status: {result['status']}")
    for entry in result["history"]:
        print(f"  {entry['phase']:<34} attempt={entry['attempt']} "
              f"score={entry['weighted_score']:.2f} passed={entry['passed']}")
    return 0 if result["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="devagent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--competition", default=None)
    p.add_argument("--model", default="gpt-4o")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("check", help="validate config").set_defaults(func=cmd_check)
    ph = sub.add_parser("phase", help="run a single phase")
    ph.add_argument("phase")
    ph.set_defaults(func=cmd_phase)
    sub.add_parser("run", help="run the full SOP").set_defaults(func=cmd_run)
    sc = sub.add_parser("selfcheck", help="run phase unit tests against artefacts on disk")
    sc.add_argument("--phase", action="append", default=None)
    sc.set_defaults(func=cmd_selfcheck)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    if not getattr(args, "command", None):
        args = build_parser().parse_args(["check"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
