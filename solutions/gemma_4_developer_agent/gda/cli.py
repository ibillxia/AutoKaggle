"""CLI entry point.

    python -m gda.cli build                 # write the zips only
    python -m gda.cli validate              # harness gates for every candidate
    python -m gda.cli evaluate [--no-proxy] # gates + rubric + proxy + ranking
    python -m gda.cli report                # regenerate REPORT.md + figures
    python -m gda.cli show                  # print the ranking
    python -m gda.cli inspect v5_verified_full
    python -m gda.cli all                   # build + evaluate + report + figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gda import build, paths
from gda.candidates import CANDIDATES, build_candidate
from gda.evaluate import evaluate, load_comparison
from gda.rubric import format_rubric, score_candidate
from gda.validate import format_report, validate


def cmd_build(args) -> int:
    built = build.build_all()
    for name, path in built.items():
        print(f"{name:<20} {path.relative_to(paths.SOL_DIR)}")
    return 0


def cmd_validate(args) -> int:
    rc = 0
    for name in CANDIDATES:
        checks = validate(build_candidate(name))
        print(format_report(name, checks))
        print()
        if any(c["level"] == "hard" and not c["passed"] for c in checks):
            rc = 1
    return rc


def cmd_evaluate(args) -> int:
    payload = evaluate(run_proxy_checks=not args.no_proxy)
    _print_ranking(payload)
    return 0


def cmd_report(args) -> int:
    from gda.figures import make_all
    from gda.report import write_report

    payload = load_comparison()
    path = write_report(payload)
    figs = make_all(payload)
    print(f"report  : {path.relative_to(paths.SOL_DIR)}")
    for f in figs:
        print(f"figure  : {f.relative_to(paths.SOL_DIR)}")
    if not figs:
        print("figure  : skipped (matplotlib unavailable)")
    return 0


def cmd_show(args) -> int:
    _print_ranking(load_comparison())
    return 0


def cmd_inspect(args) -> int:
    name = args.candidate
    if name not in CANDIDATES:
        print(f"unknown candidate {name!r}; known: {list(CANDIDATES)}", file=sys.stderr)
        return 2
    files = build_candidate(name)
    print(f"# {name}: {CANDIDATES[name]['idea']}\n")
    for rel in sorted(files):
        body = files[rel]
        print(f"──── {rel}  ({len(body)} chars)")
        print(body if len(body) < 4000 else body[:4000] + "\n... [truncated]")
        print()
    print(format_report(name, validate(files)))
    print()
    print(format_rubric(name, score_candidate(files)))
    return 0


def cmd_all(args) -> int:
    cmd_build(args)
    payload = evaluate(run_proxy_checks=not args.no_proxy)
    _print_ranking(payload)
    from gda.figures import make_all
    from gda.report import write_report

    path = write_report(payload)
    figs = make_all(payload)
    print()
    print(f"best    : {payload['best']}")
    print(f"zip     : {payload['best_zip']}")
    print(f"report  : {path.relative_to(paths.SOL_DIR)}")
    print(f"figures : {len(figs)} written")
    return 0


def _print_ranking(payload) -> None:
    print(f"# best candidate: {payload['best']}\n")
    print(f"{'rank':<5}{'candidate':<20}{'composite':>10}{'rubric':>9}"
          f"{'proxy':>8}{'compiles':>10}  hard fails")
    for i, c in enumerate(payload["candidates"], 1):
        proxy = c["proxy"]["score"]
        print(f"{i:<5}{c['name']:<20}{c['composite']:>10.3f}"
              f"{c['rubric']['score_10']:>9.2f}"
              f"{('n/a' if proxy is None else f'{proxy:.2f}'):>8}"
              f"{('yes' if c['compliant'] else 'NO'):>10}"
              f"  {','.join(c['hard_failures']) or '-'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gda", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    sub.add_parser("build", help="build submission.zip for every candidate").set_defaults(func=cmd_build)
    sub.add_parser("validate", help="run the harness gates").set_defaults(func=cmd_validate)

    ev = sub.add_parser("evaluate", help="gates + rubric + proxy + ranking")
    ev.add_argument("--no-proxy", action="store_true",
                    help="skip executing skill scripts")
    ev.set_defaults(func=cmd_evaluate)

    sub.add_parser("report", help="write REPORT.md and figures").set_defaults(func=cmd_report)
    sub.add_parser("show", help="print the stored ranking").set_defaults(func=cmd_show)

    ins = sub.add_parser("inspect", help="dump one candidate's files and scores")
    ins.add_argument("candidate")
    ins.set_defaults(func=cmd_inspect)

    allp = sub.add_parser("all", help="build + evaluate + report + figures")
    allp.add_argument("--no-proxy", action="store_true")
    allp.set_defaults(func=cmd_all)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["show"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
