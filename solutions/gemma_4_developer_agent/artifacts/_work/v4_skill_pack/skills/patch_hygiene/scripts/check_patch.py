#!/usr/bin/env python
"""Reject a working-tree diff that would score zero or be rejected.

    python check_patch.py --workspace /workspace
    python check_patch.py --workspace /workspace --diff-file some.diff

Writes <workspace>/.repro/patch_report.json and always exits 0; the agent reads
the JSON verdict rather than the exit code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

MAX_PATCH_LINES = 800
MAX_FILES = 25

DEBUG_PATTERNS = [
    ("print_call", re.compile(r"^[+].*\bprint\s*\(")),
    ("breakpoint", re.compile(r"^\+.*\bbreakpoint\s*\(")),
    ("pdb", re.compile(r"^\+.*\bpdb\.set_trace\s*\(")),
    ("marker", re.compile(r"^\+.*\b(TODO XXX|FIXME ME|XXX HACK)\b")),
    ("console_log", re.compile(r"^\+.*\bconsole\.log\s*\(")),
]

TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$)")


def git(ws: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ws), *args],
                          capture_output=True, text=True)


def collect_diff(ws: Path, diff_file: str | None) -> tuple[str, str]:
    if diff_file:
        p = Path(diff_file)
        if not p.is_absolute():
            p = ws / p
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace"), "file"
        return "", "missing"
    git(ws, "add", "-N", ".")
    proc = git(ws, "diff", "HEAD")
    return proc.stdout, "git"


def parse_files(diff: str) -> list[str]:
    files = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
    return files


def run_checks(diff: str, source: str, ws: Path) -> list[dict]:
    checks = []

    def add(cid, passed, detail=""):
        checks.append({"id": cid, "passed": bool(passed), "detail": detail})

    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    files = parse_files(diff)

    add("diff_non_empty", bool(added or removed),
        "" if (added or removed) else "no added or removed lines")

    add("diff_parses", bool(files) and "diff --git" in diff,
        "" if files else "no 'diff --git' header found")

    dirty_tests = [f for f in files if TEST_PATH.search(f)]
    add("no_test_file_edits", not dirty_tests,
        f"test files modified: {dirty_tests}" if dirty_tests else "")

    found = []
    for name, pat in DEBUG_PATTERNS:
        for line in added:
            if pat.search(line):
                found.append(f"{name}: {line.strip()[:80]}")
                break
    add("no_debug_leftovers", not found, "; ".join(found))

    traversal = [l for l in diff.splitlines()
                 if l.startswith(("--- ", "+++ ", "diff --git ")) and "../" in l]
    add("no_path_traversal", not traversal,
        f"{len(traversal)} header(s) escape the sandbox" if traversal else "")

    crlf = [l for l in added if l.endswith("\r")]
    tabbed = [l for l in added if l.startswith("+\t")]
    ws_detail = []
    if crlf:
        ws_detail.append(f"{len(crlf)} CRLF line(s)")
    if tabbed:
        ws_detail.append(f"{len(tabbed)} tab-indented line(s)")
    add("no_whitespace_damage", not ws_detail, "; ".join(ws_detail))

    huge = len(added) + len(removed) > MAX_PATCH_LINES or len(files) > MAX_FILES
    add("no_binary_or_huge", not huge,
        f"{len(files)} files / {len(added) + len(removed)} lines"
        if huge else "")

    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--diff-file", default=None)
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    diff, source = collect_diff(ws, args.diff_file)
    files = parse_files(diff)
    checks = run_checks(diff, source, ws)

    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    report = {
        "ok": all(c["passed"] for c in checks),
        "diff_source": source,
        "n_files": len(files),
        "files": files,
        "n_added_lines": len(added),
        "checks": checks,
    }

    out_dir = ws / ".repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )

    failed = [c["id"] for c in checks if not c["passed"]]
    print(f"[patch_hygiene] ok={report['ok']} files={len(files)} "
          f"added={len(added)}" + (f" failed={failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
