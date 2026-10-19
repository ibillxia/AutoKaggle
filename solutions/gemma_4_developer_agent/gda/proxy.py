"""Local proxy evaluation: actually *execute* what a candidate ships.

This is the part of the package that is not a heuristic. If a candidate declares
a skill, the proxy runs that skill's script, on a fixture repository, in a
sandboxed temp directory, and reads the machine-readable artefact the script
produced. A candidate that ships a skill which does not run is caught here.

What the proxy deliberately does NOT do: run the candidate's prompt against
Gemma 4. That needs the organiser's weights, sandbox and hidden issues. Anything
claiming to score a prompt locally would be inventing a number.

Fixtures are generated deterministically so the proxy is reproducible.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from gda import paths

REPRO_SCRIPT_REL = "skills/repro_first/scripts/run_targeted_test.py"
HYGIENE_SCRIPT_REL = "skills/patch_hygiene/scripts/check_patch.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

CALC_BUGGY = """\
def add(a, b):
    return a - b


def div(a, b):
    return a / b
"""

CALC_FIXED = """\
def add(a, b):
    return a + b


def div(a, b):
    return a / b
"""

TEST_CALC = """\
from calc import add


def test_add():
    assert add(2, 3) == 5
"""

DIFF_GOOD = """\
diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

DIFF_DEBUG = """\
diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,4 @@
 def add(a, b):
+    print("debug add", a, b)
     return a + b
"""

DIFF_TEST_EDIT = """\
diff --git a/tests/test_calc.py b/tests/test_calc.py
index 3333333..4444444 100644
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -2,3 +2,3 @@
 def test_add():
-    assert add(2, 3) == 5
+    assert add(2, 3) == 42
"""

DIFF_TRAVERSAL = """\
diff --git a/../secret.py b/../secret.py
index 5555555..6666666 100644
--- a/../secret.py
+++ b/../secret.py
@@ -1,1 +1,1 @@
-TOKEN = "old"
+TOKEN = "new"
"""

DIFF_EMPTY = """\
diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
"""


def build_fixtures(root: Path) -> Dict[str, Path]:
    """Create the fixture repos and patch files used by the proxy."""
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    out: Dict[str, Path] = {}

    # failing repo
    fail = root / "repo_fail"
    (fail / "tests").mkdir(parents=True)
    (fail / "calc.py").write_text(CALC_BUGGY, encoding="utf-8")
    (fail / "tests" / "test_calc.py").write_text(TEST_CALC, encoding="utf-8")
    out["repo_fail"] = fail

    # passing repo
    ok = root / "repo_pass"
    (ok / "tests").mkdir(parents=True)
    (ok / "calc.py").write_text(CALC_FIXED, encoding="utf-8")
    (ok / "tests" / "test_calc.py").write_text(TEST_CALC, encoding="utf-8")
    out["repo_pass"] = ok

    # a git repo with a real uncommitted fix, for the git-mode hygiene path
    g = root / "repo_git"
    (g / "tests").mkdir(parents=True)
    (g / "calc.py").write_text(CALC_BUGGY, encoding="utf-8")
    (g / "tests" / "test_calc.py").write_text(TEST_CALC, encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "proxy", "GIT_AUTHOR_EMAIL": "p@x",
           "GIT_COMMITTER_NAME": "proxy", "GIT_COMMITTER_EMAIL": "p@x"}
    for cmd in (["git", "init", "-q"],
                ["git", "add", "-A"],
                ["git", "-c", "user.name=proxy", "-c", "user.email=p@x",
                 "commit", "-q", "-m", "init"]):
        subprocess.run(cmd, cwd=str(g), capture_output=True, text=True, env={**_os_env(), **env})
    (g / "calc.py").write_text(CALC_FIXED, encoding="utf-8")
    out["repo_git"] = g

    # patch files
    patches = root / "patches"
    patches.mkdir(parents=True)
    for fname, content in {
        "good.diff": DIFF_GOOD,
        "bad_debug.diff": DIFF_DEBUG,
        "bad_test_edit.diff": DIFF_TEST_EDIT,
        "bad_traversal.diff": DIFF_TRAVERSAL,
        "empty.diff": DIFF_EMPTY,
    }.items():
        (patches / fname).write_text(content, encoding="utf-8")
    out["patches"] = patches

    return out


def _os_env():
    import os
    return os.environ.copy()


# ---------------------------------------------------------------------------
# proxy checks
# ---------------------------------------------------------------------------

@dataclass
class ProxyCheck:
    id: str
    passed: Optional[bool]
    detail: str = ""
    applicable: bool = True

    def as_dict(self) -> Dict:
        return {"id": self.id, "passed": self.passed,
                "applicable": self.applicable, "detail": self.detail}


@dataclass
class ProxyResult:
    candidate: str
    checks: List[ProxyCheck] = field(default_factory=list)

    @property
    def executed(self) -> int:
        return sum(1 for c in self.checks if c.applicable)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.applicable and c.passed)

    @property
    def score(self) -> Optional[float]:
        return (self.passed / self.executed) if self.executed else None

    def as_dict(self) -> Dict:
        return {
            "candidate": self.candidate,
            "executed": self.executed,
            "passed": self.passed,
            "score": None if self.score is None else round(self.score, 4),
            "checks": [c.as_dict() for c in self.checks],
        }


def _run_script(script: Path, args: List[str], cwd: Path,
                timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )


def run_proxy(candidate: str, files: Dict[str, str], staged: Path,
              fixtures: Dict[str, Path]) -> ProxyResult:
    res = ProxyResult(candidate=candidate)

    # --- universal: shipped python scripts must at least compile ----------
    py_scripts = [e for e in files if e.endswith(".py")]
    if py_scripts:
        bad = []
        for rel in py_scripts:
            try:
                compile(files[rel], rel, "exec")
            except SyntaxError as exc:
                bad.append(f"{rel}: {exc}")
        res.checks.append(ProxyCheck(
            "shipped_scripts_compile", not bad,
            "; ".join(bad) if bad else f"{len(py_scripts)} script(s) compile",
        ))

    # --- repro_first ------------------------------------------------------
    repro = staged / REPRO_SCRIPT_REL
    if repro.is_file():
        for fixture, expect_pass in (("repo_fail", False), ("repo_pass", True)):
            ws = fixtures[fixture]
            log = ws / ".repro" / "repro_log.json"
            if log.exists():
                log.unlink()
            try:
                proc = _run_script(repro, ["--workspace", str(ws),
                                           "--test", "tests/test_calc.py"],
                                   cwd=ws)
            except subprocess.TimeoutExpired:
                res.checks.append(ProxyCheck(
                    f"repro_{fixture}", False, "script timed out"))
                continue
            if proc.returncode != 0:
                res.checks.append(ProxyCheck(
                    f"repro_{fixture}", False,
                    f"script exited {proc.returncode}: {proc.stderr.strip()[:200]}"))
                continue
            if not log.is_file():
                res.checks.append(ProxyCheck(
                    f"repro_{fixture}", False, "no repro_log.json written"))
                continue
            record = json.loads(log.read_text(encoding="utf-8"))
            got = bool(record.get("passed"))
            res.checks.append(ProxyCheck(
                f"repro_{fixture}", got is expect_pass,
                f"expected passed={expect_pass}, got passed={got} "
                f"(rc={record.get('returncode')}, {record.get('duration_s')}s)",
            ))
    else:
        res.checks.append(ProxyCheck(
            "repro_first_skill", None, "candidate ships no repro_first skill",
            applicable=False))

    # --- patch_hygiene ----------------------------------------------------
    hygiene = staged / HYGIENE_SCRIPT_REL
    if hygiene.is_file():
        patches = fixtures["patches"]
        neutral = fixtures["repo_pass"]

        cases = [
            ("hygiene_good_patch", "good.diff", True, None),
            ("hygiene_bad_debug", "bad_debug.diff", False, "no_debug_leftovers"),
            ("hygiene_bad_test_edit", "bad_test_edit.diff", False, "no_test_file_edits"),
            ("hygiene_bad_traversal", "bad_traversal.diff", False,
             "no_path_traversal"),
            ("hygiene_empty_patch", "empty.diff", False, "diff_non_empty"),
        ]
        for cid, fname, expect_ok, expect_failed_check in cases:
            report_path = neutral / ".repro" / "patch_report.json"
            if report_path.exists():
                report_path.unlink()
            try:
                proc = _run_script(
                    hygiene,
                    ["--workspace", str(neutral),
                     "--diff-file", str(patches / fname)],
                    cwd=neutral,
                )
            except subprocess.TimeoutExpired:
                res.checks.append(ProxyCheck(cid, False, "script timed out"))
                continue
            if proc.returncode != 0:
                res.checks.append(ProxyCheck(
                    cid, False,
                    f"script exited {proc.returncode}: {proc.stderr.strip()[:200]}"))
                continue
            if not report_path.is_file():
                res.checks.append(ProxyCheck(
                    cid, False, "no patch_report.json written"))
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            ok = bool(report.get("ok"))
            failed_ids = [c["id"] for c in report.get("checks", []) if not c["passed"]]
            good = ok is expect_ok
            if good and expect_failed_check:
                good = expect_failed_check in failed_ids
            res.checks.append(ProxyCheck(
                cid, good,
                f"ok={ok} (expected {expect_ok}), failed={failed_ids}",
            ))

        # git-mode path: no --diff-file, staged intent-to-add
        ws = fixtures["repo_git"]
        report_path = ws / ".repro" / "patch_report.json"
        if report_path.exists():
            report_path.unlink()
        try:
            proc = _run_script(hygiene, ["--workspace", str(ws)], cwd=ws)
            if proc.returncode == 0 and report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                ok = bool(report.get("ok"))
                res.checks.append(ProxyCheck(
                    "hygiene_git_mode", ok and report.get("diff_source") == "git",
                    f"ok={ok}, source={report.get('diff_source')}, "
                    f"files={report.get('files')}",
                ))
            else:
                res.checks.append(ProxyCheck(
                    "hygiene_git_mode", False,
                    f"exit {proc.returncode}; {proc.stderr.strip()[:160]}"))
        except subprocess.TimeoutExpired:
            res.checks.append(ProxyCheck("hygiene_git_mode", False, "timed out"))
    else:
        res.checks.append(ProxyCheck(
            "patch_hygiene_skill", None, "candidate ships no patch_hygiene skill",
            applicable=False))

    return res


def format_proxy(result: ProxyResult) -> str:
    lines = [f"{result.candidate}: proxy "
             f"{'n/a' if result.score is None else f'{result.score:.2%}'} "
             f"({result.passed}/{result.executed} executed checks)"]
    for c in result.checks:
        tag = "n/a " if not c.applicable else ("ok  " if c.passed else "FAIL")
        lines.append(f"  [{tag}] {c.id:<24} {c.detail}")
    return "\n".join(lines)
