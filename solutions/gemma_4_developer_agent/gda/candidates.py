"""Candidate agent configurations ("versions of a solution").

Each candidate is a *strategy*, expressed as the exact file tree that would be
zipped into ``submission.zip``. The candidates form a ladder: each one adds one
engineering idea that the competition's own published failure modes make
necessary.

    v1_starter_kit      the shape of the official starter kit -- short prompt,
                        no per-task budget, no skills
    v2_budget_guard     + explicit per-task time/turn budgets and a
                        get_status-before-everything policy
    v3_graph_navigator  + read-only code_analyzer AgentTool and a graph-first
                        navigation protocol
    v4_skill_pack       + three skills, two of them with executable scripts
                        (reproduce-first and patch hygiene)
    v5_verified_full    + verification loop, context-compaction policy and
                        self-check-before-submit

The candidates are deliberately NOT random variants: each delta corresponds to a
named failure boundary in the competition, so the comparison result is
interpretable.
"""

from __future__ import annotations

from typing import Callable, Dict

from gda.paths import (
    DEFAULT_EXPECTED_TASKS,
    REQUIRED_MODEL,
    TOOL_REGISTRY,
)

# ---------------------------------------------------------------------------
# shared fragments
# ---------------------------------------------------------------------------


def _tools_block(names) -> str:
    return "\n".join(f"  - {n}" for n in names)


CORE_TOOLS = [
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
]

#: Sampling config for candidates that do not use the hosted-vLLM thinking path.
SAMPLING_PLAIN = """\
# configs/sampling.yaml -- generation parameters for the hosted vLLM server.
temperature: 0.2
top_p: 0.95
top_k: 40
max_output_tokens: 8192
"""

#: Sampling config matching what the hosted vLLM harness expects: thinking must
#: be ENABLED and expressed with `thinking_budget`; a `thinking_level` key is
#: rejected.
SAMPLING_THINKING = """\
# configs/sampling.yaml -- generation parameters for the hosted vLLM server.
# The harness runs vLLM, so thought channels are requested explicitly and the
# budget is capped; `thinking_level` is intentionally absent (vLLM rejects it).
temperature: 0.2
top_p: 0.95
top_k: 40
max_output_tokens: 8192
include_thoughts: true
thinking_budget: 1024
"""


def eval_config(expected_tasks: int, minutes: float, calls: int, turns: int,
                timeout: int) -> str:
    return f"""\
# eval_config.yaml -- per-task budgets. The competition allows 12 hours of agent
# time for the whole task set; the arithmetic below is what keeps the submission
# inside that ceiling.
expected_tasks: {expected_tasks}
max_time_minutes: {minutes}
max_tool_calls: {calls}
max_turns: {turns}
timeout_seconds: {timeout}
"""


EVAL_CONFIG_V4 = eval_config(DEFAULT_EXPECTED_TASKS, 4.0, 40, 60, 180)
EVAL_CONFIG_V5 = eval_config(DEFAULT_EXPECTED_TASKS, 4.5, 45, 70, 240)


# ---------------------------------------------------------------------------
# v1 -- starter kit shape
# ---------------------------------------------------------------------------

V1_SYSTEM = """\
# Role

You are an autonomous software engineer. You are given a Python repository at
`/workspace` and a bug report. Produce a patch that makes the repository's
tests for that issue pass.

# Method

1. Read the issue text.
2. Explore the repository to find the relevant code.
3. Edit the code to fix the bug.
4. Call submit_patch() so the patch is captured.

# Notes

- Work only inside /workspace.
- Do not modify the tests.
"""


def v1_starter_kit() -> Dict[str, str]:
    return {
        "agent.yaml": f"""\
name: gda_agent_v1_starter
model: {REQUIRED_MODEL}
description: 'Minimal SWE agent shaped like the official starter kit.'
instruction: !include prompts/system.md
tools:
{_tools_block(CORE_TOOLS)}
generate_content_config: !include configs/sampling.yaml
""",
        "configs/sampling.yaml": SAMPLING_PLAIN,
        "prompts/system.md": V1_SYSTEM,
    }


# ---------------------------------------------------------------------------
# v2 -- budget guard
# ---------------------------------------------------------------------------

V2_SYSTEM = """\
# Role

You are an autonomous software engineer working offline inside a sandbox. The
repository is at `/workspace`. You receive one issue at a time and must produce a
patch that makes that issue's validation tests pass.

# Hard budget

Your entire submission has a 12-hour wall-clock ceiling across every task, and
shell commands, skills and tool calls all debit that budget. Treat time as the
scarce resource it is:

- Call `get_status()` **first**, before any other action, to read the live
  budget and patch state.
- Call `get_status()` again before `submit_patch()` to confirm you still have
  time and that a patch is staged.
- If less than 20% of the per-task budget remains, stop exploring, make the
  best minimal fix you can justify from what you have already read, then
  `submit_patch()`.
- Never run the full test suite, never install packages, never run a command
  you cannot bound in time. Prefer one targeted test file over `pytest` with no
  arguments.

# Method

1. `get_status()`.
2. Read the issue, then locate the relevant module.
3. Read only the functions involved; do not dump whole files.
4. Make the smallest edit that fixes the root cause.
5. Run only the targeted validation test.
6. `submit_patch()`.

# Rules

- Never write or modify test files: grading runs the issue's own tests.
- Never leave the workspace with a syntax error. Run a syntax check before
  submitting.
- Prefer `edit_file` over `write_file`; a rewritten file destroys context for
  the next step and inflates the diff.
"""


def v2_budget_guard() -> Dict[str, str]:
    return {
        "agent.yaml": f"""\
name: gda_agent_v2_budget
model: {REQUIRED_MODEL}
description: 'Budget-aware SWE agent with explicit per-task budgets.'
instruction: !include prompts/system.md
tools:
{_tools_block(CORE_TOOLS)}
generate_content_config: !include configs/sampling.yaml
""",
        "configs/sampling.yaml": SAMPLING_PLAIN,
        "prompts/system.md": V2_SYSTEM,
        "eval_config.yaml": eval_config(DEFAULT_EXPECTED_TASKS, 4.0, 40, 60, 180),
    }


# ---------------------------------------------------------------------------
# v3 -- graph navigator
# ---------------------------------------------------------------------------

V3_SYSTEM = """\
# Role

You are an autonomous software engineer working offline inside a sandbox. The
repository is at `/workspace`. You receive one issue at a time and must produce a
patch that makes that issue's validation tests pass.

# Hard budget

Your entire submission has a 12-hour wall-clock ceiling across every task. Shell
commands, skills and tool calls all debit that budget.

- `get_status()` first, always.
- `get_status()` again before `submit_patch()`.
- Below 20% of the per-task budget: stop exploring, apply the best minimal fix
  you can justify, submit.
- Never run the full suite; never install anything.

# Navigation protocol (follow in order)

Do not start by grepping the whole tree. The harness ships a pre-computed
call/dependency graph and embeddings for this repository; use them first.

1. `search_similar_code(query=<issue text, keywords only>, k=10)` to find the
   symbols the issue is most likely about.
2. `get_code_neighbors(node=<top symbol>, max_neighbors=50)` to see who calls it
   and what it calls.
3. `get_code_subgraph(nodes=[<top symbol>, <its caller>, <its callee>])` to see
   the interaction you must not break.
4. Only then open files with `read_file`, and only the ranges the graph pointed
   you to.

Delegation: when the issue needs a broad read-only survey before you decide,
call the `code_analyzer` tool. It is a read-only sub-agent: it returns a
symbol-level map and a recommended edit site, and it never edits anything
itself. Do not delegate the edit itself.

# Method

1. `get_status()`.
2. Graph-first navigation as above.
3. Minimal edit at the root cause.
4. Targeted test.
5. `submit_patch()`.

# Rules

- Never write or modify test files.
- Never leave a syntax error behind.
- Prefer `edit_file` over `write_file`.
"""


ANALYZER_PROMPT = """\
# Role

You are a read-only code analyst. You never modify a file. You are handed a
repository at `/workspace` and a bug report, and you return a compact, factual
map of the code the report is about.

# What to return

1. The symbols most likely responsible, with file paths and line numbers.
2. The call/dependency relations between them (use the graph tools).
3. The single most likely edit site, and the reason for choosing it.
4. Any test file that exercises that site.

# Rules

- Use `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` and
  `read_file`. Never use `edit_file`, `write_file` or `submit_patch`.
- Be concise. Return symbol names and paths, not file dumps.
- If the evidence is ambiguous, say so explicitly and list the alternatives
  rather than guessing.
"""


REPO_TRIAGE_SKILL_MD = """\
---
name: repo_triage
description: Turn an issue report into a ranked list of candidate edit sites using the repository's code graph and embeddings, before any file is opened.
---

# repo_triage

Use this skill at the start of a task, and again whenever a first edit did not
make the targeted test pass.

## When to use it

- The issue text names a behaviour but no file.
- The obvious grep returns more than five plausible sites.
- A first edit failed and you need a second hypothesis rather than a second
  guess.

## Procedure

1. Extract 3-8 keywords from the issue: exception type, function name, option
   name, user-visible string.
2. `search_similar_code(query=<keywords>, k=10)`.
3. For each of the top three hits, `get_code_neighbors(node=<hit>)`.
4. Rank the hits by: (a) does a test file for this module exist, (b) is the
   symbol exported / public, (c) is the hit on the path the issue describes.
5. Return the top site and one fallback.

## Resources

- `resources/navigation_protocol.md` -- the ordering rules and the reasons.
- `resources/issue_taxonomy.md` -- how issue text maps onto code locations.
"""

NAVIGATION_PROTOCOL_MD = """\
# Navigation protocol

The graph tools are cheaper than reading files and much cheaper than being
wrong. Use them in this order.

| step | tool | why |
|---|---|---|
| 1 | `search_similar_code` | the embeddings find the symbol even when the issue never names it |
| 2 | `get_code_neighbors` | a bug is usually about an edge, not a node |
| 3 | `get_code_subgraph` | the induced subgraph is the contract you must not break |
| 4 | `read_file` with line ranges | read exactly the range the graph justified |

## Anti-patterns

- Grepping the whole repository first: costs turns, floods context, finds
  string matches rather than callers.
- Reading a 900-line module to change three lines.
- Editing before you can name the function that is wrong.
- Treating the test file as a hint about the fix: the test encodes the contract,
  it does not describe the implementation.
"""

ISSUE_TAXONOMY_MD = """\
# Issue taxonomy -> where the fix lives

| the issue says | the fix is usually in |
|---|---|
| a raised exception with a type name | the raise site and its input validation |
| "returns None / wrong value" | the return path, often one early-exit branch |
| "should accept X" | the argument parser / validator, not the business logic |
| "crash on empty input" | a guard clause at the top of the public entry point |
| "regression since ..." | a recently added branch; check the newest code in the hot path |
| "slower than expected" | an accidental nested loop or repeated recomputation |
| "documentation says A, code does B" | the code, not the documentation, unless the test agrees with the documentation |

## Rule of thumb

The smallest diff that makes the *validation test* pass without changing the
test is almost always the right diff.
"""


def v3_graph_navigator() -> Dict[str, str]:
    return {
        "agent.yaml": f"""\
name: gda_agent_v3_graph
model: {REQUIRED_MODEL}
description: 'SWE agent that navigates by repository code graph and delegates surveys.'
instruction: !include prompts/system.md
tools:
{_tools_block(CORE_TOOLS)}
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
""",
        "configs/sampling.yaml": SAMPLING_THINKING,
        "prompts/system.md": V3_SYSTEM,
        "prompts/analyzer.md": ANALYZER_PROMPT,
        "sub_agents/code_analyzer.yaml": f"""\
name: code_analyzer
model: {REQUIRED_MODEL}
description: 'Read-only repo surveyor. Returns symbol-level maps, never edits.'
instruction: !include ../prompts/analyzer.md
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
tools:
  - read_file
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
""",
        "eval_config.yaml": eval_config(DEFAULT_EXPECTED_TASKS, 4.0, 40, 60, 180),
        "skills/repo_triage/SKILL.md": REPO_TRIAGE_SKILL_MD,
        "skills/repo_triage/resources/navigation_protocol.md": NAVIGATION_PROTOCOL_MD,
        "skills/repo_triage/resources/issue_taxonomy.md": ISSUE_TAXONOMY_MD,
    }


# ---------------------------------------------------------------------------
# v4 -- skill pack with executable scripts
# ---------------------------------------------------------------------------

V4_SYSTEM = V3_SYSTEM + """

# Verification loop (mandatory)

Never submit a patch you have not just tested.

1. **Reproduce first.** Before editing, run the `repro_first` skill to confirm
   the targeted test actually fails, and record the failure. A patch written
   against an unreproduced bug is a guess.
2. **Fix.** Smallest edit at the root cause.
3. **Re-run the same test.** The test that failed must now pass.
4. **Hygiene.** Run the `patch_hygiene` skill on your working tree. It checks
   the diff for debug leftovers, test-file edits, path traversal and
   whitespace damage. If it reports `ok: false`, fix the finding and re-run it.
5. Only then `submit_patch()`.

A patch that fails its own targeted test must not be submitted. If you cannot
make it pass, submit nothing for that task and move on -- a failing patch costs
the same as no patch and burns budget.
"""

REPRO_SKILL_MD = """\
---
name: repro_first
description: Reproduce the reported failure and record it before editing anything, so the fix is aimed at a confirmed failure mode rather than a guess.
---

# repro_first

## When to use it

- At the very start of every task, before the first edit.
- After any edit, to confirm the failure is gone (it is the same command).

## Procedure

Run the bundled script:

```bash
python skills/repro_first/scripts/run_targeted_test.py \\
    --workspace /workspace --test <path/to/test_file.py>
```

It executes the targeted test with a bounded timeout, prints a one-line verdict,
and writes a machine-readable record to `<workspace>/.repro/repro_log.json`:

```json
{"command": "...", "returncode": 1, "passed": false, "duration_s": 0.42, "tail": "..."}
```

## How to read the result

- `passed: false` **before** your edit -- you have a reproduction. Good.
- `passed: true` **before** your edit -- the test already passes; you are
  probably looking at the wrong test, or the bug is only visible under a
  different invocation. Re-check the issue before editing.
- `passed: false` **after** your edit -- your fix is incomplete or wrong. Revert
  it, go back to `repo_triage`, and form a second hypothesis. Do not stack a
  second speculative edit on top of a failed one.

## Why it matters

Editing without a reproduction is how agents produce plausible, wrong patches
that cost a full task of budget and score zero.
"""

REPRO_SCRIPT = '''#!/usr/bin/env python
"""Run one targeted test with a bounded timeout and record the outcome.

    python run_targeted_test.py --workspace /workspace --test tests/test_x.py

Writes <workspace>/.repro/repro_log.json. Always exits 0: the point is to
*record* a failure, not to fail the agent's turn. The agent reads the JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TAIL_CHARS = 2000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--test", required=True, help="test file or node id")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    if not ws.is_dir():
        print(f"workspace not found: {ws}", file=sys.stderr)
        return 0

    cmd = [args.python, "-m", "pytest", args.test, "-q", "--no-header", "-x"]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ws), capture_output=True, text=True,
            timeout=args.timeout,
        )
        returncode, out, err = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = 124, True
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError as exc:
        returncode, timed_out = 127, False
        out, err = "", str(exc)

    duration = round(time.time() - started, 3)
    passed = returncode == 0
    tail = (out + err)[-TAIL_CHARS:]

    record = {
        "command": " ".join(cmd),
        "test": args.test,
        "returncode": returncode,
        "passed": passed,
        "timed_out": timed_out,
        "duration_s": duration,
        "tail": tail,
    }

    out_dir = ws / ".repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro_log.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8",
    )

    verdict = "PASSED" if passed else ("TIMED OUT" if timed_out else "FAILED")
    print(f"[repro_first] {args.test}: {verdict} (rc={returncode}, {duration}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

HYGIENE_SKILL_MD = """\
---
name: patch_hygiene
description: Inspect the working-tree diff before submitting and reject it if it contains debug leftovers, edits to test files, path traversal or whitespace damage.
---

# patch_hygiene

## When to use it

Immediately before `submit_patch()`, every time. Also when an edit produced a
surprisingly large diff.

## Procedure

```bash
python skills/patch_hygiene/scripts/check_patch.py --workspace /workspace
```

It stages intent-to-add (`git add -N .`), reads `git diff HEAD`, and writes
`<workspace>/.repro/patch_report.json`:

```json
{"ok": false, "n_files": 1, "n_added_lines": 4,
 "checks": [{"id": "no_debug_leftovers", "passed": false, "detail": "print( at line 12"}]}
```

## The checks, and why each one exists

| check | rejected because |
|---|---|
| `diff_non_empty` | an empty patch scores zero; you edited nothing |
| `diff_parses` | a malformed hunk is rejected by `git apply` on the grader side |
| `no_test_file_edits` | grading runs the issue's own tests; editing them cannot help and looks like gaming |
| `no_debug_leftovers` | `print`, `breakpoint`, `pdb.set_trace`, `TODO XXX` left in the diff are the classic small-model residue |
| `no_path_traversal` | `../` in a diff header escapes the sandbox |
| `no_whitespace_damage` | CRLF or leading tabs turn a 3-line fix into a 300-line diff |
| `no_binary_or_huge` | a vendored blob or minified asset swamps the patch |

## On failure

Fix the reported finding and run it again. Do not submit a diff that failed
hygiene -- the checks are cheap and the failure modes are exactly the ones that
turn a correct fix into a rejected patch.
"""

HYGIENE_SCRIPT = '''#!/usr/bin/env python
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
    ("print_call", re.compile(r"^[+].*\\bprint\\s*\\(")),
    ("breakpoint", re.compile(r"^\\+.*\\bbreakpoint\\s*\\(")),
    ("pdb", re.compile(r"^\\+.*\\bpdb\\.set_trace\\s*\\(")),
    ("marker", re.compile(r"^\\+.*\\b(TODO XXX|FIXME ME|XXX HACK)\\b")),
    ("console_log", re.compile(r"^\\+.*\\bconsole\\.log\\s*\\(")),
]

TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\\.py$|[^/]*_test\\.py$)")


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

    crlf = [l for l in added if l.endswith("\\r")]
    tabbed = [l for l in added if l.startswith("+\\t")]
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
'''


def v4_skill_pack() -> Dict[str, str]:
    files = v3_graph_navigator()
    files["agent.yaml"] = files["agent.yaml"].replace(
        "gda_agent_v3_graph", "gda_agent_v4_skills"
    ).replace(
        "description: 'SWE agent that navigates by repository code graph and delegates surveys.'",
        "description: 'Graph-navigating SWE agent with a reproduction-first, hygiene-checked patch loop.'",
    )
    files["prompts/system.md"] = V4_SYSTEM
    files["eval_config.yaml"] = EVAL_CONFIG_V4
    files["skills/repro_first/SKILL.md"] = REPRO_SKILL_MD
    files["skills/repro_first/scripts/run_targeted_test.py"] = REPRO_SCRIPT
    files["skills/patch_hygiene/SKILL.md"] = HYGIENE_SKILL_MD
    files["skills/patch_hygiene/scripts/check_patch.py"] = HYGIENE_SCRIPT
    return files


# ---------------------------------------------------------------------------
# v5 -- verified full
# ---------------------------------------------------------------------------

V5_SYSTEM = V4_SYSTEM + """

# Context compaction

Small local models lose to context exhaustion more often than to reasoning.
Manage the window deliberately:

- After each test run, keep only the failing assertion line and the traceback's
  last frame. Discard the banner, the plugin list, the collection output and any
  deprecation warning.
- After a `run_command`, keep the exit code and the last 20 lines that are not
  warnings.
- Once you have named the edit site, stop quoting the file and refer to it by
  `path:line` from then on.
- Never paste a whole traceback back into your own reasoning; extract the frame
  that names the file you will edit.

# Anti-cycle rule

If you make an edit and the targeted test still fails, and your next edit would
revert a line you just changed, you are in a cycle. Stop. Run `repo_triage`
again for a *different* hypothesis, or submit nothing for this task. Two
speculative edits in opposite directions are worse than one honest failure:
they consume the budget and leave the tree in a state neither test nor reader
can explain.

# Final sequence (do not reorder)

1. `get_status()` -- confirm budget and patch state.
2. Run the targeted test one last time on the final tree.
3. `patch_hygiene` -- must report `ok: true`.
4. `submit_patch()`.

If step 2 fails, go back to the fix; never proceed to step 3 with a red test.
"""


def v5_verified_full() -> Dict[str, str]:
    files = v4_skill_pack()
    files["agent.yaml"] = files["agent.yaml"].replace(
        "gda_agent_v4_skills", "gda_agent_v5_verified"
    ).replace(
        "description: 'Graph-navigating SWE agent with a reproduction-first, hygiene-checked patch loop.'",
        "description: 'Fully-instrumented SWE agent with graph navigation, "
        "reproduction-first edits, a hygiene gate, context compaction and a "
        "self-check before submit.'",
    )
    files["prompts/system.md"] = V5_SYSTEM
    files["eval_config.yaml"] = EVAL_CONFIG_V5
    files["skills/repro_first/resources/reading_a_failure.md"] = (
        "# Reading a failure cheaply\n\n"
        "A pytest failure is three useful lines inside two hundred useless ones.\n\n"
        "1. The `E   ` line: what was expected, what was received.\n"
        "2. The last `File \"...\"` frame in the traceback: the code you will edit.\n"
        "3. The assertion line in the test: the contract you must satisfy.\n\n"
        "Everything else -- the platform banner, the plugin list, the rootdir,"
        " the number of deselected tests -- is noise. Discard it before it"
        " occupies your context.\n"
    )
    files["skills/patch_hygiene/resources/why_hygiene_scores.md"] = (
        "# Why patch hygiene is scored, not cosmetic\n\n"
        "The grader applies the patch and runs the issue's validation tests."
        " A patch fails that step for reasons that have nothing to do with"
        " whether the logic is right:\n\n"
        "- a hunk whose line offsets are wrong does not apply at all;\n"
        "- a CRLF rewrite makes a three-line fix unreadable and unmergeable;\n"
        "- editing the test file makes the run validate the wrong contract;\n"
        "- a stray `print(` in a hot path can change output that tests assert on.\n\n"
        "Each of these is cheap to detect and cheap to fix, and each of them"
        " converts a working fix into a zero.\n"
    )
    return files


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CANDIDATES: Dict[str, Dict] = {
    "v1_starter_kit": {
        "build": v1_starter_kit,
        "idea": "Baseline shaped like the official starter kit: prompt-only, no budgets, no skills.",
        "delta": "baseline",
    },
    "v2_budget_guard": {
        "build": v2_budget_guard,
        "idea": "Adds explicit per-task budgets and a get_status-before-everything policy.",
        "delta": "12-hour budget ceiling across ~120 tasks",
    },
    "v3_graph_navigator": {
        "build": v3_graph_navigator,
        "idea": "Adds graph-first navigation and a read-only code_analyzer AgentTool.",
        "delta": "context exhaustion / blind grep on large repositories",
    },
    "v4_skill_pack": {
        "build": v4_skill_pack,
        "idea": "Adds reproduce-first and patch-hygiene skills, with real scripts.",
        "delta": "plausible-but-wrong patches; malformed or test-editing diffs",
    },
    "v5_verified_full": {
        "build": v5_verified_full,
        "idea": "Adds the verification loop, context compaction and anti-cycle rules.",
        "delta": "context bloat, edit cycles, submitting a red tree",
    },
}


def build_candidate(name: str) -> Dict[str, str]:
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; known: {list(CANDIDATES)}")
    return CANDIDATES[name]["build"]()


def candidate_names() -> list[str]:
    return list(CANDIDATES)
