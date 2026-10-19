"""Deterministic rubric scoring for a candidate configuration.

Every dimension is a fraction of *named, checkable* signals found in the
candidate's own files. There is no LLM in the loop, so a score is reproducible
byte-for-byte from the zip.

The dimensions are chosen so that each one maps onto a failure boundary the
competition's own documentation or a published post-mortem names:

    budget_discipline    the 12-hour ceiling over ~120 tasks
    navigation_protocol  "large codebases are hard to navigate"
    verification_loop    "drafts fixes ... reliably" == fixes that were tested
    patch_hygiene        patches are applied by the grader, so they must apply
    context_compaction   small local models lose to context exhaustion
    prompt_clarity       a quantified prompt beats a prose paragraph
    skill_reuse          skills are the sanctioned extension mechanism
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

WEIGHTS = {
    "budget_discipline": 0.18,
    "navigation_protocol": 0.15,
    "verification_loop": 0.18,
    "patch_hygiene": 0.14,
    "context_compaction": 0.12,
    "prompt_clarity": 0.10,
    "skill_reuse": 0.13,
}


@dataclass
class Dimension:
    name: str
    hits: List[str] = field(default_factory=list)
    misses: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.hits) + len(self.misses)
        return len(self.hits) / total if total else 0.0


@dataclass
class RubricResult:
    dimensions: Dict[str, Dimension]
    score_10: float

    def as_dict(self) -> Dict:
        return {
            "score_10": round(self.score_10, 3),
            "dimensions": {
                name: {
                    "score": round(d.score, 3),
                    "weight": WEIGHTS[name],
                    "hit": d.hits,
                    "miss": d.misses,
                }
                for name, d in self.dimensions.items()
            },
        }


def _signal(dim: Dimension, label: str, condition: bool) -> None:
    (dim.hits if condition else dim.misses).append(label)


def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _has_all(text: str, *needles: str) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def _count(text: str, *needles: str) -> int:
    low = text.lower()
    return sum(1 for n in needles if n.lower() in low)


def score_candidate(files: Dict[str, str]) -> RubricResult:
    system = files.get("prompts/system.md", "")
    all_text = "\n".join(files.values())
    has_eval = "eval_config.yaml" in files
    skill_dirs = sorted({e.split("/")[1] for e in files
                         if e.startswith("skills/") and e.count("/") >= 2})
    has_analyzer = any(e.startswith("sub_agents/") for e in files)

    dims: Dict[str, Dimension] = {}

    # --- budget_discipline ------------------------------------------------
    d = Dimension("budget_discipline")
    _signal(d, "eval_config.yaml declares per-task limits", has_eval)
    _signal(d, "prompt names get_status before acting",
            _has(system, "get_status"))
    _signal(d, "prompt states the 12-hour / total budget",
            _has(system, "12-hour", "12 hour", "wall-clock", "wall clock"))
    _signal(d, "prompt bounds exploration (stop-and-submit threshold)",
            _has(system, "20%", "stop exploring", "stop and submit"))
    _signal(d, "prompt forbids unbounded commands",
            _has(system, "never run the full", "never install", "bounded in time"))
    _signal(d, "prompt mentions turn/tool-call caps",
            _has(system, "tool call", "turn", "budget"))
    dims[d.name] = d

    # --- navigation_protocol ---------------------------------------------
    d = Dimension("navigation_protocol")
    _signal(d, "uses search_similar_code", _has(system, "search_similar_code"))
    _signal(d, "uses get_code_neighbors", _has(system, "get_code_neighbors"))
    _signal(d, "uses get_code_subgraph", _has(system, "get_code_subgraph"))
    _signal(d, "states an explicit ordered protocol",
            _has(system, "follow in order", "in order", "protocol"))
    _signal(d, "delegates surveys to a read-only sub-agent",
            has_analyzer and _has(system, "code_analyzer", "agent_tool"))
    _signal(d, "warns against blind grep",
            _has(system, "grep the whole", "do not start", "do not dump"))
    dims[d.name] = d

    # --- verification_loop ------------------------------------------------
    d = Dimension("verification_loop")
    _signal(d, "reproduce-before-edit is mandated",
            _has(system, "reproduce first", "reproduction", "reproduce"))
    _signal(d, "a repro skill ships with a script",
            any(e.startswith("skills/repro_first/scripts/") for e in files))
    _signal(d, "runs the targeted test after the edit",
            _has(system, "targeted test", "same test", "re-run"))
    _signal(d, "requires the failing test to pass before submit",
            _has(system, "must now pass", "must pass", "red test"))
    _signal(d, "patch hygiene gate before submit",
            _has(system, "patch_hygiene") and
            any(e.startswith("skills/patch_hygiene/") for e in files))
    _signal(d, "says not to submit an untested patch",
            _has(system, "must not be submitted", "submit nothing",
                 "never submit a patch you have not"))
    dims[d.name] = d

    # --- patch_hygiene ----------------------------------------------------
    d = Dimension("patch_hygiene")
    _signal(d, "forbids editing test files",
            _has(system, "never write or modify test", "modify test files",
                 "never modify the tests"))
    _signal(d, "prefers minimal diff",
            _has(system, "smallest edit", "smallest diff", "minimal", "smallest"))
    _signal(d, "prefers edit_file over write_file",
            _has_all(system, "edit_file", "write_file"))
    _signal(d, "warns about debug leftovers",
            _has(system, "debug", "print(", "breakpoint"))
    _signal(d, "guards against edit cycles",
            _has(system, "cycle", "revert", "anti-cycle", "anti pattern"))
    _signal(d, "requires a syntax check before submit",
            _has(system, "syntax"))
    dims[d.name] = d

    # --- context_compaction ----------------------------------------------
    d = Dimension("context_compaction")
    _signal(d, "names the context window as a constraint",
            _has(system, "context window", "context exhaustion", "context bloat",
                 "context compaction"))
    _signal(d, "instructs discarding tool noise",
            _has(system, "discard", "keep only", "strip"))
    _signal(d, "instructs summarising history",
            _has(system, "summarise", "summarize", "compaction"))
    _signal(d, "handles tracebacks specifically",
            _has(system, "traceback", "frame"))
    _signal(d, "refers to files by path:line instead of quoting",
            _has(system, "path:line", "refer to it by"))
    dims[d.name] = d

    # --- prompt_clarity ---------------------------------------------------
    d = Dimension("prompt_clarity")
    headings = len(re.findall(r"^#{1,3} ", system, re.MULTILINE))
    numbered = len(re.findall(r"^\s*\d+\.\s", system, re.MULTILINE))
    # Count a tool as "named" when it appears in backticks, with or without a
    # call parenthesis -- prompts write `get_status()` as often as `get_status`.
    backticked_tools = len({t for t in
                            ("run_command", "read_file", "edit_file", "write_file",
                             "get_status", "submit_patch", "search_similar_code",
                             "get_code_neighbors", "get_code_subgraph")
                            if f"`{t}`" in system or f"`{t}(" in system})
    _signal(d, "uses sectioned markdown (>=3 headings)", headings >= 3)
    _signal(d, "uses an ordered procedure (>=3 numbered steps)", numbered >= 3)
    _signal(d, "names tools in backticks (>=5 distinct)", backticked_tools >= 5)
    _signal(d, "prompt length in a workable band (600-9000 chars)",
            600 <= len(system) <= 9000)
    _signal(d, "states explicit prohibitions", _has(system, "never ", "do not "))
    dims[d.name] = d

    # --- skill_reuse ------------------------------------------------------
    d = Dimension("skill_reuse")
    _signal(d, "ships at least one skill", len(skill_dirs) >= 1)
    _signal(d, "ships >=2 skills", len(skill_dirs) >= 2)
    _signal(d, "ships >=3 skills", len(skill_dirs) >= 3)
    _signal(d, "at least one skill carries an executable script",
            any("/scripts/" in e and e.endswith(".py") for e in files))
    _signal(d, "at least two skills carry executable scripts",
            len({e.split("/")[1] for e in files
                 if e.startswith("skills/") and "/scripts/" in e and e.endswith(".py")}) >= 2)
    _signal(d, "skills ship domain-knowledge resources",
            any("/resources/" in e for e in files))
    _signal(d, "SKILL.md bodies document triggers and procedure",
            sum(1 for e in files if e.startswith("skills/") and e.endswith("SKILL.md")
                and len(files[e]) > 700) >= 1)
    dims[d.name] = d

    score_10 = sum(dims[n].score * WEIGHTS[n] for n in WEIGHTS) * 10.0
    return RubricResult(dimensions=dims, score_10=score_10)


def format_rubric(name: str, result: RubricResult) -> str:
    lines = [f"{name}: rubric {result.score_10:.2f}/10"]
    for dim, d in result.dimensions.items():
        lines.append(f"  {dim:<22} {d.score:5.2f} x {WEIGHTS[dim]:.2f}  "
                     f"missed: {', '.join(d.misses) if d.misses else '-'}")
    return "\n".join(lines)
