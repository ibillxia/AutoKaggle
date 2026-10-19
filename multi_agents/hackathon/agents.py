"""Role implementations for the hackathon SOP.

Each agent is a thin, explicit wrapper around one LLM call plus one artefact
write. The important structural difference from AutoKaggle's tabular agents is
the Developer: instead of producing ``cleaned_train.csv`` / ``processed_*.csv``,
it produces and *runs* analysis code, then reports the numbers the code printed.
That is what makes the hackathon SOP self-verifying -- a Developer turn cannot
claim a result the executed code did not produce.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE.parent
_ROOT = _MULTI.parent
for _p in (str(_ROOT), str(_MULTI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hackathon import prompts as P  # noqa: E402
from hackathon.state import HackathonState  # noqa: E402

log = logging.getLogger(__name__)

# ``llm`` pulls in the ``openai`` client, so it is imported lazily: the config,
# state and unit-test layers must work on a machine with no LLM credentials.
LLM = None  # type: ignore
_LLM_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - depends on the local environment
    from llm import LLM as _LLM  # type: ignore

    LLM = _LLM
except Exception as _exc:  # pragma: no cover
    _LLM_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class HackathonAgent:
    role = "Agent"
    description = ""

    def __init__(self, model: str = "gpt-4o", type: str = "api"):
        self.model = model
        if LLM is None:
            raise RuntimeError(
                "the `openai` package is required to drive the agents "
                f"(import error: {_LLM_IMPORT_ERROR}). Install requirements.txt in a "
                "virtualenv, or drive the phases manually -- the analysis pipeline in "
                "solutions/nfl_big_data_bowl_2027 runs without any LLM."
            )
        self.llm = LLM(model, type)

    # -- helpers -----------------------------------------------------------
    def _generate(self, prompt: str, max_tokens: int = 8192) -> str:
        reply, _ = self.llm.generate(prompt, history=[], max_completion_tokens=max_tokens)
        return reply or ""

    @staticmethod
    def _markdown(reply: str) -> str:
        m = re.search(r"```markdown(.*)```", reply, re.DOTALL)
        return (m.group(1) if m else reply).strip()

    @staticmethod
    def _json(reply: str) -> Dict[str, Any]:
        m = re.search(r"```json(.*)```", reply, re.DOTALL)
        raw = m.group(1) if m else reply
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # tolerate trailing commentary
            m2 = re.search(r"\{.*\}", raw, re.DOTALL)
            if m2:
                try:
                    return json.loads(m2.group(0))
                except json.JSONDecodeError:
                    pass
            log.error("could not parse JSON from reviewer reply")
            return {}

    @staticmethod
    def _feedback(state: HackathonState) -> str:
        parts: List[str] = []
        for i, memory in enumerate(state.memory[:-1]):
            mine = memory.get(self.role.lower(), {})
            reviewer = memory.get("reviewer", {})
            parts.append(
                f"### attempt {i}\n"
                f"your result:\n{mine.get('result', '(none)')}\n\n"
                f"reviewer score: {reviewer.get('score', {})}\n"
                f"reviewer suggestion: {reviewer.get('suggestion', {}).get(self.role.lower(), '(none)')}"
            )
        return "\n\n".join(parts) or "(this is the first attempt at this phase)"

    def action(self, state: HackathonState) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class Reader(HackathonAgent):
    role = "Reader"
    description = (
        "Reads the competition overview, rules and data dictionary and turns them "
        "into an explicit rubric checklist plus a ranked list of validity hazards."
    )

    def action(self, state: HackathonState) -> Dict[str, Any]:
        from hackathon.state import read_overview

        overview = read_overview(state.competition)
        prompt = P.PROMPT_READER_HACKATHON.replace("{overview}", overview[:60000])
        brief = self._markdown(self._generate(prompt))

        (state.restore_dir / "background.md").write_text(brief)

        # the checklist is derived from the rubric in the config, not invented
        weights = state.rubric["weights"]
        components = state.rubric["components"]
        lines = ["# Rubric checklist", ""]
        for comp, weight in weights.items():
            lines.append(f"## {comp.replace('_', ' ').title()} ({weight:.0%})")
            for question in components.get(comp, []):
                lines.append(f"- [ ] {question}")
            lines.append("")
        lines.append("## Pass / fail gates")
        for gate in state.rubric["pass_fail_gates"]:
            lines.append(f"- [ ] {gate}")
        lines.append("")
        lines.append("## Hard limits")
        for key, value in state.limits.items():
            lines.append(f"- {key}: {value}")
        checklist = "\n".join(lines)
        (state.restore_dir / "rubric_checklist.md").write_text(checklist)

        return {"result": brief, "artefacts": ["background.md", "rubric_checklist.md"]}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner(HackathonAgent):
    role = "Planner"
    description = "Turns a phase objective into a numbered, falsifiable plan."

    def action(self, state: HackathonState) -> Dict[str, Any]:
        objective = P.PLANNER_OBJECTIVES.get(state.phase, state.phase_objective())
        prompt = (
            P.PROMPT_PLANNER_HACKATHON.replace("{phase}", state.phase)
            .replace("{phase_objective}", objective)
            .replace("{background_info}", state.read_background()[:24000])
            .replace("{previous_reports}", self._previous_reports(state))
            .replace("{experience_with_suggestion}", self._feedback(state))
        )
        plan = self._markdown(self._generate(prompt))
        (state.restore_dir / "markdown_plan.txt").write_text(plan)
        return {"result": plan, "artefacts": ["markdown_plan.txt"]}

    @staticmethod
    def _previous_reports(state: HackathonState) -> str:
        reports = []
        for phase in state.config["phases"]:
            if phase == state.phase:
                break
            report = (
                state.competition_dir
                / state.config["phase_to_directory"][phase]
                / "report.txt"
            )
            if report.exists():
                reports.append(f"## {phase}\n{report.read_text()[:6000]}")
        return "\n\n".join(reports) or "(none yet)"


# ---------------------------------------------------------------------------
# Developer
# ---------------------------------------------------------------------------
class Developer(HackathonAgent):
    role = "Developer"
    description = "Writes and executes the analysis code for the phase."

    def action(self, state: HackathonState) -> Dict[str, Any]:
        plan_path = state.restore_dir / "markdown_plan.txt"
        plan = plan_path.read_text() if plan_path.exists() else "(no plan available)"
        objective = P.DEVELOPER_OBJECTIVES.get(state.phase, state.phase_objective())

        prompt = (
            P.PROMPT_DEVELOPER_HACKATHON.replace("{phase}", state.phase)
            .replace("{phase_objective}", objective)
            .replace("{markdown_plan}", plan[:16000])
            .replace("{competition_dir}", str(state.competition_dir))
            .replace("{restore_dir}", str(state.restore_dir))
            .replace("{experience_with_suggestion}", self._feedback(state))
        )
        reply = self._generate(prompt, max_tokens=8192)
        report = self._markdown(reply)

        stdout, stderr, returncode, code_path = self._run_code(reply, state)

        artefacts = self._collect_artefacts(state)
        result = (
            f"{report}\n\n"
            f"### Executed code\n"
            f"- file: {code_path}\n"
            f"- return code: {returncode}\n"
            f"### stdout (tail)\n```\n{stdout[-4000:]}\n```\n"
            + (f"### stderr (tail)\n```\n{stderr[-3000:]}\n```\n" if stderr.strip() else "")
        )
        (state.restore_dir / "developer_report.md").write_text(result)
        if returncode != 0:
            (state.restore_dir / f"{state.dir_name}_error.txt").write_text(stderr[-8000:])

        return {"result": result, "artefacts": artefacts, "returncode": returncode}

    # -- code execution ----------------------------------------------------
    @staticmethod
    def _extract_code(reply: str) -> str:
        blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", reply, re.DOTALL)
        return "\n\n".join(blocks).strip()

    def _run_code(
        self, reply: str, state: HackathonState, timeout: int = 3600
    ) -> Tuple[str, str, int, str]:
        code = self._extract_code(reply)
        code_path = state.restore_dir / "solution.py"
        if not code:
            code_path.write_text("# no code block was produced in this turn\n")
            return "", "developer produced no python code block", 1, str(code_path)
        code_path.write_text(code)

        env = dict(os.environ)
        env.setdefault("BDB_DATA_DIR", str(state.competition_dir / "data"))
        env.setdefault("BDB_ARTIFACT_DIR", str(state.restore_dir.parent.parent))
        env.setdefault("MPLCONFIGDIR", str(state.restore_dir / ".mpl"))

        log.info("running %s", code_path)
        try:
            proc = subprocess.run(
                [sys.executable, str(code_path)],
                cwd=str(state.restore_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return proc.stdout, proc.stderr, proc.returncode, str(code_path)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover
            return (
                exc.stdout or "",
                f"timed out after {timeout}s",
                124,
                str(code_path),
            )

    @staticmethod
    def _collect_artefacts(state: HackathonState) -> List[str]:
        found = []
        for name in state.required_artifacts:
            if (state.restore_dir / name).exists():
                found.append(name)
        if (state.restore_dir / "figures").exists():
            found.extend(
                str(p.relative_to(state.restore_dir))
                for p in (state.restore_dir / "figures").iterdir()
            )
        return found


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------
class Reviewer(HackathonAgent):
    role = "Reviewer"
    description = "Scores the phase against the competition rubric and decides on advancement."

    def action(self, state: HackathonState) -> Dict[str, Any]:
        from hackathon.unit_tests import format_unit_test_results, run_unit_tests

        results = run_unit_tests(state)
        developer = state.memory[-1].get("developer", {})
        work = developer.get("result", "(no developer output)")

        prompt = (
            P.PROMPT_REVIEWER_HACKATHON.replace("{phase}", state.phase)
            .replace("{phase_objective}", state.phase_objective())
            .replace("{required_artifacts}", ", ".join(state.required_artifacts))
            .replace("{work}", work[:40000])
            .replace("{unit_test_results}", format_unit_test_results(results))
        )
        raw = self._generate(prompt, max_tokens=4096)
        parsed = self._json(raw)

        score = parsed.get("score", {})
        state.set_score(score)

        # a failing unit test is a hard block regardless of the rubric opinion
        failed_tests = [n for n, r in results.items() if not r["passed"]]
        if failed_tests and int(score.get("agent developer", 3)) != 0:
            parsed["advance"] = False
            parsed.setdefault("critical_issues", []).append(
                f"unit tests failed: {failed_tests}"
            )

        suggestion = parsed.get("suggestion", {})
        if isinstance(suggestion, str):
            suggestion = {"agent developer": suggestion}

        (state.restore_dir / "review.json").write_text(json.dumps(parsed, indent=2))
        (state.restore_dir / "unit_test_results.json").write_text(
            json.dumps(results, indent=2)
        )

        return {
            "score": score,
            "weighted_score": state.weighted_score,
            "advance": bool(parsed.get("advance", False)),
            "suggestion": suggestion,
            "critical_issues": parsed.get("critical_issues", []),
            "strengths": parsed.get("strengths", []),
            "unit_tests": results,
        }


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------
class Summarizer(HackathonAgent):
    role = "Summarizer"
    description = "Compresses the phase into an actionable handoff report."

    def action(self, state: HackathonState) -> Dict[str, Any]:
        material = "\n\n".join(
            f"### {role}\n{memory.get('result', '')}"
            for role, memory in state.memory[-1].items()
            if isinstance(memory, dict)
        )
        prompt = (
            P.PROMPT_SUMMARIZER_HACKATHON.replace("{phase}", state.phase)
            .replace("{phase_objective}", state.phase_objective())
            .replace("{work}", material[:40000])
        )
        report = self._markdown(self._generate(prompt, max_tokens=4096))
        (state.restore_dir / "report.txt").write_text(report)
        state.set_background_info(state.background_info + "\n\n" + report)
        return {"report": report, "artefacts": ["report.txt"]}


ROLE_TO_CLASS = {
    "Reader": Reader,
    "Planner": Planner,
    "Developer": Developer,
    "Reviewer": Reviewer,
    "Summarizer": Summarizer,
}


def create_agent(name: str, model: str = "gpt-4o", type: str = "api") -> HackathonAgent:
    cls = ROLE_TO_CLASS.get(name)
    if cls is None:
        raise ValueError(f"unknown agent role: {name}")
    return cls(model=model, type=type)
