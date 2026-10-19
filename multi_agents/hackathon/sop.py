"""The hackathon-mode SOP: same control philosophy as AutoKaggle, new phases.

Control flow is deliberately identical to :class:`multi_agents.sop.SOP` so that
the two modes remain comparable:

    for each phase:
        run its agents in order
        score with the reviewer
        run the phase unit tests
        if (rubric pass) and (all unit tests pass): advance
        elif attempts remain:                     repeat with feedback
        else:                                     fail

What changes is *what* is scored (the published judging rubric, not a
leaderboard metric) and *what* gates advancement (artefact + gate unit tests,
not dataframe shape assertions).
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE.parent
_ROOT = _MULTI.parent
for _p in (str(_ROOT), str(_MULTI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hackathon.agents import create_agent  # noqa: E402
from hackathon.state import (  # noqa: E402
    HackathonState,
    load_hackathon_config,
    read_overview,
)

log = logging.getLogger(__name__)

# Per-phase model assignment, mirroring AutoKaggle's convention of spending the
# strong model on planning/development and the cheap one on reading/reviewing.
PHASE_MODELS: Dict[str, Dict[str, str]] = {
    "Reader": {"model": "gpt-4o-mini"},
    "Planner": {"model": "gpt-4o"},
    "Developer": {"model": "gpt-4o"},
    "Reviewer": {"model": "gpt-4o-mini"},
    "Summarizer": {"model": "gpt-4o-mini"},
}


class HackathonSOP:
    def __init__(
        self,
        competition: str,
        model: str = "gpt-4o",
        *,
        config: Optional[Dict[str, Any]] = None,
        developer_model: Optional[str] = None,
    ):
        self.competition = competition
        self.model = model
        self.developer_model = developer_model or model
        self.config = config or load_hackathon_config()
        self.max_iterations = int(self.config["scoring"]["max_iterations_per_phase"])
        self.attempts: Dict[str, int] = {
            phase: 0 for phase in self.config["phases"]
        }
        self.state_records: List[HackathonState] = []
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ setup
    def _agent_model(self, name: str) -> str:
        if name == "Developer":
            return self.developer_model
        if name == "Planner":
            return self.model
        return PHASE_MODELS.get(name, {}).get("model", self.model)

    def _create_agent(self, name: str):
        return create_agent(name, model=self._agent_model(name), type="api")

    # ------------------------------------------------------------------- step
    def run_phase(self, state: HackathonState) -> Tuple[str, Optional[HackathonState]]:
        """Run one phase to completion, with repeats driven by the reviewer."""
        attempt = 0
        first_error = None
        while attempt < self.max_iterations:
            attempt += 1
            state = self._run_agents(state)
            state.restore_memory()
            state.restore_report()

            passed = state.passes()
            self.history.append(
                {
                    "phase": state.phase,
                    "attempt": attempt,
                    "weighted_score": state.weighted_score,
                    "unit_tests": state.unit_test_results,
                    "passed": passed,
                }
            )
            log.info(
                "phase %s attempt %d: weighted_score=%.2f unit_tests=%s",
                state.phase,
                attempt,
                state.weighted_score,
                {k: v["passed"] for k, v in state.unit_test_results.items()},
            )
            if passed:
                return "Success", self._next_state(state)

            if attempt >= self.max_iterations:
                first_error = (
                    f"phase {state.phase!r} failed to pass after {attempt} attempts "
                    f"(score={state.weighted_score:.2f})"
                )
                break

            state = self._repeat_state(state)
        return "Fail", None if first_error is None else None

    def _run_agents(self, state: HackathonState) -> HackathonState:
        state.current_step = 0
        state.finished = False
        while not state.finished:
            name = state.get_current_agent()
            log.info("phase %s -> agent %s", state.phase, name)
            agent = self._create_agent(name)
            result = agent.action(state)
            state.update_memory({name.lower(): result})
            state.next_step()
            state.check_finished()
        return state

    def _repeat_state(self, state: HackathonState) -> HackathonState:
        new_state = HackathonState(
            phase=state.phase, competition=self.competition, config=self.config
        )
        new_state.memory = copy.deepcopy(state.memory) + [{}]
        new_state.background_info = state.background_info
        new_state.make_dir()
        return new_state

    def _next_state(self, state: HackathonState) -> Optional[HackathonState]:
        phases = self.config["phases"]
        idx = phases.index(state.phase)
        if idx + 1 >= len(phases):
            return None
        nxt = HackathonState(
            phase=phases[idx + 1], competition=self.competition, config=self.config
        )
        nxt.make_dir()
        return nxt

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        overview = read_overview(self.competition)
        state = HackathonState(
            phase=self.config["phases"][0],
            competition=self.competition,
            config=self.config,
        )
        state.make_dir()
        state.set_background_info(f"# Competition overview\n{overview[:20000]}")

        while state is not None:
            status, new_state = self.run_phase(state)
            self.state_records.append(copy.deepcopy(state))
            if status == "Fail":
                log.error("SOP stopped: phase %s could not pass", state.phase)
                return {"status": "failed", "phase": state.phase, "history": self.history}
            state = new_state

        log.info("hackathon SOP completed")
        return {"status": "completed", "history": self.history}
