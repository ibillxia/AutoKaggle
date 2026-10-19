"""The DevAgent SOP: phase sequencing, gate-driven retries, run record.

The control flow mirrors the original AutoKaggle SOP (Reader -> Planner ->
Developer -> Reviewer -> Summarizer, reviewer-driven retries, a per-phase
attempt budget). What changes is what is scored and what gates advancement --
here, an evidence score plus the phase unit tests.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE.parent
if str(_MULTI) not in sys.path:
    sys.path.insert(0, str(_MULTI))

from devagent.state import (  # noqa: E402
    DevAgentState,
    load_devagent_config,
    read_overview,
)
from devagent.unit_tests import run_unit_tests  # noqa: E402

log = logging.getLogger(__name__)


class DevAgentSOP:
    def __init__(self, competition: str, model: str = "gpt-4o", *,
                 config: Optional[Dict[str, Any]] = None):
        self.competition = competition
        self.model = model
        self.config = config or load_devagent_config()
        self.state_records: List[DevAgentState] = []

    # ------------------------------------------------------------------ phase
    def run_phase(self, state: DevAgentState) -> Tuple[str, DevAgentState]:
        from devagent.phases import PHASE_FUNCS

        fn = PHASE_FUNCS.get(state.phase)
        if fn is None:
            raise ValueError(f"no implementation for phase {state.phase!r}")

        state.make_dir()
        state.make_context()
        if not state.background_info:
            try:
                state.set_background_info(read_overview(self.competition))
            except FileNotFoundError:
                state.set_background_info("")

        attempts = 0
        max_attempts = int(self.config["scoring"]["max_iterations_per_phase"])
        while True:
            attempts += 1
            log.info("phase %s attempt %d", state.phase, attempts)
            raw_scores = fn(state)
            state.set_score(raw_scores)
            run_unit_tests(state)
            state.restore_memory()

            if state.passes() or attempts >= max_attempts:
                break

        status = "Success" if state.passes() else "Fail"
        state.memory[-1]["_attempts"] = attempts
        state.memory[-1]["_weighted_score"] = round(state.weighted_score, 3)
        return status, state

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        history = []
        for phase in self.config["phases"]:
            state = DevAgentState(phase=phase, competition=self.competition,
                                  config=self.config)
            status, state = self.run_phase(state)
            self.state_records.append(copy.deepcopy(state))
            history.append({
                "phase": phase,
                "attempt": state.memory[-1].get("_attempts", 1),
                "weighted_score": state.weighted_score,
                "passed": status == "Success",
            })
            if status != "Success":
                return {"status": "blocked", "history": history,
                        "blocked_at": phase}
        return {"status": "completed", "history": history}
