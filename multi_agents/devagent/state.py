"""State container for AutoKaggle's DevAgent-mode SOP.

Structurally a sibling of :class:`multi_agents.hackathon.state.HackathonState`:

* artefacts are phase documents + machine-readable reports, not dataframes;
* the "score" is a deterministic evidence score, not a competition metric;
* phase advancement requires an evidence pass **and** every phase unit test.

The one thing this mode adds is an explicit prohibition: the state carries a
``official_score`` field that stays ``None`` for the entire run. Nothing in this
SOP may populate it, because only the organiser's harness can.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_MULTI_AGENTS_DIR = Path(__file__).resolve().parent.parent
PREFIX_MULTI_AGENTS = str(_MULTI_AGENTS_DIR)

DEVAGENT_CONFIG = _MULTI_AGENTS_DIR / "devagent" / "config_devagent.json"
SOLUTION_DIR = _MULTI_AGENTS_DIR.parent / "solutions" / "gemma_4_developer_agent"


def load_devagent_config(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or DEVAGENT_CONFIG, "r", encoding="utf-8") as fh:
        return json.load(fh)


class DevAgentState:
    """One phase of the DevAgent SOP."""

    def __init__(self, phase: str, competition: str, *,
                 config: Optional[Dict[str, Any]] = None,
                 message: str = "There is no message."):
        self.config = config or load_devagent_config()
        self.phase = phase
        self.competition = competition
        self.message = message

        self.memory: List[Dict[str, Any]] = [{}]
        self.current_step = 0
        self.score = 0.0
        self.weighted_score = 0.0
        self.unit_test_results: Dict[str, Dict[str, Any]] = {}
        self.finished = False

        #: Only the organiser's harness may set this. Enforced by a unit test.
        self.official_score: Optional[float] = None

        self.agents: List[str] = self.config["phase_to_agents"][phase]
        self.required_artifacts: List[str] = self.config["phase_to_required_artifacts"].get(phase, [])
        self.unit_tests: List[str] = self.config["phase_to_unit_tests"].get(phase, [])
        self.rubric: Dict[str, Any] = self.config["rubric"]
        self.limits: Dict[str, Any] = self.rubric["hard_limits"]

        self.competition_dir = Path(os.environ.get(
            "GDA_COMPETITION_DIR",
            Path(PREFIX_MULTI_AGENTS) / "competition" / competition,
        ))
        self.solution_dir = Path(os.environ.get("GDA_SOLUTION_DIR", SOLUTION_DIR))

        self.dir_name = self.config["phase_to_directory"][phase]
        self.restore_dir = self.competition_dir / self.dir_name
        self.restore_dir.mkdir(parents=True, exist_ok=True)
        self.background_info = ""
        self.context = ""

    # ------------------------------------------------------------------ misc
    def __str__(self) -> str:
        return (f"DevAgentState(phase={self.phase!r}, step={self.current_step}, "
                f"agent={self.get_current_agent()}, finished={self.finished})")

    def get_current_agent(self) -> str:
        return self.agents[self.current_step % len(self.agents)]

    def next_step(self) -> None:
        self.current_step += 1

    def check_finished(self) -> bool:
        self.finished = self.current_step >= len(self.agents)
        return self.finished

    def make_dir(self) -> None:
        self.restore_dir.mkdir(parents=True, exist_ok=True)

    def make_context(self) -> str:
        self.context = (
            f"competition={self.competition} mode=devagent phase={self.phase} "
            f"agents={self.agents} required={self.required_artifacts}"
        )
        return self.context

    def set_background_info(self, text: str) -> None:
        self.background_info = text

    def update_memory(self, memory: Dict[str, Any]) -> None:
        self.memory[-1].update(memory)

    def restore_memory(self) -> None:
        (self.restore_dir / "memory.json").write_text(
            json.dumps(self.memory, indent=2, default=str), encoding="utf-8")

    # --------------------------------------------------------------- scoring
    def set_score(self, raw_scores: Dict[str, Any]) -> None:
        weights = self.rubric["weights"]
        total, weight_sum = 0.0, 0.0
        for component, weight in weights.items():
            value = raw_scores.get(component)
            if value is None:
                continue
            try:
                total += float(value) * float(weight)
                weight_sum += float(weight)
            except (TypeError, ValueError):
                continue
        self.weighted_score = total / weight_sum if weight_sum else 0.0
        self.score = self.weighted_score

    def passes(self) -> bool:
        threshold = float(self.config["scoring"]["pass_threshold"])
        if self.weighted_score < threshold:
            return False
        return all(r.get("passed", False) for r in self.unit_test_results.values())


def read_overview(competition: str) -> str:
    path = Path(PREFIX_MULTI_AGENTS) / "competition" / competition / "overview.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing competition overview: {path}")
    return path.read_text(encoding="utf-8")
