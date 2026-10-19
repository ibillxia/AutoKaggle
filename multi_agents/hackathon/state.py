"""State container for AutoKaggle's hackathon-mode SOP.

Differs from :class:`multi_agents.state.State` in three structural ways:

1.  No ``train.csv`` / ``test.csv`` / ``sample_submission.csv``. Artefacts are
    phase-specific documents, result tables, figures, a writeup and a notebook.
2.  The competition metric is replaced by the judging **rubric**, so the score
    gate is a weighted average over the rubric components rather than a single
    leaderboard number.
3.  Phase advancement requires *both* a rubric pass and every phase unit test
    passing, because a hackathon entry can be statistically fine and still fail
    the pass/fail gates (no notebook, over the word limit, no tracking data).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Resolved locally rather than through ``utils.py`` on purpose: ``utils`` imports
# ``api_handler`` which imports ``openai``, and the config / state / unit-test
# layer must remain importable (and testable) without an LLM client installed.
_MULTI_AGENTS_DIR = Path(__file__).resolve().parent.parent
PREFIX_MULTI_AGENTS = str(_MULTI_AGENTS_DIR)

HACKATHON_CONFIG = _MULTI_AGENTS_DIR / "hackathon" / "config_hackathon.json"


def load_hackathon_config(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or HACKATHON_CONFIG, "r") as fh:
        return json.load(fh)


class HackathonState:
    """One phase of the hackathon SOP."""

    def __init__(
        self,
        phase: str,
        competition: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        message: str = "There is no message.",
    ):
        self.config = config or load_hackathon_config()
        self.phase = phase
        self.competition = competition
        self.message = message

        self.memory: List[Dict[str, Any]] = [{}]
        self.current_step = 0
        self.score = 0.0
        self.weighted_score = 0.0
        self.unit_test_results: Dict[str, Dict[str, Any]] = {}
        self.finished = False

        self.agents: List[str] = self.config["phase_to_agents"][phase]
        self.required_artifacts: List[str] = self.config["phase_to_required_artifacts"].get(
            phase, []
        )
        self.unit_tests: List[str] = self.config["phase_to_unit_tests"].get(phase, [])
        self.rubric: Dict[str, Any] = self.config["rubric"]
        self.limits: Dict[str, Any] = self.rubric["hard_limits"]

        # honour BDB_COMPETITION_DIR so that the synthetic self-test can be run
        # against a disposable directory instead of the real competition tree
        self.competition_dir = Path(
            os.environ.get(
                "BDB_COMPETITION_DIR",
                Path(PREFIX_MULTI_AGENTS) / "competition" / competition,
            )
        )
        self.dir_name = self.config["phase_to_directory"][phase]
        self.restore_dir = self.competition_dir / self.dir_name
        self.background_info = ""
        self.context = ""

    # ------------------------------------------------------------------ misc
    def __str__(self) -> str:
        return (
            f"HackathonState(phase={self.phase!r}, step={self.current_step}, "
            f"agent={self.get_current_agent()}, finished={self.finished})"
        )

    def get_current_agent(self) -> str:
        return self.agents[self.current_step % len(self.agents)]

    def next_step(self) -> None:
        self.current_step += 1

    def check_finished(self) -> bool:
        self.finished = self.current_step >= len(self.agents)
        return self.finished

    def make_dir(self) -> None:
        self.restore_dir.mkdir(parents=True, exist_ok=True)
        (self.restore_dir / "figures").mkdir(exist_ok=True)

    def set_background_info(self, text: str) -> None:
        self.background_info = text

    def update_memory(self, memory: Dict[str, Any]) -> None:
        self.memory[-1].update(memory)

    def restore_memory(self) -> None:
        with open(self.restore_dir / "memory.json", "w") as fh:
            json.dump(self.memory, fh, indent=2, default=str)

    def restore_report(self) -> None:
        report = self.memory[-1].get("summarizer", {}).get("report", "")
        if report:
            (self.restore_dir / "report.txt").write_text(report)

    # --------------------------------------------------------------- context
    def phase_objective(self) -> str:
        from hackathon.prompts import PHASE_OBJECTIVES

        return PHASE_OBJECTIVES.get(self.phase, "")

    def build_context(self) -> str:
        from hackathon.prompts import HACKATHON_CONTEXT

        self.context = HACKATHON_CONTEXT.format(
            competition=self.competition,
            max_words=self.limits["max_words"],
            max_figure_limit=self.limits["max_tables_and_figures"] + 1,
            phases="\n".join(
                f"{i + 1}. {p}" for i, p in enumerate(self.config["phases"])
            ),
            phase=self.phase,
            phase_objective=self.phase_objective(),
            required_artifacts=", ".join(self.required_artifacts) or "(none)",
        )
        return self.context

    def read_background(self) -> str:
        """Everything discovered in earlier phases, concatenated for the LLM."""
        chunks: List[str] = []
        for phase in self.config["phases"]:
            if phase == self.phase:
                break
            directory = self.competition_dir / self.config["phase_to_directory"][phase]
            for name in self.config["phase_to_required_artifacts"].get(phase, []):
                path = directory / name
                if path.exists() and path.suffix in {".md", ".json", ".csv", ".txt"}:
                    try:
                        text = path.read_text()
                    except Exception:  # pragma: no cover
                        continue
                    if path.suffix == ".csv":
                        text = "\n".join(text.splitlines()[:40])
                    chunks.append(f"\n### {phase} / {name}\n{text[:8000]}")
        self.background_info = "\n".join(chunks) or "(no prior artefacts yet)"
        return self.background_info

    # ---------------------------------------------------------------- scoring
    def set_score(self, raw_scores: Dict[str, Any]) -> None:
        """Weighted rubric score, with the developer kill-switch honoured."""
        weights = self.rubric["weights"]
        if int(raw_scores.get("agent developer", 3)) == 0:
            self.score = 0.0
            self.weighted_score = 0.0
            return
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
        """Advance only if the rubric passes AND every unit test passed."""
        threshold = float(self.config["scoring"]["pass_threshold"])
        if self.weighted_score < threshold:
            return False
        return all(result.get("passed", False) for result in self.unit_test_results.values())


def read_overview(competition: str) -> str:
    path = Path(PREFIX_MULTI_AGENTS) / "competition" / competition / "overview.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing competition overview: {path}")
    return path.read_text()
