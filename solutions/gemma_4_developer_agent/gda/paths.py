"""Filesystem contract for the Gemma 4 Developer Agent solution package.

Everything this package reads or writes lives under three anchors:

    SOL_DIR    solutions/gemma_4_developer_agent/   <- this package
    REPO_ROOT  repository root                      <- where multi_agents/ lives

Artefacts are written under ``SOL_DIR/artifacts`` so that the package is
self-contained and a reader can diff two runs without touching the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- anchors ---------------------------------------------------------------
SOL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SOL_DIR.parent.parent

# --- inputs ----------------------------------------------------------------
COMPETITION = "gemma_4_developer_agent"
COMPETITION_DIR = REPO_ROOT / "multi_agents" / "competition" / COMPETITION
OVERVIEW_PATH = COMPETITION_DIR / "overview.txt"
FIXTURES_DIR = SOL_DIR / "fixtures"

# --- outputs ---------------------------------------------------------------
ARTIFACTS_DIR = SOL_DIR / "artifacts"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
WORK_DIR = ARTIFACTS_DIR / "_work"          # scratch for proxy execution
BEST_ZIP = ARTIFACTS_DIR / "best_submission.zip"

# --- competition constants (see overview.txt section 4) --------------------
REQUIRED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"

#: Extensions the released validator accepts inside submission.zip.
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}

#: The complete, closed tool registry available to a submission.
TOOL_REGISTRY = (
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
)

#: Tool names that look plausible but are NOT in the registry. Listing any of
#: these in ``tools:`` makes the harness raise ToolNotFoundError at compile time.
KNOWN_BAD_TOOL_NAMES = (
    "run_skill_script",
    "load_skill_resource",
    "bash",
    "shell",
    "python_exec",
    "apply_patch",
    "run_pytest",
    "git",
)

#: Total wall-clock budget for the whole hidden task set, in minutes.
TOTAL_BUDGET_MINUTES = 12 * 60

#: Number of tasks a candidate budgets for. The public task file has 1129 tasks
#: but the scored hidden set is reported to be ~120; candidates must be safe at
#: the larger of the two only if they claim to cover it, so we make the expected
#: task count explicit and gate on it.
DEFAULT_EXPECTED_TASKS = 120


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, SUBMISSIONS_DIR, EVALUATION_DIR, REPORTS_DIR,
              FIGURES_DIR, WORK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rel_to_sol(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(SOL_DIR))
    except ValueError:
        return str(path)


def env_override(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default
