"""Phase unit tests for the hackathon SOP.

In the original AutoKaggle the unit tests check properties of dataframes
(no missing values, no duplicate columns, submission shape). A hackathon has no
such artefacts, so the tests here check properties of the *deliverable*:

* the pass/fail gates the organizers actually enforce (notebook attached,
  tracking data used, Combine-to-NFL linkage explicit),
* the hard numeric limits (word count, figure count),
* the analytical content that separates a scored submission from an unscored
  one (stated uncertainty, controlled confounds, a formula-defined metric).

Every test returns ``(passed, message)`` and is deliberately conservative: it
fails when the evidence is absent rather than assuming the best.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

TEST_REGISTRY: Dict[str, Callable[..., Tuple[bool, str]]] = {}


def register(name: str):
    def _wrap(fn):
        TEST_REGISTRY[name] = fn
        return fn

    return _wrap


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _artifact(state, name: str) -> Path:
    return Path(state.restore_dir) / name


def _read(state, name: str) -> str:
    path = _artifact(state, name)
    return path.read_text(errors="ignore") if path.exists() else ""


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"[#*_>|-]", " ", text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", _strip_code(text)))


def count_figures(directory: Path) -> int:
    if not directory.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)


def _has_any(text: str, patterns: List[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


# ---------------------------------------------------------------------------
# generic
# ---------------------------------------------------------------------------
@register("test_document_exist")
def test_document_exist(state, **_):
    missing = [
        name
        for name in state.required_artifacts
        if not _artifact(state, name).exists()
    ]
    if missing:
        return False, f"missing required artefacts: {missing}"
    return True, f"all {len(state.required_artifacts)} artefacts present"


@register("test_rubric_checklist_complete")
def test_rubric_checklist_complete(state, **_):
    text = _read(state, "rubric_checklist.md")
    if not text:
        return False, "rubric_checklist.md is empty"
    components = ["football", "data science", "writeup", "visualization"]
    missing = [c for c in components if c not in text.lower()]
    if missing:
        return False, f"rubric checklist omits components: {missing}"
    if not _has_any(text, [r"pass\s*/\s*fail", r"gate"]):
        return False, "rubric checklist does not restate the pass/fail gates"
    return True, "all four rubric components and the gates are covered"


# ---------------------------------------------------------------------------
# phase 2 -- data audit
# ---------------------------------------------------------------------------
@register("test_audit_reports_effective_n")
def test_audit_reports_effective_n(state, **_):
    text = _read(state, "audit_report.md") + _read(state, "audit_summary.json")
    if not re.search(r"\bn\s*=\s*\d+", text) and '"n"' not in text:
        return False, "audit does not report an effective sample size per group"
    groups = ["wr", "te", "db", "ol", "dl"]
    covered = [g for g in groups if re.search(rf"\b{g.upper()}\b", text)]
    if len(covered) < 3:
        return False, f"audit covers only {covered}; expected the focus position groups"
    return True, f"effective N reported for {covered}"


@register("test_audit_lists_confounds")
def test_audit_lists_confounds(state, **_):
    text = _read(state, "audit_report.md")
    needed = {
        "draft capital": [r"draft\s+capital", r"draft\s+pick", r"opportunity"],
        "position": [r"position\s+group", r"position-stratif", r"by position"],
        "exposure": [r"exposure", r"survivor", r"seasons?"],
        "missingness": [r"missing", r"opt-?out"],
    }
    missing = [k for k, pats in needed.items() if not _has_any(text, pats)]
    if missing:
        return False, f"audit does not discuss: {missing}"
    return True, "draft capital, position, exposure and missingness are all addressed"


# ---------------------------------------------------------------------------
# phase 3 -- feasibility
# ---------------------------------------------------------------------------
@register("test_feasibility_covers_all_focus_groups")
def test_feasibility_covers_all_focus_groups(state, **_):
    path = _artifact(state, "feasibility_results.csv")
    if not path.exists():
        return False, "feasibility_results.csv missing"
    import pandas as pd

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return False, f"cannot read feasibility_results.csv: {exc}"
    if "position_group" not in df.columns:
        return False, "feasibility_results.csv has no position_group column"
    covered = set(df["position_group"].dropna().unique())
    required = {"WR", "TE", "DB", "OL", "DL"}
    if len(covered & required) < 4:
        return False, f"only scanned {sorted(covered)}; expected most of {sorted(required)}"
    return True, f"scanned position groups: {sorted(covered)}"


@register("test_feasibility_reports_uncertainty")
def test_feasibility_reports_uncertainty(state, **_):
    text = _read(state, "feasibility_report.md")
    if not _has_any(text, [r"q-?value", r"fdr", r"benjamini", r"false discovery"]):
        return False, "feasibility report does not mention multiple-comparison control"
    if not _has_any(text, [r"\bn\s*=\s*\d+", r"sample size", r"confidence", r"ci\b"]):
        return False, "feasibility report does not report sample size or uncertainty"
    return True, "multiple-comparison control and uncertainty are reported"


# ---------------------------------------------------------------------------
# phase 4 -- topic selection
# ---------------------------------------------------------------------------
@register("test_topic_selection_justified")
def test_topic_selection_justified(state, **_):
    text = _read(state, "topic_selection.md")
    if not text:
        return False, "topic_selection.md is empty"
    if not _has_any(text, [r"novel", r"beyond .*stopwatch", r"not already", r"conventional"]):
        return False, "topic selection does not argue novelty versus conventional wisdom"
    if not _has_any(text, [r"actionab", r"coach", r"scout", r"front office", r"week-to-week"]):
        return False, "topic selection does not argue actionability"
    if not re.search(r"\breject|runner-?up|alternative|candidate", text.lower()):
        return False, "topic selection does not show the rejected alternatives"
    return True, "novelty, actionability and the rejected alternatives are all argued"


@register("test_analysis_plan_has_falsification")
def test_analysis_plan_has_falsification(state, **_):
    text = _read(state, "analysis_plan.md")
    if not _has_any(text, [r"falsif", r"would abandon", r"would reject", r"disconfirm",
                           r"null result", r"kill criterion"]):
        return False, "analysis plan states no falsification criterion"
    return True, "the plan names the result that would kill the topic"


# ---------------------------------------------------------------------------
# phase 5 -- deep dive
# ---------------------------------------------------------------------------
@register("test_statistics_report_uncertainty")
def test_statistics_report_uncertainty(state, **_):
    text = _read(state, "deep_dive.md") + _read(state, "statistics.json")
    if not _has_any(text, [r"confidence interval", r"\bci\b", r"bootstrap", r"ci_low", r"ci_high"]):
        return False, "no confidence interval or bootstrap uncertainty reported"
    if not _has_any(text, [r"\bn\s*=\s*\d+", r'"n"\s*:']):
        return False, "sample size not reported alongside the statistics"
    return True, "uncertainty intervals and sample sizes are reported"


@register("test_confounds_controlled")
def test_confounds_controlled(state, **_):
    text = _read(state, "deep_dive.md")
    needed = {
        "draft capital": [r"draft\s+capital", r"draft\s+pick", r"log_draft", r"opportunity"],
        "position": [r"position"],
        "exposure": [r"exposure", r"per season", r"snaps per", r"seasons"],
    }
    missing = [k for k, pats in needed.items() if not _has_any(text, pats)]
    if missing:
        return False, f"deep dive does not control for: {missing}"
    return True, "draft capital, position and exposure are all handled"


# ---------------------------------------------------------------------------
# phase 6 -- metric
# ---------------------------------------------------------------------------
@register("test_metric_has_formula_and_direction")
def test_metric_has_formula_and_direction(state, **_):
    text = _read(state, "metric_definition.md")
    if not text:
        return False, "metric_definition.md is empty"
    if not re.search(r"=\s*[^\n]{3,}", text):
        return False, "metric has no explicit formula"
    if not _has_any(text, [r"higher is better", r"lower is better", r"direction", r"interpret"]):
        return False, "metric direction/interpretation is not stated"
    if not _has_any(text, [r"unit", r"yards?/s", r"seconds", r"ratio", r"percentile", r"z-score"]):
        return False, "metric units are not stated"
    return True, "metric has a formula, a direction and units"


@register("test_robustness_checks_present")
def test_robustness_checks_present(state, **_):
    text = _read(state, "robustness.md")
    if not text:
        return False, "robustness.md is empty"
    checks = {
        "influence/leave-one-out": [r"leave-?one-?out", r"influence", r"cook", r"jackknife"],
        "reliability": [r"reliab", r"split-?half", r"attenuat"],
        "alternative aggregation": [r"best attempt", r"mean attempt", r"median attempt",
                                    r"aggregat"],
        "out-of-sample": [r"cross-?valid", r"out-of-sample", r"hold-?out"],
    }
    missing = [k for k, pats in checks.items() if not _has_any(text, pats)]
    if len(missing) > 1:
        return False, f"robustness checks missing: {missing}"
    return True, f"robustness coverage ok (missing only {missing})"


# ---------------------------------------------------------------------------
# phase 7 -- figures
# ---------------------------------------------------------------------------
@register("test_figure_count_within_limit")
def test_figure_count_within_limit(state, **_):
    limit = int(state.limits["max_tables_and_figures"])
    n = count_figures(Path(state.restore_dir) / "figures")
    if n > limit:
        return False, f"{n} figures exceeds the {limit} table+figure budget"
    if n == 0:
        return False, "no figures produced in this phase"
    return True, f"{n} figures <= budget of {limit}"


@register("test_figures_have_takeaways")
def test_figures_have_takeaways(state, **_):
    text = _read(state, "figure_captions.md")
    if not text:
        return False, "figure_captions.md is empty"
    figures = re.findall(r"(?m)^#{1,3}\s*\S+", text)
    if len(figures) < 2:
        return False, "fewer than two figure sections found in the captions"
    if not _has_any(text, [r"takeaway", r"shows that", r"reading", r"message"]):
        return False, "captions do not state an explicit per-figure takeaway"
    return True, f"{len(figures)} figure sections each carry a takeaway"


# ---------------------------------------------------------------------------
# phase 8 -- writeup
# ---------------------------------------------------------------------------
@register("test_word_count_within_limit")
def test_word_count_within_limit(state, **_):
    text = _read(state, "writeup.md")
    if not text:
        return False, "writeup.md is empty"
    n = word_count(text)
    limit = int(state.limits["max_words"])
    if n > limit:
        return False, f"{n} words exceeds the {limit}-word limit"
    if n < limit * 0.5:
        return False, f"{n} words is well under the {limit}-word budget; the argument is thin"
    return True, f"{n} words within the {limit}-word limit"


@register("test_public_notebook_exists")
def test_public_notebook_exists(state, **_):
    path = _artifact(state, "public_notebook.ipynb")
    if not path.exists():
        return False, "public_notebook.ipynb missing (a pass/fail gate)"
    try:
        nb = json.loads(path.read_text())
    except Exception as exc:
        return False, f"public_notebook.ipynb is not valid JSON: {exc}"
    cells = nb.get("cells", [])
    if not cells:
        return False, "notebook has no cells"
    if not any(c.get("cell_type") == "code" for c in cells):
        return False, "notebook has no code cells"
    return True, f"valid notebook with {len(cells)} cells"


@register("test_uses_tracking_data")
def test_uses_tracking_data(state, **_):
    text = _read(state, "writeup.md") + _read(state, "public_notebook.ipynb")
    patterns = [r"combine_tracking", r"game_tracking", r"10\s*hz", r"tracking data",
                r"optical tracking", r"player tracking"]
    if not _has_any(text, patterns):
        return False, "writeup does not demonstrably use the player tracking data"
    return True, "player tracking data is demonstrably used"


@register("test_combine_to_nfl_linkage_explicit")
def test_combine_to_nfl_linkage_explicit(state, **_):
    text = _read(state, "writeup.md")
    combine_terms = [r"combine", r"drill", r"sensor"]
    nfl_terms = [r"regular[- ]season", r"in-game", r"nfl (?:game|performance|snaps)",
                r"separation", r"get-?off", r"pass[- ]rush"]
    if not _has_any(text, combine_terms) or not _has_any(text, nfl_terms):
        return False, "writeup does not explicitly connect Combine data to NFL performance"
    return True, "Combine-to-NFL linkage is explicit"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_unit_tests(state) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name in state.unit_tests:
        fn = TEST_REGISTRY.get(name)
        if fn is None:
            results[name] = {"passed": False, "message": f"unknown test {name!r}"}
            continue
        try:
            passed, message = fn(state)
        except Exception as exc:  # pragma: no cover
            passed, message = False, f"test raised {type(exc).__name__}: {exc}"
        results[name] = {"passed": bool(passed), "message": str(message)}
    state.unit_test_results = results
    return results


def format_unit_test_results(results: Dict[str, Dict[str, Any]]) -> str:
    if not results:
        return "No unit tests configured for this phase."
    lines = []
    for name, res in results.items():
        lines.append(
            f"[{'PASS' if res['passed'] else 'FAIL'}] {name}: {res['message']}"
        )
    return "\n".join(lines)
