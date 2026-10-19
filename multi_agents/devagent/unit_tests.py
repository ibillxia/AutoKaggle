"""Phase unit tests for the DevAgent SOP.

In the tabular SOP these tests check dataframe properties. Here there is no
dataframe and no metric, so they check the **artefact contract** of a code-agent
submission: that the phase documents exist, that they were derived from the real
build rather than asserted, and above all that **no official score is claimed**.

Every test returns ``(passed, message)`` and fails when evidence is absent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

TEST_REGISTRY: Dict[str, Callable[..., Tuple[bool, str]]] = {}


def register(name: str):
    def _wrap(fn):
        TEST_REGISTRY[name] = fn
        return fn
    return _wrap


def _artifact(state, name: str) -> Path:
    return Path(state.restore_dir) / name


def _read(state, name: str) -> str:
    p = _artifact(state, name)
    return p.read_text(errors="ignore") if p.exists() else ""


def _load_json(state, name: str) -> Any:
    p = _artifact(state, name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# generic
# ---------------------------------------------------------------------------
@register("test_document_exist")
def test_document_exist(state, **_):
    missing = [n for n in state.required_artifacts if not _artifact(state, n).exists()]
    if missing:
        return False, f"missing required artefacts: {missing}"
    return True, f"all {len(state.required_artifacts)} artefact(s) present"


# ---------------------------------------------------------------------------
# 01 rules
# ---------------------------------------------------------------------------
@register("test_gate_checklist_covers_hard_gates")
def test_gate_checklist_covers_hard_gates(state, **_):
    text = _read(state, "gate_checklist.md").lower()
    if not text:
        return False, "gate_checklist.md empty"
    required = ["agent.yaml", "tool", "include", "budget", "extension"]
    missing = [r for r in required if r not in text]
    if missing:
        return False, f"checklist does not mention: {missing}"
    return True, "checklist covers the harness gate families"


# ---------------------------------------------------------------------------
# 02 candidates
# ---------------------------------------------------------------------------
@register("test_candidate_ladder_is_ablation")
def test_candidate_ladder_is_ablation(state, **_):
    data = _load_json(state, "ablation_plan.json")
    if not isinstance(data, list) or len(data) < 3:
        return False, "ablation_plan.json must list >=3 candidates"
    if len({row.get("delta") for row in data}) != len(data):
        return False, "each candidate must add a distinct engineering delta"
    if any(not row.get("boundary") for row in data):
        return False, "every candidate must name the failure boundary it addresses"
    return True, f"{len(data)} candidates, all deltas distinct"


# ---------------------------------------------------------------------------
# 03 assembly
# ---------------------------------------------------------------------------
@register("test_every_candidate_has_agent_yaml")
def test_every_candidate_has_agent_yaml(state, **_):
    data = _load_json(state, "build_manifest.json")
    if not isinstance(data, dict) or not data.get("candidates"):
        return False, "build_manifest.json has no candidates"
    bad = [c["name"] for c in data["candidates"] if "agent.yaml" not in c.get("files", [])]
    if bad:
        return False, f"no agent.yaml in: {bad}"
    zips = [c for c in data["candidates"] if not c.get("zip")]
    if zips:
        return False, f"no submission.zip built for: {[c['name'] for c in zips]}"
    return True, f"{len(data['candidates'])} candidates built, all with agent.yaml at root"


# ---------------------------------------------------------------------------
# 04 gates
# ---------------------------------------------------------------------------
@register("test_gate_report_records_every_candidate")
def test_gate_report_records_every_candidate(state, **_):
    data = _load_json(state, "gate_report.json")
    if not isinstance(data, dict) or not data.get("candidates"):
        return False, "gate_report.json missing candidates"
    rows = data["candidates"]
    if any("checks" not in r for r in rows):
        return False, "a candidate row has no checks"
    if any("hard" not in " ".join(c.get("level", "") for c in r["checks"]) for r in rows):
        return False, "no gate is marked as a hard gate"
    return True, f"{len(rows)} candidates gated, hard gates reported"


# ---------------------------------------------------------------------------
# 05 proxy
# ---------------------------------------------------------------------------
@register("test_proxy_report_has_evidence_or_absence_is_declared")
def test_proxy_report_has_evidence_or_absence_is_declared(state, **_):
    data = _load_json(state, "proxy_report.json")
    if not isinstance(data, dict) or not data.get("candidates"):
        return False, "proxy_report.json missing candidates"
    for row in data["candidates"]:
        executed = row.get("executed", 0)
        if executed == 0 and not row.get("note"):
            return False, (f"{row.get('name')}: no checks executed and no note "
                           f"declaring why")
    return True, "every candidate either has executed evidence or a declared reason"


# ---------------------------------------------------------------------------
# 06 ranking
# ---------------------------------------------------------------------------
@register("test_ranking_derived_from_gates_and_rubric")
def test_ranking_derived_from_gates_and_rubric(state, **_):
    data = _load_json(state, "ranking.json")
    if not isinstance(data, dict) or not data.get("order"):
        return False, "ranking.json missing order"
    rule = json.dumps(data.get("ranking_rule", {})).lower()
    if "rubric" not in rule or "compliance" not in rule:
        return False, "ranking_rule must state the compliance and rubric terms"
    rows = data.get("candidates", [])
    for r in rows:
        if not r.get("compliant") and r.get("composite", 0) > 0:
            return False, f"{r.get('name')}: non-compliant candidate has a positive composite"
    order = data["order"]
    comps = {r["name"]: r["composite"] for r in rows}
    if [comps[n] for n in order] != sorted((comps[n] for n in order), reverse=True):
        return False, "order is not sorted by composite"
    return True, f"{len(order)} candidates ranked by the stated rule"


@register("test_ranking_declares_limits")
def test_ranking_declares_limits(state, **_):
    text = (_read(state, "ranking.md") + json.dumps(_load_json(state, "ranking.json") or {})).lower()
    needles = ["not", "official", "score"]
    if not all(n in text for n in needles):
        return False, "ranking must state that no official score is claimed"
    return True, "ranking states that no official score is claimed"


# ---------------------------------------------------------------------------
# 07 packaging
# ---------------------------------------------------------------------------
@register("test_best_candidate_passes_hard_gates")
def test_best_candidate_passes_hard_gates(state, **_):
    data = _load_json(state, "submission_summary.json")
    if not isinstance(data, dict):
        return False, "submission_summary.json missing"
    if not data.get("compliant"):
        return False, "the selected candidate does not pass every hard gate"
    if not data.get("hard_failures") == []:
        return False, f"hard failures present: {data.get('hard_failures')}"
    zip_path = data.get("zip")
    if not zip_path or not Path(zip_path).exists():
        return False, f"selected zip missing on disk: {zip_path}"
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        if "agent.yaml" not in zf.namelist():
            return False, "agent.yaml is not at the archive root"
    return True, "selected candidate compiles and agent.yaml is at the zip root"


@register("test_submission_summary_declares_no_official_score")
def test_submission_summary_declares_no_official_score(state, **_):
    data = _load_json(state, "submission_summary.json")
    if not isinstance(data, dict):
        return False, "submission_summary.json missing"
    if data.get("official_score") not in (None, "not computed", "n/a"):
        return False, f"an official score is being claimed: {data.get('official_score')!r}"
    if data.get("official_score") is None and not data.get("official_score_note"):
        return False, "official_score is null but not explained"
    return True, "no official score claimed, and the omission is explained"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_unit_tests(state) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name in state.unit_tests:
        fn = TEST_REGISTRY.get(name)
        if fn is None:
            results[name] = {"passed": False, "message": "unknown test"}
            continue
        try:
            passed, message = fn(state)
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"{type(exc).__name__}: {exc}"
        results[name] = {"passed": bool(passed), "message": message}
    state.unit_test_results = results
    return results


def format_unit_test_results(results: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for name, r in results.items():
        lines.append(f"  [{'ok  ' if r['passed'] else 'FAIL'}] {name:<52} {r['message']}")
    return "\n".join(lines) if lines else "  (no tests)"
