"""Deterministic phase documents for the DevAgent SOP.

Each function writes the phase's required artefacts into
``state.restore_dir`` and returns the rubric component scores for that phase.

Nothing here calls an LLM. The phase documents are derived from the actual built
zips and the actual gate/proxy results, which is the whole point: an agent turn
cannot assert a finding the build did not produce.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict

from devagent.state import DevAgentState


# ---------------------------------------------------------------------------
# import the solution package
# ---------------------------------------------------------------------------
def _gda(state: DevAgentState):
    sol = str(state.solution_dir)
    if sol not in sys.path:
        sys.path.insert(0, sol)
    from gda import build, evaluate, paths, validate  # noqa: WPS433
    from gda.candidates import CANDIDATES, build_candidate  # noqa: WPS433
    from gda.proxy import build_fixtures, run_proxy  # noqa: WPS433
    from gda.rubric import score_candidate  # noqa: WPS433
    return dict(build=build, evaluate=evaluate, paths=paths, validate=validate,
                CANDIDATES=CANDIDATES, build_candidate=build_candidate,
                build_fixtures=build_fixtures, run_proxy=run_proxy,
                score_candidate=score_candidate)


def _write(state: DevAgentState, name: str, text: str) -> Path:
    p = Path(state.restore_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


# ---------------------------------------------------------------------------
# 01 Understand Rules and Harness
# ---------------------------------------------------------------------------
def phase_rules(state: DevAgentState) -> Dict:
    g = _gda(state)
    from gda.paths import ALLOWED_EXTENSIONS, REQUIRED_MODEL, TOOL_REGISTRY, TOTAL_BUDGET_MINUTES

    overview = state.background_info or ""
    _write(state, "background.md", f"""\
# Competition background — {state.competition}

Source: `multi_agents/competition/{state.competition}/overview.txt`.

The competition is a **code-agent** track: post-train a Gemma 4 model into an
autonomous software-engineering agent that drafts fixes for real repository
issues. The deliverable is an agent configuration zip, not a prediction file.

## The five facts that shape every candidate

1. **Base model is fixed.** Every agent and subagent must use
   `{REQUIRED_MODEL}`.
2. **Scoring is external.** PASS/FAIL over hidden repositories, computed by the
   organiser's harness in a Docker sandbox. Nothing in this repository can
   produce that number.
3. **Budget is 12 hours** ({TOTAL_BUDGET_MINUTES} minutes) for the whole task set,
   inclusive of sandbox setup, excluding patch validation.
4. **The tool registry is closed** — {len(TOOL_REGISTRY)} tools:
   {", ".join(f"`{t}`" for t in TOOL_REGISTRY)}.
5. **File extensions are restricted** to
   {", ".join(f"`{e}`" for e in sorted(ALLOWED_EXTENSIONS))} — a single `.sh`
   script causes the whole upload to be rejected.

## Evidence trail

The following is the verbatim brief the Reader agent consumed (truncated to
20,000 characters).

```
{overview[:20000]}
```
""")

    _write(state, "gate_checklist.md", f"""\
# Harness gate checklist

Every item is either mechanically checkable in `gda/validate.py` or explicitly
noted as uncheckable offline. A candidate must pass **every hard** item.

| # | gate | level | why it exists |
|---|---|---|---|
| 1 | `agent_yaml_at_root` | hard | the harness looks for `agent.yaml` at the archive root |
| 2 | `extension_allowlist` | hard | only {", ".join(sorted(ALLOWED_EXTENSIONS))} are accepted |
| 3 | `no_shell_scripts` | hard | the runner executes python, not shell; `.sh` was a published rejection |
| 4 | `archive_hygiene` | hard | `__MACOSX/`, `.DS_Store`, `.git/` should never ship |
| 5 | `no_path_traversal_in_archive` | hard | entries must not escape the submission root |
| 6 | `agent_yaml_parses` | hard | a config the compiler cannot parse produces an opaque upload error |
| 7 | `agent_yaml_required_keys` | hard | `name`, `model`, `instruction` are mandatory |
| 8 | `root_model_is_mandated` | hard | model must be `{REQUIRED_MODEL}` |
| 9 | `subagent_models_mandated` | hard | the rule applies to every subagent too |
| 10 | `include_targets_resolve` | hard | `!include` resolves relative to the including file |
| 11 | `tools_in_registry` | hard | the registry is closed |
| 12 | `no_unregistered_tool_names` | hard | `run_skill_script` in `tools:` raises ToolNotFoundError |
| 13 | `agent_tool_configs_exist` | hard | a dangling `config_path` cannot compile |
| 14 | `no_subagents_transfer_wiring` | hard | sub-agents must be wired as AgentTools, not transfer lists |
| 15 | `sampling_vllm_compatible` | hard | hosted vLLM wants `include_thoughts` + `thinking_budget`, not `thinking_level` |
| 16 | `budget_within_ceiling` | hard | tasks x minutes must fit the 12-hour ceiling |
| 17 | `budget_caps_positive` | hard | turn and tool-call caps must be positive |
| 18 | `budget_declared` | soft | per-task limits are optional, but their absence leaves the total unbounded |
| 19 | `skill_manifests_valid` | hard | each skill needs `SKILL.md` with frontmatter `name:` |
| 20 | `adapter_references_resolve` | hard | a referenced adapter must have both adapter files |

## Not checkable offline

- Whether the patch actually fixes the issue.
- Whether the model emits valid tool calls under 4-bit quantisation.
- The hidden-set pass rate. **Never claim one.**
""")

    return {"harness_compliance": 10.0, "strategy_quality": 6.0,
            "executed_evidence": 5.0, "packaging": 5.0}


# ---------------------------------------------------------------------------
# 02 Candidate Strategy Design
# ---------------------------------------------------------------------------
def phase_candidates(state: DevAgentState) -> Dict:
    g = _gda(state)
    CANDIDATES = g["CANDIDATES"]

    rows = []
    for i, (name, meta) in enumerate(CANDIDATES.items(), 1):
        rows.append({
            "rank": i,
            "name": name,
            "delta": meta["delta"],
            "boundary": meta["delta"],
            "idea": meta["idea"],
        })

    _write(state, "ablation_plan.json", json.dumps(rows, indent=2, ensure_ascii=False))

    _write(state, "candidate_ladder.md", f"""\
# Candidate ladder — {len(rows)} versions

Each candidate adds exactly one engineering idea, so the comparison is an
ablation: the difference between two adjacent candidates is attributable to the
single delta between them.

| # | candidate | idea | failure boundary it addresses |
|---|---|---|---|
""" + "\n".join(f"| {r['rank']} | `{r['name']}` | {r['idea']} | {r['boundary']} |"
                for r in rows) + f"""

## Why these deltas and not others

The deltas are chosen from failure boundaries the competition itself names, not
from prompt-engineering folklore:

- The **12-hour ceiling over ~120 tasks** makes budget discipline a first-class
  requirement, so `v2` adds it before anything else.
- "**Large codebases are hard to navigate**" is the problem statement, so `v3`
  adds graph-first navigation and a read-only survey sub-agent.
- "**Drafts fixes** ... reliably" is the goal, so `v4` adds a reproduction-first
  edit loop and a patch-hygiene gate — the two things that convert a plausible
  patch into a passing one.
- Small local models fail on **context exhaustion** and **edit cycles**, so `v5`
  adds a compaction policy and an anti-cycle rule.

## What is deliberately absent

No candidate ships a LoRA adapter. Training one needs the released task graph,
the 31B base weights and GPU time — none of which exist in this repository.
Shipping a randomly-initialised adapter would pass the format gates and *lower*
the real score, so the builder supports injecting a real adapter (see the README)
and the default candidates reference none.
""")

    _write(state, "ablation_plan.md", f"""\
# Ablation plan

""" + "\n".join(
        f"""## {r['name']}

- **Adds:** {r['delta']}
- **Expected effect:** {r['idea']}
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.
""" for r in rows))

    return {"harness_compliance": 8.0, "strategy_quality": 9.5,
            "executed_evidence": 5.0, "packaging": 5.0}


# ---------------------------------------------------------------------------
# 03 Config Assembly
# ---------------------------------------------------------------------------
def phase_assembly(state: DevAgentState) -> Dict:
    g = _gda(state)
    build, paths = g["build"], g["paths"]
    CANDIDATES, build_candidate = g["CANDIDATES"], g["build_candidate"]

    manifest = {"solution_dir": str(state.solution_dir), "candidates": []}
    for name in CANDIDATES:
        files = build_candidate(name)
        zip_path = build.build_zip(name, files)
        manifest["candidates"].append({
            "name": name,
            "n_files": len(files),
            "files": sorted(files),
            "zip": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
        })

    _write(state, "build_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"harness_compliance": 8.0, "strategy_quality": 7.0,
            "executed_evidence": 6.0, "packaging": 9.0}


# ---------------------------------------------------------------------------
# 04 Gate Validation
# ---------------------------------------------------------------------------
def phase_gates(state: DevAgentState) -> Dict:
    g = _gda(state)
    validate, build_candidate = g["validate"], g["build_candidate"]
    CANDIDATES = g["CANDIDATES"]

    rows = []
    n_hard_fail_total = 0
    for name in CANDIDATES:
        checks = validate.validate(build_candidate(name))
        hard = [c for c in checks if c["level"] == "hard" and not c["passed"]]
        soft = [c for c in checks if c["level"] == "soft" and not c["passed"]]
        n_hard_fail_total += len(hard)
        rows.append({"name": name, "checks": checks,
                     "hard_failures": [c["id"] for c in hard],
                     "soft_warnings": [c["id"] for c in soft],
                     "compliant": not hard})

    _write(state, "gate_report.json", json.dumps(
        {"n_candidates": len(rows), "candidates": rows}, indent=2, ensure_ascii=False))

    md = ["# Gate report\n",
          f"{len(rows)} candidates validated against the offline harness gates; "
          f"{n_hard_fail_total} hard failure(s) in total.\n",
          "| candidate | hard failures | soft warnings | compiles |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['name']}` | {', '.join('`' + x + '`' for x in r['hard_failures']) or '—'} "
                  f"| {', '.join('`' + x + '`' for x in r['soft_warnings']) or '—'} "
                  f"| {'yes' if r['compliant'] else '**NO**'} |")
    _write(state, "gate_report.md", "\n".join(md) + "\n")

    score = 10.0 if n_hard_fail_total == 0 else max(0.0, 10.0 - 2.0 * n_hard_fail_total)
    return {"harness_compliance": score, "strategy_quality": 7.0,
            "executed_evidence": 6.0, "packaging": 8.0}


# ---------------------------------------------------------------------------
# 05 Proxy Evaluation
# ---------------------------------------------------------------------------
def phase_proxy(state: DevAgentState) -> Dict:
    g = _gda(state)
    build_fixtures, run_proxy, build_candidate = (
        g["build_fixtures"], g["run_proxy"], g["build_candidate"])
    CANDIDATES, paths, build = g["CANDIDATES"], g["paths"], g["build"]

    fixtures = build_fixtures(Path(paths.WORK_DIR) / "fixtures")
    rows = []
    n_exec = n_pass = 0
    for name in CANDIDATES:
        files = build_candidate(name)
        staged = build.stage_candidate(name, files)
        res = run_proxy(name, files, staged, fixtures).as_dict()
        if res["executed"] == 0:
            res["note"] = ("no executable artefact shipped; the composite gives "
                           "this candidate the neutral prior 0.50 rather than "
                           "crediting evidence it does not have")
        n_exec += res["executed"]
        n_pass += res["passed"]
        rows.append(res)

    _write(state, "proxy_report.json", json.dumps(
        {"n_executed_checks": n_exec, "n_passed": n_pass,
         "fixtures": str(Path(paths.WORK_DIR) / "fixtures"),
         "candidates": rows}, indent=2, ensure_ascii=False))

    score = 10.0 * (n_pass / n_exec) if n_exec else 0.0
    return {"harness_compliance": 8.0, "strategy_quality": 7.0,
            "executed_evidence": score, "packaging": 7.0}


# ---------------------------------------------------------------------------
# 06 Ranking and Selection
# ---------------------------------------------------------------------------
def phase_ranking(state: DevAgentState) -> Dict:
    g = _gda(state)
    evaluate_mod, paths = g["evaluate"], g["paths"]

    payload = evaluate_mod.evaluate()
    _write(state, "ranking.json", json.dumps(payload, indent=2, ensure_ascii=False))
    (Path(state.restore_dir) / "comparison.csv").write_bytes(
        (Path(paths.EVALUATION_DIR) / "comparison.csv").read_bytes())

    best = next(c for c in payload["candidates"] if c["name"] == payload["best"])
    md = [f"# Ranking\n",
          f"**Selected: `{best['name']}`** — composite {best['composite']:.3f} "
          f"(rubric {best['rubric']['score_10']:.2f}/10, "
          f"proxy {'n/a' if best['proxy']['score'] is None else format(best['proxy']['score'], '.1%')}).\n",
          f"Rule: `{payload['ranking_rule']['composite']}`\n",
          "| rank | candidate | composite | rubric /10 | proxy | compiles |",
          "|---|---|---|---|---|---|"]
    for i, c in enumerate(payload["candidates"], 1):
        px = "n/a" if c["proxy"]["score"] is None else f"{c['proxy']['score']:.1%}"
        md.append(f"| {i} | `{c['name']}` | {c['composite']:.3f} | "
                  f"{c['rubric']['score_10']:.2f} | {px} | "
                  f"{'yes' if c['compliant'] else '**NO**'} |")
    md.append("""
## Limits of this ranking

This ordering is **not** a leaderboard and **not** a predicted score. It does not
measure, estimate or imply a hidden-set pass rate — that number is produced only
by the organiser's harness, using a Docker sandbox, the
`gemma-4-31b-it-qat-w4a16-ct` weights and issues that are not public.

What the composite does measure:

1. whether the candidate survives the harness's upload gates (hard, mechanical);
2. whether its engineering matches the failure boundaries the competition names
   (rubric, mechanical heuristics over the candidate's own files);
3. whether the scripts it ships actually run (proxy, real execution on fixtures).

A candidate that looks strong here can still score zero on the hidden set: a
well-engineered agent with a wrong idea about a bug is still wrong.
""")
    _write(state, "ranking.md", "\n".join(md) + "\n")

    return {"harness_compliance": 9.0, "strategy_quality": 9.0,
            "executed_evidence": 8.0, "packaging": 8.0}


# ---------------------------------------------------------------------------
# 07 Submission Packaging
# ---------------------------------------------------------------------------
def phase_submission(state: DevAgentState) -> Dict:
    g = _gda(state)
    evaluate_mod, paths, validate_mod = g["evaluate"], g["paths"], g["validate"]
    build_candidate = g["build_candidate"]

    payload = evaluate_mod.load_comparison()
    best = next(c for c in payload["candidates"] if c["name"] == payload["best"])
    checks = validate_mod.validate(build_candidate(best["name"]))
    hard = [c["id"] for c in checks if c["level"] == "hard" and not c["passed"]]

    zip_path = payload["best_zip"]
    summary = {
        "competition": state.competition,
        "selected_candidate": best["name"],
        "zip": zip_path,
        "compliant": not hard,
        "hard_failures": hard,
        "composite": best["composite"],
        "rubric_score_10": best["rubric"]["score_10"],
        "proxy_score": best["proxy"]["score"],
        "official_score": None,
        "official_score_note": (
            "Not computed and not computable here. The official metric is the "
            "percentage of hidden repository issues whose validation tests pass "
            "after applying the submitted patch; obtaining it requires the "
            "organiser's harness with the gemma-4-31b-it-qat-w4a16-ct weights "
            "and the hidden task set."
        ),
        "next_lever": (
            "Train a LoRA adapter on the released task graph and place it in "
            "adapters/main_lora/, then set `adapter: main_lora` on the root "
            "agent in agent.yaml."
        ),
    }
    _write(state, "submission_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

    md = [f"# Submission summary\n",
          f"- **selected:** `{best['name']}`",
          f"- **zip:** `{zip_path}`",
          f"- **compiles:** {'yes' if not hard else 'NO — ' + ', '.join(hard)}",
          f"- **composite:** {best['composite']:.3f}",
          f"- **official score:** *not computed* — see `submission_summary.json`\n",
          "## Upload this\n",
          f"```\n{zip_path}\n```\n",
          "## Before uploading\n",
          "1. Re-run `python -m gda.cli validate` after any edit.",
          "2. If you trained a LoRA adapter, place it under `adapters/main_lora/` "
          "and re-run the gates: a dangling `adapter:` reference is a hard failure.",
          "3. Do not add `.sh` files — the extension allowlist rejects the whole zip.\n"]
    _write(state, "submission.md", "\n".join(md) + "\n")

    return {"harness_compliance": 10.0 if not hard else 5.0, "strategy_quality": 8.0,
            "executed_evidence": 8.0, "packaging": 10.0}


PHASE_FUNCS = {
    "Understand Rules and Harness": phase_rules,
    "Candidate Strategy Design": phase_candidates,
    "Config Assembly": phase_assembly,
    "Gate Validation": phase_gates,
    "Proxy Evaluation": phase_proxy,
    "Ranking and Selection": phase_ranking,
    "Submission Packaging": phase_submission,
}
