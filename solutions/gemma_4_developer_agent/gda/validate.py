"""Offline reproduction of the harness's submission gates.

These checks mirror what the released ``adk-submission`` validator
(``validate_directory`` / ``discover_declared_models`` / ``compile_submission``
in version 0.2.x) enforces, plus the budget arithmetic the competition states in
prose. A candidate that fails any HARD gate cannot be uploaded at all -- the
observed failure mode is a bare "Notebook Threw Exception" with no traceback, so
catching these locally is the difference between a scored submission and a
wasted upload.

Each gate returns ``{"id", "level", "passed", "detail"}`` with level in
``{"hard", "soft"}``. Hard gates disqualify; soft gates feed the rubric.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Dict, List

from gda import paths

ALLOWED_EXT = paths.ALLOWED_EXTENSIONS
REGISTRY = set(paths.TOOL_REGISTRY)
BAD_TOOLS = set(paths.KNOWN_BAD_TOOL_NAMES)
REQUIRED_MODEL = paths.REQUIRED_MODEL

_INCLUDE_RE = re.compile(r"!include\s+([^\s#]+)")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_NAME_RE = re.compile(r"^name:\s*(\S+)", re.MULTILINE)

SHELL_EXTS = {".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"}


# ---------------------------------------------------------------------------
# tiny YAML helpers -- we need enough to read tool lists and scalar keys and we
# deliberately do not need a full YAML engine for the structural gates.
# ---------------------------------------------------------------------------

def _load_yaml(text: str):
    """Load a config the way the harness does, tolerating ``!include``.

    Plain ``yaml.safe_load`` cannot parse ``!include``, because it is a
    competition-specific tag. The harness resolves it before parsing. Here we
    substitute the *path string* so structural checks can proceed; the actual
    resolution is verified separately by the ``include_targets_resolve`` gate.
    """
    import yaml

    class _ConfigLoader(yaml.SafeLoader):
        pass

    def _unknown(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_mapping(node, deep=True)

    _ConfigLoader.add_multi_constructor("!", _unknown)
    return yaml.load(text, Loader=_ConfigLoader)


def _tool_names_from_agent(doc) -> List[str]:
    """Pull the string entries out of a ``tools:`` list."""
    names: List[str] = []
    if not isinstance(doc, dict):
        return names
    tools = doc.get("tools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                # could be {agent_tool: {...}}, {name: x, args: {...}}, etc.
                for key in item:
                    if key == "agent_tool":
                        names.append("agent_tool")
                    elif key == "name":
                        names.append(str(item[key]))
    return names


def _agent_tool_config_paths(doc) -> List[str]:
    out = []
    if isinstance(doc, dict) and isinstance(doc.get("tools"), list):
        for item in doc["tools"]:
            if isinstance(item, dict) and "agent_tool" in item:
                spec = item["agent_tool"] or {}
                if isinstance(spec, dict) and spec.get("config_path"):
                    out.append(str(spec["config_path"]))
    return out


def _resolve_include(including_file: str, target: str) -> str | None:
    """Resolve an !include path relative to the including file's directory."""
    base = posixpath.dirname(including_file)
    joined = posixpath.normpath(posixpath.join(base, target))
    return joined


def _escapes_root(rel_path: str) -> bool:
    if rel_path.startswith("/") or re.match(r"^[A-Za-z]:", rel_path):
        return True
    return rel_path == ".." or rel_path.startswith("../")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def validate(files: Dict[str, str]) -> List[Dict]:
    checks: List[Dict] = []

    def gate(cid, passed, detail="", level="hard"):
        checks.append({"id": cid, "level": level,
                       "passed": bool(passed), "detail": detail})

    entries = sorted(files)

    # --- G1: agent.yaml at the archive root -------------------------------
    gate("agent_yaml_at_root", "agent.yaml" in files,
         "" if "agent.yaml" in files else "no agent.yaml at the archive root")

    # --- G2: extension allowlist ------------------------------------------
    bad_ext = [e for e in entries if Path(e).suffix.lower() not in ALLOWED_EXT]
    gate("extension_allowlist", not bad_ext,
         f"disallowed extensions: {bad_ext[:6]}" if bad_ext else "")

    # --- G2b: no shell scripts --------------------------------------------
    shell = [e for e in entries if Path(e).suffix.lower() in SHELL_EXTS]
    gate("no_shell_scripts", not shell,
         f"shell scripts (runner only executes python): {shell}" if shell else "")

    # --- G3: archive hygiene ----------------------------------------------
    junk = [e for e in entries
            if e.startswith("__MACOSX") or e.endswith(".DS_Store")
            or "/.git/" in e or e.startswith(".git/")]
    gate("archive_hygiene", not junk,
         f"OS/git junk: {junk[:6]}" if junk else "")

    # --- G4: no path traversal --------------------------------------------
    traversal = [e for e in entries if _escapes_root(e.replace("\\", "/"))]
    gate("no_path_traversal_in_archive", not traversal,
         f"escaping entries: {traversal[:6]}" if traversal else "")

    # --- G5: agent.yaml parses and has required keys ----------------------
    doc = None
    if "agent.yaml" in files:
        try:
            doc = _load_yaml(files["agent.yaml"])
            gate("agent_yaml_parses", True)
        except Exception as exc:  # noqa: BLE001
            gate("agent_yaml_parses", False, f"{type(exc).__name__}: {exc}")
    else:
        gate("agent_yaml_parses", False, "agent.yaml missing")

    if isinstance(doc, dict):
        missing = [k for k in ("name", "model", "instruction") if not doc.get(k)]
        gate("agent_yaml_required_keys", not missing,
             f"missing/empty keys: {missing}" if missing else "")
    else:
        gate("agent_yaml_required_keys", False,
             "agent.yaml did not parse to a mapping")

    # --- G6: root model is the mandated variant ---------------------------
    model = (doc or {}).get("model")
    gate("root_model_is_mandated", model == REQUIRED_MODEL,
         f"model={model!r}, required {REQUIRED_MODEL!r}")

    # --- G7: sub-agent models also mandated -------------------------------
    bad_sub = []
    for rel, text in files.items():
        if rel.startswith("sub_agents/") and rel.endswith((".yaml", ".yml")):
            try:
                sub = _load_yaml(text)
            except Exception:  # noqa: BLE001
                bad_sub.append(f"{rel}: unparseable")
                continue
            if isinstance(sub, dict) and sub.get("model") != REQUIRED_MODEL:
                bad_sub.append(f"{rel}: model={sub.get('model')!r}")
    gate("subagent_models_mandated", not bad_sub,
         "; ".join(bad_sub) if bad_sub else "")

    # --- G8: !include targets resolve inside the root ---------------------
    problems = []
    for rel, text in files.items():
        if not rel.endswith((".yaml", ".yml", ".md")):
            continue
        for target in _INCLUDE_RE.findall(text):
            resolved = _resolve_include(rel, target)
            if resolved is None or _escapes_root(resolved):
                problems.append(f"{rel} -> {target} escapes the submission root")
            elif resolved not in files:
                problems.append(f"{rel} -> {target} (resolves to {resolved}) missing")
    gate("include_targets_resolve", not problems, "; ".join(problems[:6]))

    # --- G9: every named tool is in the closed registry -------------------
    tool_names = _tool_names_from_agent(doc)
    unknown = sorted({t for t in tool_names if t not in REGISTRY and t != "agent_tool"})
    gate("tools_in_registry", not unknown,
         f"not in the harness registry: {unknown}" if unknown else "")

    bad_named = sorted({t for t in tool_names if t in BAD_TOOLS})
    gate("no_unregistered_tool_names", not bad_named,
         f"these raise ToolNotFoundError at compile time: {bad_named}"
         if bad_named else "")

    # --- G10: agent_tool config_path exists -------------------------------
    missing_cfg = [p for p in _agent_tool_config_paths(doc)
                   if posixpath.normpath(p) not in files]
    gate("agent_tool_configs_exist", not missing_cfg,
         f"missing sub-agent configs: {missing_cfg}" if missing_cfg else "")

    # --- G11: sub-agents must be wired as AgentTools, not transfer lists ---
    has_transfer = isinstance(doc, dict) and isinstance(doc.get("sub_agents"), list)
    gate("no_subagents_transfer_wiring", not has_transfer,
         "agent.yaml uses a sub_agents: transfer list; the harness expects "
         "`- agent_tool: {config_path: ..., skip_summarization: true}`"
         if has_transfer else "")

    # --- G12: sampling config is vLLM-compatible --------------------------
    sampling_rel = None
    for rel, text in files.items():
        if rel.endswith("sampling.yaml") or rel.endswith("sampling.yml"):
            sampling_rel = rel
            break
    if sampling_rel is None and isinstance(doc, dict):
        inc = _INCLUDE_RE.findall(files.get("agent.yaml", ""))
        for t in inc:
            if "sampling" in t:
                sampling_rel = _resolve_include("agent.yaml", t)
    if sampling_rel and sampling_rel in files:
        try:
            samp = _load_yaml(files[sampling_rel]) or {}
            ok = True
            detail = []
            if samp.get("include_thoughts") is False:
                ok = False
                detail.append("include_thoughts: false (hosted vLLM expects true)")
            if "thinking_level" in samp:
                ok = False
                detail.append("thinking_level present (vLLM rejects it)")
            if samp.get("include_thoughts") and "thinking_budget" not in samp:
                ok = False
                detail.append("include_thoughts without thinking_budget")
            gate("sampling_vllm_compatible", ok, "; ".join(detail))
        except Exception as exc:  # noqa: BLE001
            gate("sampling_vllm_compatible", False, f"unparseable sampling: {exc}")
    else:
        gate("sampling_vllm_compatible", True, "no sampling config declared",
             level="soft")

    # --- G13: budget arithmetic -------------------------------------------
    if "eval_config.yaml" in files:
        try:
            ev = _load_yaml(files["eval_config.yaml"]) or {}
            tasks = ev.get("expected_tasks")
            minutes = ev.get("max_time_minutes")
            if tasks is None or minutes is None:
                gate("budget_within_ceiling", False,
                     "eval_config.yaml lacks expected_tasks/max_time_minutes")
            else:
                total = float(tasks) * float(minutes)
                gate("budget_within_ceiling",
                     total <= paths.TOTAL_BUDGET_MINUTES,
                     f"{tasks} tasks x {minutes} min = {total:.0f} min, "
                     f"ceiling {paths.TOTAL_BUDGET_MINUTES} min")
            caps = [ev.get("max_tool_calls"), ev.get("max_turns")]
            gate("budget_caps_positive",
                 all(isinstance(c, (int, float)) and c > 0 for c in caps),
                 f"caps={caps}")
        except Exception as exc:  # noqa: BLE001
            gate("budget_within_ceiling", False, f"unparseable: {exc}")
            gate("budget_caps_positive", False, f"unparseable: {exc}")
        gate("budget_declared", True)
    else:
        gate("budget_declared", False,
             "no eval_config.yaml: the agent has no per-task time ceiling, "
             "so the 12-hour total budget is unbounded", level="soft")

    # --- G14: skill manifests ---------------------------------------------
    skill_dirs = {e.split("/")[1] for e in entries
                  if e.startswith("skills/") and e.count("/") >= 2}
    bad_sk = []
    for sk in sorted(skill_dirs):
        manifest = f"skills/{sk}/SKILL.md"
        if manifest not in files:
            bad_sk.append(f"{sk}: no SKILL.md")
            continue
        m = _FRONTMATTER_RE.match(files[manifest])
        if not m or not _NAME_RE.search(m.group(1)):
            bad_sk.append(f"{sk}: SKILL.md frontmatter missing `name:`")
    gate("skill_manifests_valid", not bad_sk,
         "; ".join(bad_sk) if bad_sk else "")

    # --- G15: adapter references have real files --------------------------
    adapter_refs = set()
    if isinstance(doc, dict) and doc.get("adapter"):
        adapter_refs.add(str(doc["adapter"]))
    for rel, text in files.items():
        if rel.startswith("sub_agents/") and rel.endswith((".yaml", ".yml")):
            try:
                sub = _load_yaml(text) or {}
            except Exception:  # noqa: BLE001
                continue
            if sub.get("adapter"):
                adapter_refs.add(str(sub["adapter"]))
    missing_ad = []
    for a in sorted(adapter_refs):
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            if f"adapters/{a}/{fname}" not in files:
                missing_ad.append(f"adapters/{a}/{fname}")
    gate("adapter_references_resolve", not missing_ad,
         f"missing: {missing_ad}" if missing_ad else "")
    if not adapter_refs:
        checks.append({"id": "adapter_optional", "level": "soft",
                       "passed": True,
                       "detail": "no LoRA adapter declared (optional in this competition)"})

    return checks


def hard_failures(checks: List[Dict]) -> List[Dict]:
    return [c for c in checks if c["level"] == "hard" and not c["passed"]]


def soft_failures(checks: List[Dict]) -> List[Dict]:
    return [c for c in checks if c["level"] == "soft" and not c["passed"]]


def format_report(name: str, checks: List[Dict]) -> str:
    hard = hard_failures(checks)
    lines = [f"candidate {name}: "
             f"{'COMPILES' if not hard else 'REJECTED'} "
             f"({len(checks) - len(hard)}/{len(checks)} hard gates pass)"]
    for c in checks:
        mark = "ok  " if c["passed"] else ("FAIL" if c["level"] == "hard" else "warn")
        lines.append(f"  [{mark}] {c['id']:<32} {c['detail']}")
    return "\n".join(lines)
