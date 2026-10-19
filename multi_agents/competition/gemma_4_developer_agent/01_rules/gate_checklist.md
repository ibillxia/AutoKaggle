# Harness gate checklist

Every item is either mechanically checkable in `gda/validate.py` or explicitly
noted as uncheckable offline. A candidate must pass **every hard** item.

| # | gate | level | why it exists |
|---|---|---|---|
| 1 | `agent_yaml_at_root` | hard | the harness looks for `agent.yaml` at the archive root |
| 2 | `extension_allowlist` | hard | only .json, .md, .py, .safetensors, .txt, .yaml, .yml are accepted |
| 3 | `no_shell_scripts` | hard | the runner executes python, not shell; `.sh` was a published rejection |
| 4 | `archive_hygiene` | hard | `__MACOSX/`, `.DS_Store`, `.git/` should never ship |
| 5 | `no_path_traversal_in_archive` | hard | entries must not escape the submission root |
| 6 | `agent_yaml_parses` | hard | a config the compiler cannot parse produces an opaque upload error |
| 7 | `agent_yaml_required_keys` | hard | `name`, `model`, `instruction` are mandatory |
| 8 | `root_model_is_mandated` | hard | model must be `gemma-4-31b-it-qat-w4a16-ct` |
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
