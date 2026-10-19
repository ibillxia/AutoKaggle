# Ablation plan

## v1_starter_kit

- **Adds:** baseline
- **Expected effect:** Baseline shaped like the official starter kit: prompt-only, no budgets, no skills.
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.

## v2_budget_guard

- **Adds:** 12-hour budget ceiling across ~120 tasks
- **Expected effect:** Adds explicit per-task budgets and a get_status-before-everything policy.
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.

## v3_graph_navigator

- **Adds:** context exhaustion / blind grep on large repositories
- **Expected effect:** Adds graph-first navigation and a read-only code_analyzer AgentTool.
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.

## v4_skill_pack

- **Adds:** plausible-but-wrong patches; malformed or test-editing diffs
- **Expected effect:** Adds reproduce-first and patch-hygiene skills, with real scripts.
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.

## v5_verified_full

- **Adds:** context bloat, edit cycles, submitting a red tree
- **Expected effect:** Adds the verification loop, context compaction and anti-cycle rules.
- **How it is measured:** rubric signals in the corresponding dimension, plus the
  proxy checks for any shipped script.
