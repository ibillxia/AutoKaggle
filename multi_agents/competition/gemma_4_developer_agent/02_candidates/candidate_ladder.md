# Candidate ladder — 5 versions

Each candidate adds exactly one engineering idea, so the comparison is an
ablation: the difference between two adjacent candidates is attributable to the
single delta between them.

| # | candidate | idea | failure boundary it addresses |
|---|---|---|---|
| 1 | `v1_starter_kit` | Baseline shaped like the official starter kit: prompt-only, no budgets, no skills. | baseline |
| 2 | `v2_budget_guard` | Adds explicit per-task budgets and a get_status-before-everything policy. | 12-hour budget ceiling across ~120 tasks |
| 3 | `v3_graph_navigator` | Adds graph-first navigation and a read-only code_analyzer AgentTool. | context exhaustion / blind grep on large repositories |
| 4 | `v4_skill_pack` | Adds reproduce-first and patch-hygiene skills, with real scripts. | plausible-but-wrong patches; malformed or test-editing diffs |
| 5 | `v5_verified_full` | Adds the verification loop, context compaction and anti-cycle rules. | context bloat, edit cycles, submitting a red tree |

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
