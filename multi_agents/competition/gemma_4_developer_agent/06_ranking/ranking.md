# Ranking

**Selected: `v5_verified_full`** — composite 1.000 (rubric 10.00/10, proxy 100.0%).

Rule: `compliance x (0.60 x rubric_norm + 0.40 x proxy_norm)`

| rank | candidate | composite | rubric /10 | proxy | compiles |
|---|---|---|---|---|---|
| 1 | `v5_verified_full` | 1.000 | 10.00 | 100.0% | yes |
| 2 | `v4_skill_pack` | 0.914 | 8.57 | 100.0% | yes |
| 3 | `v3_graph_navigator` | 0.565 | 6.09 | n/a | yes |
| 4 | `v2_budget_guard` | 0.445 | 4.08 | n/a | yes |
| 5 | `v1_starter_kit` | 0.236 | 0.60 | n/a | yes |

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

