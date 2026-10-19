# Gemma 4 Developer Agent — solution package

A competition-specific package for
[kaggle.com/competitions/gemma-4-developer-agent](https://www.kaggle.com/competitions/gemma-4-developer-agent),
plus the AutoKaggle **DevAgent-mode** SOP that drives it.

## Why this competition needs a different AutoKaggle

The 2026 Gemma 4 Developer Agent competition is a *code-agent* track. AutoKaggle's
default SOP is built for tabular prediction, so almost nothing carries over:

| | tabular competition (AutoKaggle default) | gemma-4-developer-agent (this package) |
|---|---|---|
| inputs | `train.csv` / `test.csv` | a competition brief; **no data at all** |
| objective | one numeric metric | PASS/FAIL of patches on hidden repos |
| output | `submission.csv` | `submission.zip` = `agent.yaml` + prompts + skills |
| model | none (sklearn/xgboost) | fixed: `gemma-4-31b-it-qat-w4a16-ct` |
| scoring | computed locally | **computed only by the organiser's harness** |
| main risk | underfitting | a config that does not even compile |

The last row is the important one. Two published post-mortems of this competition
show submissions failing upload with *"Notebook Threw Exception"* and **no
traceback** because of a `.sh` skill script, an unregistered tool name, or a
sub-agent wired as a transfer list rather than an AgentTool. A submission that
does not compile scores zero regardless of how good the prompt is.

## The honest boundary

**Nothing in this repository can produce a score for this competition.** The
official metric is the percentage of hidden repository issues whose validation
tests pass after applying the submitted patch. Computing it needs the organiser's
Docker sandbox, the `gemma-4-31b-it-qat-w4a16-ct` weights, and a hidden task set
(~22 GB of snapshots, graphs, embeddings and wheels).

So this package does not try. It measures the two things that *are* measurable
offline and that determine whether your submission gets to be scored at all:

1. **Harness-gate compliance** — mechanical checks reproducing what
   `adk-submission`'s validator rejects (extension allowlist, closed tool
   registry, `!include` resolution, sub-agent wiring, vLLM sampling keys,
   12-hour budget arithmetic).
2. **Agent engineering quality** — a deterministic rubric over the candidate's
   own files, plus a proxy that *executes* the scripts the candidate ships
   against fixture repositories.

## Candidate ladder

Five versions, each adding exactly one engineering idea, so the comparison is an
ablation rather than five random prompts.

| candidate | adds | failure boundary addressed |
|---|---|---|
| `v1_starter_kit` | baseline shaped like the official starter kit | — |
| `v2_budget_guard` | per-task budgets, `get_status()` discipline | 12-hour ceiling over ~120 tasks |
| `v3_graph_navigator` | graph-first navigation + read-only `code_analyzer` AgentTool | "large codebases are hard to navigate" |
| `v4_skill_pack` | `repro_first` + `patch_hygiene` skills, with real scripts | plausible-but-wrong patches; rejected diffs |
| `v5_verified_full` | verification loop, context compaction, anti-cycle rule | context exhaustion, edit cycles, red-tree submits |

## Quick start

```bash
cd solutions/gemma_4_developer_agent

python -m gda.cli all          # build + gate + rubric + proxy + rank + report
python -m gda.cli show         # print the ranking
python -m gda.cli validate     # gates only
python -m gda.cli inspect v5_verified_full
```

Requires PyYAML (already a repo dependency) and, for the proxy, pytest.
matplotlib is optional and only used for the figures.

Outputs land in `artifacts/`:

```
artifacts/
├── submissions/<candidate>/submission.zip   # one zip per candidate
├── best_submission.zip                      # the selected candidate
├── evaluation/comparison.json + .csv
├── reports/REPORT.md
└── figures/fig01..fig04.png
```

## Selection rule

```
composite = compliance x ( 0.60 x rubric_norm + 0.40 x proxy_norm )
```

- `compliance` is 0/1 over the hard gates — a failing gate disqualifies.
- `proxy_norm` is the fraction of executed proxy checks passed, or **0.50** when
  the candidate ships nothing executable. The neutral prior exists so a
  prompt-only agent is neither penalised as broken nor credited with engineering
  it cannot demonstrate.

The rubric leads because it is the only signal available for every candidate.

## Adding a LoRA adapter

Adapters are optional in the competition, and no candidate here ships one:
training one needs the released task graph, the 31B base weights and GPU time,
none of which exist in this repository. Shipping an untrained adapter would pass
every format gate and *lower* the real score, so the default is to reference
none.

To attach a real one:

1. Put the PEFT output in `adapters/main_lora/` containing `adapter_config.json`
   and `adapter_model.safetensors`.
2. Add `adapter: main_lora` to the root agent in `agent.yaml` (and to any
   subagent that should use a different adapter, e.g. `adapter: tool_lora`).
3. Re-run `python -m gda.cli validate` — a dangling `adapter:` reference is a
   hard gate failure.

## Layout

```
solutions/gemma_4_developer_agent/
├── gda/
│   ├── paths.py        filesystem contract + competition constants
│   ├── candidates.py   the five strategies, as file trees
│   ├── build.py        stage + zip (deterministic bytes)
│   ├── validate.py     the harness gates
│   ├── rubric.py       deterministic engineering rubric
│   ├── proxy.py        executes shipped scripts against fixtures
│   ├── evaluate.py     build -> gate -> score -> rank -> select
│   ├── figures.py      comparison figures
│   ├── report.py       REPORT.md
│   └── cli.py          entry point
└── artifacts/          everything generated

multi_agents/devagent/  DevAgent-mode SOP (deterministic, LLM-free)
multi_agents/competition/gemma_4_developer_agent/overview.txt
```

## Limitations

- The proxy cannot run the prompt against Gemma 4. It says nothing about model
  behaviour, tool-call validity under 4-bit quantisation, or whether a patch is
  *correct* — only whether the machinery around the patch works.
- The rubric is a heuristic over prompt text. It rewards stating a practice, not
  doing it; a prompt that claims graph-first navigation without using it will
  still score here.
- Fixture repositories are three files. They prove the scripts run, not that they
  generalise.
- Gate sets change. These reproduce the 0.2.x-release behaviour; re-check against
  the harness README shipped with the competition data before the final upload.
