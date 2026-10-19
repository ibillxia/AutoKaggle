# NFL Big Data Bowl 2027 — solution package

A competition-specific analysis pipeline for
[kaggle.com/competitions/nfl-big-data-bowl-2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027),
plus the AutoKaggle **hackathon mode** SOP that drives it.

## Why this competition needs a different AutoKaggle

The 2027 Big Data Bowl is a *judged hackathon*, not a leaderboard competition:

| | tabular competition (AutoKaggle default) | BDB 2027 (this package) |
|---|---|---|
| inputs | `train.csv` / `test.csv` | 9 CSVs, 2.2 GiB, no split |
| objective | one numeric metric | judged rubric, 4 components |
| output | `submission.csv` | ≤2,000-word writeup + public notebook |
| gates | metric improves | pass/fail: notebook attached, tracking data used |
| risk | underfitting | **false discovery** from searching thousands of trait × outcome pairs |

So the pipeline here is not `train → clean → engineer → fit → predict`. It is
`audit → screen hypotheses → confirm → build a metric → communicate`, with
multiple-comparison control and confound handling built in rather than bolted on.

## The problem in one paragraph

Combine tracking sensors record how an athlete moves during each drill; a
stopwatch only records how long the drill took. A total is not decomposable, so
two prospects with the same 3-cone time can be entirely different athletes. The
pipeline measures the *shape* of the movement — braking rate into the sharpest
cut, the fraction of entry speed retained through it, and the time taken to climb
back to speed — and asks whether that shape predicts in-game NFL performance
after every conventional Combine number and draft capital is already accounted
for.

## Quick start

```bash
# from the repository root
python3 -m venv .venv && .venv/bin/python -m pip install -r solutions/nfl_big_data_bowl_2027/requirements.txt

# 1. credentials (see below), then fetch the data
.venv/bin/python -m bdb.cli download --dest multi_agents/competition/nfl_big_data_bowl_2027/data

# 2. run the whole pipeline
cd solutions/nfl_big_data_bowl_2027
.venv/../../.venv/bin/python -m bdb.cli all        # audit + features + feasibility screen
.venv/../../.venv/bin/python -m bdb.cli submission # figures + writeup + notebook

# 3. verify the deliverable against the competition's own gates
cd ../..
.venv/bin/python multi_agents/hackathon/run.py selfcheck
```

### Data access

The competition data is CC BY-NC 4.0 and requires accepting the rules.

1. Sign in to Kaggle → <https://www.kaggle.com/settings> → **API** → **Create New Token**
2. Move the downloaded file to `~/.kaggle/kaggle.json` (`chmod 600`), **or** place it at the
   repository root and set `KAGGLE_CONFIG_DIR` to the repository root
3. Accept the rules at <https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules>
4. `python -m bdb.cli download`

Without credentials you can still exercise the entire pipeline on schema-faithful
synthetic data that embeds a **known** ground-truth signal:

```bash
python -m bdb.cli synthetic --out /tmp/bdb/data --n-players 510
BDB_DATA_DIR=/tmp/bdb/data python -m bdb.cli all
BDB_DATA_DIR=/tmp/bdb/data python -m bdb.cli submission
```

## Pipeline stages

| command | what it does | output |
|---|---|---|
| `bdb.cli info` | resolve paths, list missing files | — |
| `bdb.cli download` | fetch + unzip via the Kaggle API | `data/*.csv` |
| `bdb.cli audit` | join integrity, effective N, drill coverage, opt-outs, exposure | `phase_a_audit.json` |
| `bdb.cli features` | 10 Hz → kinematics → per-player traits → in-game outcomes | parquet caches |
| `bdb.cli analyze` | two-stage feasibility screen with FDR control | `phase_d_feasibility.csv` |
| `bdb.cli deepdive` | bootstrap CIs, leave-one-out, metric, blind spots | `deep_dive.json` |
| `bdb.cli submission` | 9 figures + writeup + notebook + phase documents | `08_writeup/`, `07_visualization/` |

### `bdb.kinematics` — the analytical core

Per drill attempt and 10 Hz frame, from smoothed signals only (the raw `a`
column is a finite difference and is never used):

* **speed** — peak / mean speed, peak acceleration, time to peak
* **path** — path length, net displacement, path efficiency
* **turn** — detected cuts via angular velocity (>60 °/s sustained, >25° total)
* **brake** — peak deceleration into the cut
* **reaccel** — peak acceleration out of the cut, and `regain_time_90`, the
  seconds to climb back to 90% of entry speed (censored at 2 s, not `NaN`)
* **asym** — signed cut direction, for left/right asymmetry

## Hackathon mode SOP

`multi_agents/hackathon/` extends AutoKaggle with a judged-writeup SOP. The
control flow is deliberately identical to the original (Reader → Planner →
Developer → Reviewer → Summarizer, reviewer-driven retries, a per-phase attempt
budget). What changes is what is scored and what gates advancement.

Eight phases:

1. Understand Background and Rubric
2. Data Audit and Sensor Inventory
3. Hypothesis Generation and Feasibility Scan
4. Topic Selection and Analysis Plan
5. Deep Dive and Statistical Validation
6. Metric Construction and Robustness
7. Visualization and Narrative
8. Writeup Assembly and Rubric Self-Review

```bash
python multi_agents/hackathon/run.py check        # validate config (18 unit tests)
python multi_agents/hackathon/run.py run          # full SOP (needs api_key.txt)
python multi_agents/hackathon/run.py phase "Data Audit and Sensor Inventory"
python multi_agents/hackathon/run.py selfcheck    # gate tests against artefacts on disk
```

The Developer agent writes code, **executes it**, and reports only what the code
printed — so an agent turn cannot assert a result the analysis did not produce.
`selfcheck` re-runs the 18 gate tests against whatever is currently on disk, so
the pipeline and the SOP are verifiable independently.

## Statistical safeguards

These are the reason to trust (or distrust) any number this package prints.

| hazard | mitigation |
|---|---|
| draft-capital confounding | `log_draft_capital` + UDFA indicator in every model; per-play rate outcomes |
| position heterogeneity | every analysis stratified by position group; traits z-scored within group |
| unequal exposure (2023 vs 2025 class) | per-season normalisation; season-clustered bootstrap |
| 20–35% Combine opt-out rates | median imputation **plus** explicit `*_not_tested` indicators |
| multiple comparisons | pre-registered pairs, BH-FDR at q < 0.10 |
| overfitting with n ≈ 60–120 | compact baseline block, ridge with inner CV, repeated 5-fold |
| single-player artefacts | leave-one-out influence; sign-flip count reported |
| unreliable traits | split-half (odd/even attempt) reliability, Spearman-Brown corrected |
| over-claimed precision | percentile bootstrap CIs, not normal-theory intervals |
| tracking noise | Savitzky-Golay smoothing before any differentiation |

## Synthetic-data self-test

The synthetic generator embeds a latent change-of-direction skill that drives
in-game outcomes **and** is deliberately absent from the traditional Combine
numbers. A correct pipeline must recover it. Verified on 510 synthetic prospects:

* effective N recovered from 28 → 112 after fixing complete-case deletion
* the embedded sensor-only signal is detected (q ≪ 0.10, ΔR² up to +0.38)
* noise traits are penalised (ΔR² down to −1.00; 149 of 276 pairs rejected)
* all 18 hackathon gate tests pass on the generated deliverable

## Layout

```
solutions/nfl_big_data_bowl_2027/
├── bdb/
│   ├── paths.py         filesystem contract (data in, artefacts out)
│   ├── schema.py        column contracts + position/drill mappings
│   ├── io.py            validated, parquet-cached loaders
│   ├── audit.py         phase A
│   ├── kinematics.py    phase B — 10 Hz → movement traits
│   ├── ingame.py        phase C — in-game outcomes
│   ├── feasibility.py   phase D — two-stage screen with FDR
│   ├── stats.py         bootstrap, HC3 OLS, influence, metric helpers
│   ├── deepdive.py      phase E/F — confirmation + metric construction
│   ├── figures.py       phase G — the 9-figure suite
│   ├── phases.py        deterministic phase documents (no LLM)
│   ├── report.py        phase H — writeup + notebook + self-review
│   ├── synthetic.py     schema-faithful data with a known answer
│   └── cli.py           entry point

multi_agents/hackathon/  hackathon-mode SOP
multi_agents/competition/nfl_big_data_bowl_2027/
├── overview.txt         Reader input (full competition brief)
├── 01_background/ … 08_writeup/
```

## Limitations

* Three draft classes cannot separate a real positional effect from a
  class-specific one.
* In-game outcomes such as separation and get-off are themselves model-derived
  Next Gen Stats quantities, so measurement error attenuates every correlation
  reported here.
* Metric component weights are equal by design and **not** fitted to the
  outcome. Fitting them would raise predictive power and destroy the metric's
  out-of-sample honesty; that trade is made deliberately.
