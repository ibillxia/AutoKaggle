# Background brief (deterministic extract)

## Problem statement
Build an analytical framework that uncovers non-obvious, actionable linkages between 10 Hz Combine sensor tracking and regular-season NFL game performance, looking beyond conventional scouting wisdom. The organizers explicitly invite a narrow focus -- one drill, one position group, or one aspect of movement -- rather than a survey of everything.

## Deliverable and hard limits
- at most **2000 words**
- fewer than **10 tables or figures**
- a public Kaggle notebook is **required**
- the writeup **must** use the player tracking data (enforced)

## Pass / fail gates
- Explicitly links Combine sensor tracking data to regular-season NFL game performance.
- Readable markdown writeup with embedded visuals and an attached public Kaggle Notebook.

## Scored components and weights
- **Football** (30%)
- **Data Science** (30%)
- **Writeup** (20%)
- **Visualization** (20%)

## Data inventory
Nine CSV tables. The four that carry the analysis: `combine_tracking.csv` (10 Hz Combine drills, the sensor layer), `combine_results.csv` (the conventional stopwatch battery to be beaten), `player_play.csv` (in-game Next Gen Stats outcomes), and `game_tracking_2023/2024/2025.csv` (10 Hz in-game tracking, used to re-derive movement quantities independently of the vendor columns).

## Validity hazards, ranked
1. **Draft capital confounding.** Snap counts and targets follow the draft board, so any trait-outcome correlation can be an artefact of draft position.
2. **Small N.** Tens of players per position group. Complex models will not validate; uncertainty must be reported.
3. **Multiple comparisons.** Hundreds of candidate trait x outcome pairs guarantee false positives without explicit FDR control.
4. **Complete-case deletion.** Combine opt-out rates of 20-35% on the agility tests will silently halve the sample.
5. **Tracking noise.** 10 Hz optical speed is noisy and the raw acceleration column is a finite difference; features must be built from smoothed signals.
6. **Survivorship and unequal exposure.** 2023 draftees have three seasons of follow-up, 2025 draftees one.
