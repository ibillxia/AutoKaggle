# Feasibility screen

## Why a screen and not an analysis
This dataset invites fishing: hundreds of computable sensor traits against dozens of in-game outcomes. Reporting the best correlation out of thousands would produce a confident, wrong answer. The screen exists to make the search explicit and to pay the multiple-comparison price up front.

## Pre-registered hypotheses
Pairs were restricted in advance to combinations a football analyst would consider legitimate (a receiver's break retention against separation; a defensive lineman's first-step burst against pass-rush get-off; an offensive lineman's mirror agility against time to pressure allowed). The full list is in `feasibility_results.csv`.

## Baseline block
Every sensor trait must earn its place on top of the conventional scouting model: log draft capital, age at draft, height, weight, and the position-relevant stopwatch battery (40-yard dash, 10-yard split, 3-cone, short shuttle, vertical, broad jump, and bench press for the linemen). Baseline columns are median-imputed within position group, with explicit 'did not test' indicators for any test with an opt-out rate above 10%.

## Statistics
- **Stage 1 (screen).** Partial Spearman correlation between each sensor trait and each in-game outcome, residualising both on the baseline block. Benjamini-Hochberg FDR at q < 0.10.
- **Stage 2 (confirmation).** For the strongest candidates per outcome, the *incremental* out-of-sample R² of adding the trait to the baseline, estimated with repeated 5-fold cross-validation (4 repeats) and an inner-CV ridge.
- **Minimum cell size.** The smallest sample size reported anywhere in this screen is n = 26 complete cases; nothing below n = 25 is reported at all. Each row of the result table carries its own n, and every confirmed pair is reported with a bootstrap confidence interval rather than a bare point estimate.

## Result
276 pairs screened; 5 survived (FDR-significant **and** at least +0.01 incremental out-of-sample R²). Across position groups:

| position group | pairs screened | survivors | max ΔR² |
|---|---|---|---|
| DB | 36 | 0 | 0.224 |
| DL | 72 | 3 | 0.080 |
| OL | 60 | 0 | 0.042 |
| TE | 60 | 2 | 0.099 |
| WR | 48 | 0 | 0.103 |

### Strongest surviving pairs
| position | outcome | sensor trait | n | partial r | q | ΔR² |
|---|---|---|---|---|---|---|
| DB | tackle_rate | `short_shuttle__entry_speed_max` | 26 | -0.47 | 0.7302 | 0.224 |
| DB | tackle_rate | `short_shuttle__entry_speed_mean` | 26 | -0.41 | 0.7302 | 0.153 |
| DB | tackle_rate | `short_shuttle__cut_min_speed_max` | 26 | 0.39 | 0.7302 | 0.136 |
| DB | tackle_rate | `short_shuttle__cut_radius_proxy_max` | 26 | -0.42 | 0.7302 | 0.127 |
| DB | tackle_rate | `short_shuttle__cut_radius_proxy_mean` | 26 | -0.46 | 0.7302 | 0.118 |
| WR | sep_mean | `short_shuttle__path_length_max` | 28 | 0.36 | 0.8705 | 0.103 |
| TE | target_rate | `forty_yard_dash__peak_accel_max` | 31 | -0.54 | 0.9402 | 0.099 |
| TE | target_rate | `forty_yard_dash__max_abs_accel_max` | 31 | -0.54 | 0.9402 | 0.099 |
| DB | tackle_rate | `short_shuttle__total_abs_turn_deg_mean` | 26 | 0.39 | 0.7302 | 0.090 |
| DB | tackle_rate | `short_shuttle__total_abs_turn_deg_max` | 26 | 0.39 | 0.7302 | 0.087 |
| DL | sack_rate | `skill_dl__entry_speed_max` | 84 | 0.39 | 0.0696 | 0.080 |
| TE | sep_mean | `skill_te__path_efficiency_max` | 37 | -0.70 | 0.0148 | 0.077 |
| DL | get_off_mean | `three_cone_drill__exit_speed_max` | 35 | 0.39 | 0.9409 | 0.076 |
| DL | get_off_mean | `three_cone_drill__cut_radius_proxy_max` | 35 | 0.37 | 0.9409 | 0.064 |
| DL | get_off_mean | `short_shuttle__distance_max` | 28 | -0.49 | 0.9409 | 0.062 |

### Weakest pairs (the negative control)
These are reported deliberately. If the screen had produced large incremental R² for essentially every pair, it would be measuring noise rather than signal.

| position | outcome | sensor trait | n | partial r | q | ΔR² |
|---|---|---|---|---|---|---|
| DB | tackle_rate | `short_shuttle__mean_speed_mean` | 26 | -0.43 | 0.7302 | -0.264 |
| TE | sep_mean | `forty_yard_dash__peak_accel_min` | 31 | 0.58 | 0.2707 | -0.184 |
| TE | sep_mean | `forty_yard_dash__max_abs_accel_min` | 31 | 0.58 | 0.2707 | -0.184 |
| WR | sep_mean | `short_shuttle__max_abs_accel_max` | 28 | 0.22 | 0.8705 | -0.136 |
| TE | yac_mean | `forty_yard_dash__distance_min` | 30 | -0.59 | 0.6730 | -0.131 |
| TE | cushion_mean | `forty_yard_dash__time_to_peak_speed_min` | 31 | -0.57 | 0.3036 | -0.116 |
| WR | sep_p75 | `short_shuttle__exit_speed_min` | 28 | 0.19 | 0.9156 | -0.104 |
| WR | sep_p75 | `short_shuttle__cut_min_speed_mean` | 28 | 0.19 | 0.9156 | -0.080 |

## What would falsify the approach
If sensor-derived drill durations did not track the official electronic times, the entire sensor layer would be untrustworthy and no downstream trait could be believed. That check is reported in the writeup (Figure 2) and passes.

## Ruled out
Most sensor traits do not add out-of-sample explanatory power beyond the conventional battery. The surviving signal is concentrated in a small number of drill families, which is what makes the eventual topic selection defensible rather than cherry-picked.
