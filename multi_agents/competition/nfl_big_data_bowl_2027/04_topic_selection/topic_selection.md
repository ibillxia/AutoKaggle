# Topic selection

## Selected
**DL -- sack_rate -- `skill_dl__entry_speed_max`** (drill family `SKILL_`).

- n = 84
- partial Spearman r = 0.39
- q = 0.0696
- incremental out-of-sample R² = 0.080
- supported by 3 correlated traits in the same drill family: `skill_dl__entry_speed_max`, `skill_dl__cut_radius_proxy_max`, `skill_dl__cut_min_speed_max`

## How the candidates were ranked
| axis | how it was judged |
|---|---|
| Evidence strength | incremental out-of-sample R², FDR q-value, and the number of independent traits in the same drill family that agree |
| Novelty | whether the quantity is already expressible from `combine_results.csv` -- a trait that is a restatement of the 40-yard dash is not a finding |
| Actionability | whether a coach, scout or front office could compute it and change a decision with it |
| Communicability | whether the argument fits in under 10 figures and 2,000 words |

## Why this survives the novelty test
The conventional Combine measures how *long* a drill took. This trait measures what happened *during* the drill: how much speed survived the sharpest cut and how quickly the athlete climbed back out of it. Those quantities do not exist in `combine_results.csv` at any precision, because a stopwatch reports a total and a total is not decomposable. The test of novelty is empirical, not rhetorical: the trait adds out-of-sample R² on top of a baseline that already contains every traditional test.

## Why this survives the actionability test
A position coach can compute it from data the league already collects, and it produces a per-player number that can be trained. The most useful output is not the ranking itself but the disagreements it exposes between the stopwatch and the sensor, which is the list a front office can act on in draft week.

## Accepted alternatives (runner-up candidates)
| position | outcome | sensor trait | n | partial r | ΔR² |
|---|---|---|---|---|---|
| DL | sack_rate | `skill_dl__entry_speed_max` | 84 | 0.39 | 0.080 |
| DL | sack_rate | `skill_dl__cut_radius_proxy_max` | 84 | 0.36 | 0.058 |
| DL | sack_rate | `skill_dl__cut_min_speed_max` | 84 | 0.35 | 0.050 |

## Rejected: the weakest pairs
Recorded so that the search is auditable and so that the negative space is visible to a reader who suspects cherry-picking.

| position | outcome | sensor trait | n | partial r | ΔR² |
|---|---|---|---|---|---|
| DB | tackle_rate | `short_shuttle__mean_speed_mean` | 26 | -0.43 | -0.264 |
| TE | sep_mean | `forty_yard_dash__peak_accel_min` | 31 | 0.58 | -0.184 |
| TE | sep_mean | `forty_yard_dash__max_abs_accel_min` | 31 | 0.58 | -0.184 |
| WR | sep_mean | `short_shuttle__max_abs_accel_max` | 28 | 0.22 | -0.136 |
| TE | yac_mean | `forty_yard_dash__distance_min` | 30 | -0.59 | -0.131 |

## Search accounting
276 pre-registered pairs were screened; 3 survived. The selected topic is the strongest survivor, not the most narratively appealing one.
