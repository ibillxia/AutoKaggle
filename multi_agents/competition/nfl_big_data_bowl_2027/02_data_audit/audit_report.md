# Data audit

The purpose of this phase is to establish what the data can support before any hypothesis is tested, so that topic selection is constrained by evidence rather than by narrative appeal.

## Join integrity
| relationship | child keys | orphan keys | coverage |
|---|---|---|---|
| combine_results -> players | 510 | 0 | 100.00% |
| career_successes -> players | 510 | 0 | 100.00% |
| combine_tracking -> players | 510 | 0 | 100.00% |

`players.nfl_id` unique: True. `combine_results.nfl_id` unique: True. `player_play` (game_id, play_id, nfl_id) unique: True.

## Effective sample size -- the binding constraint
The published table sizes (510 prospects) overstate what can be estimated. Inference happens inside position groups, so the real N is:

| position group | n = prospects |
|---|---|
| WR | n = 106 |
| TE | n = 42 |
| DB | n = 118 |
| OL | n = 121 |
| DL | n = 88 |

Every analysis in this submission is therefore position-stratified, and every reported effect carries its own sample size and confidence interval. With n in the tens per group, the minimum reportable cell is set at n = 25 complete cases, and no model more complex than a regularised linear one can be validated.

## Sensor coverage
6310 tracked drill attempts across 510 players (0 prospects have no tracking at all). Frames per attempt: {'mean': 73.4, 'median': 67.0, 'p05': 42.0, 'p95': 131.0}.

Which drills were captured for which group matters: a trait can only be computed for a position group that actually ran the drill.

## Missingness and drill opt-outs
| test | opt-out rate |
|---|---|
| forty | 16.9% |
| ten_yd_split | 16.7% |
| three_cone | 64.7% |
| short_shuttle | 60.6% |
| vertical | 14.1% |
| broad_jump | 17.1% |
| bench_reps | 62.4% |

In-game outcome availability (fraction of snaps on which the vendor column is populated):

| column | availability |
|---|---|
| separation_at_pass_forward | 14.9% |
| cushion | 33.0% |
| player_get_off | 14.0% |
| time_to_pressure | 1.2% |
| time_to_qb_hurry | 1.2% |
| peak_pressure_probability_allowed | 18.5% |
| time_to_pressure_allowed | 1.6% |
| dropback_duration | 19.4% |
| pass_rushers_encountered | 19.4% |
| expected_yards_after_catch | 2.9% |
| yards_after_catch | 2.0% |
| pass_length | 58.8% |
| expected_points_added | 100.0% |
| target | 100.0% |
| sack | 27.6% |
| tackle | 3.8% |

## Exposure and survivorship
| draft year | mean career games active | mean career offensive snaps |
|---|---|---|
| 2023 | 33.06 | 537.81 |
| 2024 | 21.94 | 306.47 |
| 2025 | 11.01 | 154.26 |

The 2023 class has roughly three times the follow-up of the 2025 class, so career *totals* are not comparable across draft years. All outcomes are expressed per play or per season.

Undrafted free agents in cohort: 127 (24.9% of the cohort). Draft capital is therefore imputed for UDFAs rather than dropping them, and a UDFA indicator is carried through every model.

## What this constrains
Draft capital is a first-class control; position is a stratifier; exposure is normalised; missingness is imputed with explicit indicators; and the minimum reportable cell is 25 complete cases. Any hypothesis that cannot survive those constraints is not worth pursuing in a 2,000-word writeup.
