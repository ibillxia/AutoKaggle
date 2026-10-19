# Confirmatory analysis

## Headline
In DL, skill_dl__entry_speed_max is positively associated with sack_rate after controlling for the conventional Combine battery and draft capital (partial r = +0.39, 95% CI [+0.14, +0.44], n = 84), adding +0.080 out-of-sample R^2.

| quantity | value |
|---|---|
| partial Spearman r | 0.388 |
| bootstrap 95% CI | [0.144, 0.443] |
| n | 84 |
| bootstrap p | 0.0000 |
| baseline CV R² | -0.033 |
| baseline + trait CV R² | 0.047 |
| incremental out-of-sample R² | 0.080 |
| trait split-half reliability | n/a |

## Draft capital, position and exposure
**Draft capital.** Every model in this table conditions on log draft capital (UDFAs imputed at the maximum pick, with an explicit UDFA indicator). The reported coefficient is therefore the part of the sensor signal that survives teams' own evaluations. The same trait correlates strongly with draft position on its own, and that raw correlation is deliberately *not* what is claimed.

**Position.** The cohort is not pooled. All statistics are computed inside a single position group, so a 190 lb receiver and a 320 lb defensive tackle are never placed on a shared scale.

**Exposure.** Outcomes are per-play rates, not career totals, precisely because the 2023 draft class has three seasons of follow-up against one for 2025. Where a career-level outcome is used it is divided by seasons observed.

## Robust-regression view
```
OLS with HC3 robust SE   n=84  k=20  R2=0.354  adjR2=0.163
  intercept                    +10.1292 [-13.7904, +34.0487]  p=0.4096
  log_draft_capital            -0.2590 [-0.5347, +0.0167]  p=0.0703
  age_at_draft                 -0.1199 [-0.3492, +0.1094]  p=0.3094
  combine_height               -0.1486 [-0.3245, +0.0273]  p=0.1026
  combine_weight               -0.0099 [-0.0259, +0.0061]  p=0.2281
  forty                        +0.3478 [-3.7588, +4.4545]  p=0.8687
  ten_yd_split                 -0.4650 [-12.2045, +11.2744]  p=0.9351
  three_cone                   +0.3848 [-2.2332, +3.0028]  p=0.7742
  short_shuttle                +0.4861 [-3.1518, +4.1240]  p=0.7942
  vertical                     -0.0383 [-0.1567, +0.0800]  p=0.5279
  broad_jump                   +0.0156 [-0.0596, +0.0908]  p=0.6859
  bench_reps                   +0.0602 [-0.0282, +0.1486]  p=0.1864
  forty_not_tested             +0.1183 [-0.1858, +0.4225]  p=0.4486
  ten_yd_split_not_tested      +0.1183 [-0.1858, +0.4225]  p=0.4486
  three_cone_not_tested        -0.2400 [-1.4930, +1.0130]  p=0.7086
  short_shuttle_not_tested     -0.2402 [-1.2795, +0.7991]  p=0.6521
  vertical_not_tested          -0.6116 [-2.0608, +0.8376]  p=0.4112
  broad_jump_not_tested        +0.1314 [-1.3331, +1.5959]  p=0.8610
  bench_reps_not_tested        +0.4404 [-0.1543, +1.0351]  p=0.1515
  __trait_z                    +0.3655 [+0.0978, +0.6332]  p=0.0094
```

## Does the in-game data actually reward the metric?
the in-game outcome favours the sensor's preferred quadrant

Mean outcome by archetype: `{'confirmed_elite': 0.010186512364211853, 'confirmed_limited': 0.008536066631661827, 'sensor_beats_stopwatch': 0.0117562522094829, 'stopwatch_beats_sensor': 0.006350914056780171}` (lower is better: False).

Prospects the sensor likes and the stopwatch does not:

| player | overall pick | sensor z | stopwatch z | in-game outcome |
|---|---|---|---|---|
| Marshawn Kneeland | 56.0 | 1.84 | -1.51 | 0.00 |
| Lukas Van Ness | 13.0 | 0.85 | -1.51 | 0.02 |
| Isaiah Land | None | 0.60 | -0.90 | 0.02 |
| Jah Joyner | None | 0.97 | -0.46 | 0.00 |
| Jalen Redmond | None | 0.73 | -0.56 | 0.01 |

## Uncertainty is the point
The headline is quoted as an interval, not a point, and the interval is the bootstrap one rather than a normal-theory approximation, because with n in the tens the asymptotic approximation is not trustworthy. The bootstrap p-value is reported alongside so a reader can see how often a resample landed on the wrong side of zero.

## Still uncertain
- The metric's own partial correlation is 0.38 ([0.13, 0.59]); it is a composite and inherits the missingness of its components.
- Whether the effect is stable across draft classes cannot be settled with three classes; the clustered bootstrap widens the interval accordingly.
