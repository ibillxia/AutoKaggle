# Analysis plan

## Question
For DL, does the selected sensor trait predict sack_rate over and above the conventional Combine battery and draft capital?

## Model
Partially-linear design: both the sensor trait and the in-game outcome are rank-transformed and residualised on the baseline block, then correlated. The effect size is reported in outcome standard deviations from an OLS fit with HC3 heteroskedasticity-robust standard errors. A ridge model with repeated 5-fold cross-validation supplies the incremental out-of-sample R².

## Uncertainty
Percentile bootstrap confidence intervals (4,000 resamples). Resampling is clustered by draft class, because players drafted in the same year share a season, a scheme environment and a coaching staff.

## Confounds controlled
Draft capital (log-imputed for UDFAs, with a UDFA indicator); age at draft; height and weight; the full position-relevant stopwatch battery, median-imputed with 'did not test' indicators; position held fixed by stratification; outcomes expressed per play or per season so exposure cannot leak into the result.

## Robustness checks
1. Leave-one-out: recompute the effect with each prospect removed in turn and report the range and the number of sign flips.
2. Split-half reliability of the trait across the player's own repeated Combine attempts (odd vs even attempt), Spearman-Brown corrected.
3. Alternative aggregation: mean across attempts vs single best attempt.
4. Alternative outcome definitions where more than one is defensible.
5. Out-of-sample incremental R² rather than in-sample fit.

## Falsification criteria
The topic would be abandoned if any of the following held:
- the bootstrap confidence interval for the partial correlation contained zero;
- the incremental out-of-sample R² were zero or negative, i.e. the trait only re-expressed information already in the stopwatch battery;
- leave-one-out showed the effect was carried by a single prospect;
- sensor-derived durations failed to track the official electronic times, which would invalidate the sensor layer itself;
- the in-game outcome did not, on average, favour the quadrant the sensor preferred -- which would mean the metric ranks players the NFL does not reward.
