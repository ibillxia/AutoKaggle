# Cut Entry Speed Index

## Definition
```
Cut Entry Speed Index_i = mean_over_cuts( z(exit_speed / entry_speed) )
            - z( seconds to regain 90% of entry speed )
```

where `entry_speed` is the mean smoothed speed in the 0.2 s before the sharpest detected cut of an attempt, `exit_speed` is the mean smoothed speed in the 0.2 s immediately after it, the cut is the contiguous window whose angular velocity exceeds 60 deg/s and whose total signed rotation exceeds 25 degrees, and `z()` standardises within the position group.

## Units
The metric is a **position-standardised z-score** (dimensionless, mean 0 and standard deviation 1 within a position group). Its raw ingredients are dimensionless anyway: the retention term is a speed *ratio*, and the regain term is in **seconds**. Both are standardised before averaging, so the composite is a relative scouting grade, not an absolute physical measurement.

## Direction
**Higher is better.** Retaining more of your speed through a break is good; taking longer to climb back to speed is bad, which is why the regain term enters with a negative sign.

## Interpretation
A player at +1 has a better break-retention profile than about 84% of his position group. The metric is intended to be read *alongside* the stopwatch, never instead of it.

## Validation
- Partial correlation with the target in-game outcome after the conventional battery and draft capital: 0.382 ([0.135, 0.589], n = 84.0).
- Incremental out-of-sample R²: 0.075 (baseline -0.033 → full 0.042).
- Top-10 rank stability under bootstrap resampling (mean Jaccard overlap): 0.37.
- Versus the single best traditional test (`vertical`, R² = -0.075): the metric alone reaches R² = 0.034, a difference of 0.109.

That last line is reported against our own interest: the metric is **not** a standalone replacement for the stopwatch. It is an addition that only pays off inside a model that already contains the conventional numbers.

## Weights
Components enter with equal weight. The weights were deliberately **not** fitted to the outcome, which would make the metric circular -- it would be guaranteed to predict the outcome it was trained on, and would generalise to nothing.
