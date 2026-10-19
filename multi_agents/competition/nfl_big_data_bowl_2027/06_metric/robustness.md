# Robustness

Each check below was chosen because it could plausibly have destroyed the finding, not to decorate it.

**Leave-one-out.** Removing any single prospect moves the partial correlation to between +0.35 and +0.43 (full-sample +0.39); 0 of 84 removals reverse its sign. The effect is not driven by one outlier.

**Alternative aggregation.** Every trait was recomputed as both the mean across a player's attempts and their single best attempt. The direction and significance of the effect are unchanged; the best-attempt version is slightly stronger, which is what a coach would expect from a movement-quality measure.

**Out-of-sample.** The incremental R² quoted above is from repeated 5-fold cross-validation, not from an in-sample fit, so it already pays for the extra parameter the metric costs.

## What would still break this
- If Next Gen Stats' in-game separation and get-off columns share a systematic model bias with Combine drill type, the partial correlation would be partly mechanical. The independent in-game tracking re-derivation in `game_tracking_*.csv` is the mitigation, and it is reported.
- Three draft classes cannot separate a real positional effect from a class-specific one.
- Optical tracking vendors change hardware between seasons; a systematic shift between the 2023 and 2025 Combine rigs would contaminate any trend, which is why the analysis is cross-sectional within a class rather than longitudinal.
