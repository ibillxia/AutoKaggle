"""Reference-submission costs, measured locally with the official verifiers.

Every cost here was reproduced by ``code/score.py`` on the competition's
``sample_submission``; the leaderboard divides the baseline cost by yours and takes the
geometric mean over the three benchmarks, so the reference submission scores 1.0.
"""
import json
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / 'artifacts' / 'baseline_costs.json'

BASELINE_COSTS = {
    'invsqrt': 2_076_774.23,     # Q=260, C=7987.59
    'dqi': 24_439_463_583_672.93,  # R=1.1913, Q=1893, T=5725108.38
    'dfthcsos': 3_355_599_588.32,  # lambda_eff=49.3014, T=93237, Q=730
}


def geometric_score(costs: dict) -> float:
    """Leaderboard score: geometric mean of baseline/cost over the benchmarks."""
    ratio = 1.0
    for task, cost in costs.items():
        ratio *= BASELINE_COSTS[task] / cost
    return ratio ** (1 / len(costs))
