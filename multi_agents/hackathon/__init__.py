"""AutoKaggle hackathon mode.

AutoKaggle's original SOP assumes a tabular Kaggle competition: train.csv and
test.csv, a numeric evaluation metric, a leaderboard, and a submission file.
A large and growing share of Kaggle's highest-profile competitions -- the NFL
Big Data Bowl among them -- are instead *judged hackathons*: the submission is a
writeup plus a public notebook, scored by domain experts against a published
rubric, with hard pass/fail gates.

This subpackage provides the SOP for that second kind of competition. It reuses
AutoKaggle's multi-agent control philosophy (Reader -> Planner -> Developer ->
Reviewer -> Summarizer, with reviewer-driven iteration and per-phase retry
budgets) while replacing:

* the phases          (audit -> hypothesis -> validation -> metric -> narrative)
* the scoring         (the published rubric, weighted, instead of a metric)
* the gates           (deliverable unit tests instead of dataframe assertions)
* the developer output (executed analysis code instead of processed CSVs)

Entry point::

    python -m multi_agents.hackathon.run --competition nfl_big_data_bowl_2027
"""

from __future__ import annotations

from .state import HackathonState, load_hackathon_config, read_overview
from .sop import HackathonSOP

__all__ = ["HackathonState", "HackathonSOP", "load_hackathon_config", "read_overview"]
