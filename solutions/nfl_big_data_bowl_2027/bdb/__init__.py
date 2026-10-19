"""NFL Big Data Bowl 2027 -- competition-specific solution package.

The pipeline converts 10 Hz Combine optical tracking into coach-readable
movement traits, links them to independently measured regular-season NFL
outcomes, and quantifies how much each trait adds *beyond* the traditional
stopwatch Combine numbers.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["paths", "schema", "io", "audit", "kinematics", "ingame", "feasibility"]
