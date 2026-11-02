"""Exact T/Q breakdown of a dfthcsos submission directory, register by register.

Runs the official verifier and dumps its metrics and resource dicts verbatim.

Usage: python shipped_breakdown.py SUBMISSION_DIR
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import verify  # noqa: E402

d = Path(sys.argv[1])
res = verify(d)
if isinstance(res, tuple) and len(res) == 2:
    metrics, resources = res
else:
    metrics, resources = None, res
print('=== metrics ===')
print(json.dumps(metrics, indent=2, default=str) if isinstance(metrics, dict) else metrics)
print('=== resources ===')
print(json.dumps(resources, indent=2, default=str) if isinstance(resources, dict) else resources)
