"""Score a circuit directory with the official verifier.

Reads .kmb data files only; no third-party code is imported or executed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import verify  # noqa: E402

if __name__ == '__main__':
    d = Path(sys.argv[1])
    t0 = time.monotonic()
    try:
        r = verify(str(d) + '/')
        print(json.dumps({k: r[k] for k in ('status', 'score', 'toffoli_count',
                                           'num_qubits', 'lambda_eff',
                                           'verified_shots')}, indent=1))
        print('resources:', json.dumps(r['resources'], indent=1))
        print('metrics:', json.dumps(r['metrics'], indent=1))
        (d / 'verify.json').write_text(json.dumps(r, indent=1))
    except Exception as e:  # noqa: BLE001
        print(f'{type(e).__name__}: {e}')
    print(f'{time.monotonic()-t0:.0f}s')
