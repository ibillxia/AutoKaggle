"""Run the official invsqrt verifier against a .kmb/.kmx circuit.

usage: run_invsqrt.py CIRCUIT [shots_per_x] [batch_size]

Full verification is 64 shots of each of the 3,133,452 inputs. Fewer shots is a much
weaker check, but it exercises the same code path and reports the same cost.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402  (sets TMP/TEMP, then puts kickmix and verifiers on sys.path)

from invsqrt import verify_submission_invsqrt as V  # noqa: E402

if __name__ == '__main__':
    km = env.import_kickmix()
    path = Path(sys.argv[1])
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    spec = V.ProblemSpecInvSqrt.for_n_p(10, shots_per_x=shots)
    circuit = km.Circuit.from_file(path, format='auto')
    print(json.dumps(V.verify(circuit, spec, batch_size=batch), indent=2))
