"""Run the official DQI verifier against a .kmb/.kmx circuit."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402

from dqi import verify_submission_dqi as V  # noqa: E402

if __name__ == '__main__':
    km = env.import_kickmix()
    path = Path(sys.argv[1])
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 102400
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    print('loading', path, flush=True)
    circuit = km.Circuit.from_file(path, format='auto')
    print('qubits', circuit.num_qubits, 'ops', len(circuit), flush=True)
    spec = V.ProblemSpecDQI.benchmark_instance(num_shots=shots)
    r = V.verify(circuit, spec, batch_size=batch)
    print(json.dumps(r, indent=2, default=str))
