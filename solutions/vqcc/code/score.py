"""Score a submission.zip (or a directory) with the official verifiers, locally.

The leaderboard's score for one split is the geometric mean over the three benchmarks of
``baseline_cost / cost``, with the reference submission scoring 1.0. This script computes
the same thing from local runs.

Reduced-shot options exist only to save local time; costs are reported either way, but a
reduced run is a far weaker correctness check, so use full shots for anything you submit.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402

BASELINE = {  # the reference submission's costs (see SCORING_README / verify.py)
    'invsqrt': 2_076_822.0,
    'dqi': None,
    'dfthcsos': None,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path, help='submission .zip or a directory with the three benchmarks')
    ap.add_argument('--tasks', default='invsqrt,dqi,dfthcsos')
    ap.add_argument('--shots-per-x', type=int, default=64)
    ap.add_argument('--dqi-shots', type=int, default=102400)
    ap.add_argument('--dfthc-shots', type=int, default=64)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    root = args.submission
    if root.suffix == '.zip':
        dest = env.SCRATCH / 'unpacked'
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root) as z:
            z.extractall(dest)
        root = dest

    km = env.import_kickmix()
    results = {}
    for task in args.tasks.split(','):
        import time
        t0 = time.perf_counter()
        if task == 'invsqrt':
            from invsqrt import verify_submission_invsqrt as V
            spec = V.ProblemSpecInvSqrt.for_n_p(10, shots_per_x=args.shots_per_x)
            circuit = km.Circuit.from_file(root / 'invsqrt.kmb', format='auto')
            r = V.verify(circuit, spec, batch_size=256)
            results[task] = {'cost': r['score'], 'qubits': r['num_qubits'],
                             'toffoli_equiv': r['toffoli_equivalent'], 'status': r['status']}
        elif task == 'dqi':
            from dqi import verify_submission_dqi as V
            spec = V.ProblemSpecDQI.benchmark_instance(num_shots=args.dqi_shots)
            circuit = km.Circuit.from_file(root / 'dqi.kmb', format='auto')
            r = V.verify(circuit, spec)
            results[task] = {k: r[k] for k in r if isinstance(r[k], (int, float, str))}
            results[task]['cost'] = r['score']
        elif task == 'dfthcsos':
            from dfthcsos import verify as V
            from dfthcsos.scoreinfra.reference import FACTORIZATION_KEYS, ORBITALS
            import numpy as np
            d = root / 'dfthcsos'
            circuits = [km.Circuit.from_file(p, format='auto')
                        for p in sorted(d.glob('angle_k_*.km*'),
                                        key=lambda p: int(p.stem.split('_')[-1]))]
            with np.load(d / 'submission.npz', allow_pickle=False) as z:
                npz = {k: np.array(z[k]) for k in FACTORIZATION_KEYS}
                tree_edges = np.array(z['tree_edges'])
            r = V.verify(circuits, npz, tree_edges,
                         shots_per_address=args.dfthc_shots)
            results[task] = {k: r[k] for k in r if k != 'resources'}
        results[task]['seconds'] = round(time.perf_counter() - t0, 1)
        print(task, json.dumps(results[task], default=str), flush=True)

    out = {'results': results}
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
