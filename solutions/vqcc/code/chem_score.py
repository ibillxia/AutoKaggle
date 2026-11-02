"""Score a factorization at several angle bit widths with the official Part-2 encoder.

Wraps the starter kit's own `encode_inputs` + `evaluate_inputs`, so the numbers are the
official rounding pipeline (two-body words -> decoded X -> one-body eigenvectors -> encoded
on the same tree), not a stale shortcut. Works for any (R, B, C) shape.

Usage: python chem_score.py NPZ [--system FeMoCo76o113e] [--brots 8,10,12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401  (kickmix path + DLL directory)

KIT = Path(__file__).resolve().parent.parent / 'data' / 'starter' / 'kit' / 'starter_kit'
sys.path.insert(0, str(KIT))

from starter.task1_factorization.export_npz import read_factorization  # noqa: E402
from starter.task2_angles.encode import encode_inputs  # noqa: E402
from starter.task2_angles.trees import log_depth_tree_edges, path_tree_edges  # noqa: E402
from verifier.score import evaluate_inputs, load_inputs, score_from_metrics  # noqa: E402

TREES = {'path': path_tree_edges, 'log_depth': log_depth_tree_edges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--system', default='FeMoCo76o113e')
    ap.add_argument('--brots', default='8,10,12')
    a = ap.parse_args()

    params, _ = read_factorization(a.npz)
    U = np.asarray(params['U'])
    R, B, N = U.shape
    C = np.asarray(params['W']).shape[2]
    print(f'factorization {Path(a.npz).name}: R={R} B={B} C={C} N={N}  A=N+R*B={N + R * B}', flush=True)

    inputs = load_inputs(a.system, params)
    print(f'{"tree":10} {"brot":>5} {"error_1":>10} {"error_2":>10} {"Lambda":>10} {"E_gap":>10} {"gate":>6}')
    best = None
    for tname, tfn in TREES.items():
        edges = tfn(N)
        for brot in [int(x) for x in a.brots.split(',')]:
            words = encode_inputs(inputs, edges, brot)
            m = evaluate_inputs(inputs, words, edges, brot)
            ok = (m['error_1'] <= 0.020 and m['error_2'] <= 0.020 and m['E_gap'] > 3.914898)
            print(f'{tname:10} {brot:>5} {1e3 * m["error_1"]:>10.3f} {1e3 * m["error_2"]:>10.3f} '
                  f'{m["Lambda"]:>10.3f} {m["E_gap"]:>10.4f} {("PASS" if ok else "fail"):>6}', flush=True)
            if ok and (best is None or m['lambda_eff'] < best[0]):
                best = (m['lambda_eff'], tname, brot, m)
    if best:
        lam, tname, brot, m = best
        print(f'\nBest passing: tree={tname} brot={brot} lambda_eff={lam:.4f}')
        print(json.dumps({k: float(v) for k, v in m.items()}, indent=2))


if __name__ == '__main__':
    main()
