"""Decompose Lambda and lambda_eff for a factorization, and measure the W freedom.

Lambda = sum|eig(h1_prime)| + lambda_h2,  lambda_h2 = 0.25 sum_rc (|w_B| + L1(w_rc))^2,
w_B = W_rc + sum_b w_b^(rho c).

W enters h1_prime (through -sum_rc W_rc X_rc), E_SOS (through -0.5*||W||^2) and
lambda_h2 (through w_B). So W is a free parameter that the public solutions may
not have tuned for the verifier's floors. This measures how much of Lambda is
each piece and what the floors would give.

Usage: python lambda_breakdown.py [SUBMISSION_NPZ]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402

E_UPPER = -22140.27335
LAM_FLOOR, GAP_FLOOR = 49.0, 3.914898


def main() -> None:
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    z = np.load(npz)
    w, W = z['W'], z['W']            # W is (R,B,C); W_rc is the separate (R,C)
    w = z['W'] if z['W'].ndim == 3 else None
    w = z['W']
    if w.ndim != 3:
        raise SystemExit('expected (R,B,C)')
    # submission.npz stores w as 'W' (R,B,C); W_rc is derived in the starter kit.
    inputs = V.load_inputs(npz)
    ref = inputs.params
    print('params keys:', sorted(ref.keys())[:20])

    w = np.asarray(z['W'], dtype=np.float64)
    R, B, C = w.shape
    # Recover the one-body shift term the starter kit uses: W_rc such that
    # w_B = W_rc + sum_b w_b. The npz does not carry W_rc separately in the
    # verifier's view, so report both the L1 term and the h1 term.
    l1 = np.linalg.norm(w, ord=1, axis=1)          # (R, C)
    s = w.sum(axis=1)                              # (R, C)
    print(f'R,B,C = {R},{B},{C}')
    print(f'L1(w) per (rho,c): min {l1.min():.6f} max {l1.max():.6f} sum {l1.sum():.4f}')
    print(f'sum_b w_b: min {s.min():.6f} max {s.max():.6f}')
    lam_h2_if_W_eq_neg_s = 0.25 * np.sum((l1 + np.abs(s - s)) ** 2)
    print(f'lambda_h2 with W_rc = -sum_b w_b  ->  0.25*sum(L1^2) = {0.25*np.sum(l1**2):.4f}')

    m = V.evaluate_words(inputs, z['angle_words'], z['tree_edges'], 10)
    print(json.dumps({k: m[k] for k in ('Lambda', 'E_gap', 'lambda_eff',
                                        'error_1', 'error_2')}, indent=1))
    lam = m['Lambda']
    print(f'Lambda = {lam:.4f}')
    for name, val in (('floor Lambda', LAM_FLOOR), ('floor E_gap', GAP_FLOOR)):
        print(f'  {name} = {val}')
    print(f"lambda_eff floor at (49.0, 3.914898) = "
          f"{math.sqrt(GAP_FLOOR*(2*LAM_FLOOR-GAP_FLOOR)):.4f}")
    print(f"  current {m['lambda_eff']:.4f} -> potential "
          f"{m['lambda_eff']/math.sqrt(GAP_FLOOR*(2*LAM_FLOOR-GAP_FLOOR)):.2f}x")


if __name__ == '__main__':
    main()
